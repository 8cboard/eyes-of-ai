# remote_server/server.py — VisionTracker Remote LLM Server
#
# FastAPI server for single-object visual identification.
# Supports GGUF (llama-cpp-python) and Safetensors (transformers) VLMs.
#
# Recommended model: Qwen2-VL-7B-Instruct (state-of-the-art, <10 GB)
#   Main GGUF:  bartowski/Qwen2-VL-7B-Instruct-GGUF
#                 Qwen2-VL-7B-Instruct-Q4_K_M.gguf  (~4.5 GB)
#   mmproj:     mmproj-Qwen2-VL-7B-Instruct-f16.gguf (~1.0 GB)
#   chat_format: qwen2-vl
#
# Alternative: LLaVA-1.6-Mistral-7B (very stable with llama-cpp)
#   Main GGUF:  cjpais/llava-1.6-mistral-7b-gguf
#                 llava-1.6-mistral-7b.Q4_K_M.gguf   (~4.4 GB)
#   mmproj:     mmproj-model-f16.gguf                (~631 MB)
#   chat_format: llava-1-6
#
# Quick start:
#   # Colab / Kaggle — install with CUDA wheels
#   pip install llama-cpp-python \
#     --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
#
#   # Start server
#   python server.py \
#     --model-path  /path/to/Qwen2-VL-7B-Instruct-Q4_K_M.gguf \
#     --mmproj-path /path/to/mmproj-Qwen2-VL-7B-Instruct-f16.gguf

from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn

app = FastAPI(title="VisionTracker Remote ID Server", version="3.1.0")

# Limit request body to 10 MB — a 800px JPEG is well under 500 KB,
# so this guards against accidental giant payloads without being restrictive.
_MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MB


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_BODY_BYTES:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request body too large (max {_MAX_BODY_BYTES // 1024 // 1024} MB)"},
        )
    return await call_next(request)


# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────

_model = None
_model_type = None  # 'gguf' | 'safetensors'
_model_name = "unknown"
_MAX_GB = 10.0

# Executor for running synchronous inference without blocking the event loop
_executor = None  # set to ThreadPoolExecutor in main()


# ─────────────────────────────────────────────────────────────────────────────
# Request / response models
# ─────────────────────────────────────────────────────────────────────────────

class IdentifyRequest(BaseModel):
    annotated_image: str  # base64 JPEG — full frame with ONE green box drawn


class IdentifyResponse(BaseModel):
    result: str  # single common noun, e.g. "person", "car"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────────────────────────────────────


def _model_size_gb(path: str) -> float:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / (1024 ** 3)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 ** 3)
    return 0.0


def _detect_format(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file() and p.suffix.lower() == ".gguf":
        return "gguf"
    if p.is_dir():
        if any(p.glob("*.safetensors")) or (p / "pytorch_model.bin").exists():
            return "safetensors"
    return None


def _auto_chat_format(model_stem: str) -> Optional[str]:
    """Guess the llama-cpp chat_format from the model filename stem."""
    s = model_stem.lower()
    if "qwen2-vl" in s or "qwen2vl" in s:
        return "qwen2-vl"
    if "llava-1.6" in s or "llava-1-6" in s or "llava1.6" in s:
        return "chatml"
    if "llava" in s:
        return "chatml"
    if "minicpm" in s:
        return "minicpm-v"
    if "phi-3" in s or "phi3" in s:
        return "chatml"
    if "mistral" in s or "mixtral" in s:
        return "mistral-instruct"
    return None  # let llama-cpp auto-detect


def load_gguf_model(
    model_path: str,
    mmproj_path: Optional[str] = None,
    chat_format: Optional[str] = None,
    n_ctx: int = 4096,
) -> None:
    """Load a GGUF vision-language model via llama-cpp-python."""
    global _model, _model_type, _model_name

    from llama_cpp import Llama

    print(f"[Server] Loading GGUF model: {model_path}")
    print(f"[Server] Model size: {_model_size_gb(model_path):.2f} GB")

    kwargs: dict = dict(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=-1,  # offload all layers to GPU
        verbose=False,
    )

    if mmproj_path:
        if not Path(mmproj_path).exists():
            raise FileNotFoundError(f"mmproj file not found: {mmproj_path}")
        kwargs["clip_model_path"] = mmproj_path
        print(f"[Server] Vision encoder (mmproj): {mmproj_path}")
        print(f"[Server] mmproj size: {_model_size_gb(mmproj_path):.2f} GB")
    else:
        print(
            "[Server] WARNING: --mmproj-path not provided.\n"
            "         Vision models (LLaVA, Qwen2-VL, etc.) need a separate\n"
            "         vision-encoder file to process images.\n"
            "         Without it the model is TEXT-ONLY and cannot see images.\n"
            "         Download the matching mmproj-*.gguf from HuggingFace."
        )

    resolved_fmt = chat_format or _auto_chat_format(Path(model_path).stem)
    if resolved_fmt:
        kwargs["chat_format"] = resolved_fmt
        print(f"[Server] chat_format: {resolved_fmt}")
    else:
        print("[Server] chat_format: auto-detect (llama-cpp default)")

    _model = Llama(**kwargs)
    _model_type = "gguf"
    _model_name = Path(model_path).stem
    print(f"[Server] ✓ GGUF model loaded: {_model_name}")


def load_safetensors_model(model_path: str) -> None:
    """Load a HuggingFace safetensors VLM via transformers."""
    global _model, _model_type, _model_name

    import torch
    from transformers import AutoProcessor

    print(f"[Server] Loading Safetensors model: {model_path}")
    print(f"[Server] Model size: {_model_size_gb(model_path):.2f} GB")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[Server] Device: {device} | dtype: {dtype}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model = None
    tried = []
    for cls_name in [
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
        "AutoModel",
    ]:
        try:
            import transformers
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            print(f"[Server] Loaded with {cls_name}")
            break
        except Exception as exc:
            tried.append(f"{cls_name}: {exc}")

    if model is None:
        raise RuntimeError(
            f"Could not load model from {model_path}.\nAttempts:\n"
            + "\n".join(tried)
        )

    if device == "cpu":
        model = model.to(device)

    _model = {"model": model, "processor": processor, "device": device}
    _model_type = "safetensors"
    _model_name = Path(model_path).name
    print(f"[Server] ✓ Safetensors model loaded: {_model_name}")


def load_model(
    model_path: str,
    mmproj_path: Optional[str] = None,
    chat_format: Optional[str] = None,
) -> None:
    """Auto-detect model format and load accordingly."""
    size = _model_size_gb(model_path)
    if mmproj_path:
        size += _model_size_gb(mmproj_path)

    if size > _MAX_GB:
        raise ValueError(
            f"Combined model size ({size:.2f} GB) exceeds limit of {_MAX_GB} GB"
        )

    fmt = _detect_format(model_path)
    if fmt is None:
        raise ValueError(
            f"Cannot detect model format for: {model_path}\n"
            "Expected a .gguf file or a directory with *.safetensors files."
        )

    if fmt == "gguf":
        load_gguf_model(model_path, mmproj_path=mmproj_path, chat_format=chat_format)
    else:
        load_safetensors_model(model_path)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT = (
    "There is a green bounding box drawn around exactly one object in this image.\n"
    "What is the object inside the green box?\n\n"
    "Answer with ONLY one short noun phrase (1–3 words). "
    "Use everyday words like: person, car, dog, chair, laptop, bottle, cup, phone, book.\n"
    "Do NOT explain. Do NOT say 'I see'. Output the noun only."
)


def _identify_gguf(image: Image.Image) -> str:
    """Robust GGUF identify: try two prompt formats and defensive parsing."""
    img_buf = io.BytesIO()
    image.save(img_buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(img_buf.getvalue()).decode()

    STOP_TOKENS = ["</s>", "\n", "<|im_end|>", "<|endoftext|>"]

    def _call_model(messages, max_tokens=40):
        try:
            resp = _model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                stop=STOP_TOKENS,
            )
        except Exception as exc:
            print("[_identify_gguf] create_chat_completion raised:", exc)
            raise
        # guard: if not dict, wrap it so parsing code doesn't blow up
        if not isinstance(resp, dict):
            print("[_identify_gguf] Non-dict raw response from create_chat_completion:")
            print(resp)
            resp = {"choices": [{"message": {"content": str(resp)}}]}
        return resp

    def _extract_raw(resp):
        """Extract a plain string from various llama-cpp response shapes."""
        try:
            choice0 = resp.get("choices", [{}])[0]
            # support both 'message' dict and older 'text' placements
            if isinstance(choice0, dict) and "message" in choice0:
                message = choice0.get("message", {}) or {}
                content = message.get("content", "")
            else:
                content = choice0.get("text", "") if isinstance(choice0, dict) else ""
            raw_text = ""
            if isinstance(content, str):
                raw_text = content
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        t = part.get("text") or part.get("content")
                        if t:
                            parts.append(str(t))
                    elif isinstance(part, str):
                        parts.append(part)
                raw_text = " ".join(parts)
            elif isinstance(content, dict):
                raw_text = content.get("text") or content.get("content") or str(content)
            else:
                raw_text = str(content)
            return raw_text.strip()
        except Exception:
            print("[_identify_gguf] Failed to extract text from resp (full resp below):")
            print(resp)
            return ""

    def _is_control_or_garbage(s: str) -> bool:
        if not s:
            return True
        s0 = s.strip().lower()
        # common control tokens / roles we want to reject
        if re.fullmatch(r"[<]*im_[a-z0-9_<>]*[>]*", s0):
            return True
        if s0 in ("system", "assistant", "user"):
            return True
        if s0 in ("im_startstream", "im_endstream", "im_start", "im_end", "image_start", "image_end"):
            return True
        # also reject purely numeric or extremely long garbage
        if re.fullmatch(r"^\s*\d+\s*$", s0):
            return True
        return False

    # Strategy 1: simple text message embedding a data URI (works for chatml/chat-format handlers)
    messages1 = [
        {
            "role": "user",
            "content": f"{_PROMPT}\n[data:image/jpeg;base64,{img_b64}]"
        }
    ]

    try:
        resp1 = _call_model(messages1, max_tokens=30)
    except Exception:
        # bubble up so identify() logs the traceback
        raise

    raw1 = _extract_raw(resp1)
    print("[_identify_gguf] raw resp (strategy1):", repr(raw1))
    if not _is_control_or_garbage(raw1):
        return _parse_noun(raw1)

    # Strategy 2: explicit fallback — label the blob as a Base64 image block (different parser heuristics)
    messages2 = [
        {
            "role": "user",
            "content": (
                f"{_PROMPT}\n"
                "The image is provided below as base64. Please answer with ONLY one noun (1-3 words).\n\n"
                "ImageBase64:\n"
                f"{img_b64}"
            )
        }
    ]

    try:
        resp2 = _call_model(messages2, max_tokens=60)
    except Exception:
        raise

    raw2 = _extract_raw(resp2)
    print("[_identify_gguf] raw resp (strategy2):", repr(raw2))
    if not _is_control_or_garbage(raw2):
        return _parse_noun(raw2)

    # Log full responses for diagnosis and return generic fallback
    print("[_identify_gguf] Both strategies returned empty/control output. Full resp1 / resp2 below:")
    print(resp1)
    print(resp2)
    return "object"


def _identify_safetensors(image: Image.Image) -> str:
    import torch
    model = _model["model"]
    processor = _model["processor"]
    device = _model["device"]

    try:
        inputs = processor(
            text=_PROMPT, images=image, return_tensors="pt"
        )
    except TypeError:
        inputs = processor(images=image, text=_PROMPT, return_tensors="pt")

    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=False,
        )

    raw = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    if _PROMPT[:20] in raw:
        raw = raw.split(_PROMPT[-20:])[-1].strip()
    return _parse_noun(raw)


def _parse_noun(text: str) -> str:
    """Normalise raw LLM output to a clean 1–3 word noun phrase.

    Strips common control / image stream tokens (e.g. <im_startstream>, im_startstream)
    and falls back to 'object' when nothing sensible remains.
    """
    if not isinstance(text, str):
        text = str(text)

    # basic normalisation
    text = text.strip().lower()
    text = text.strip("\"'`")
    text = text.split("\n")[0].strip()

    # Remove angle-bracketed image/control tokens like: <im_startstream>, </im_startstream>
    text = re.sub(r"<\/?im_[^>\s]*>", "", text)

    # Remove standalone control tokens like: im_startstream, im_end, im_start, im_endstream
    text = re.sub(r"\bim_[a-z0-9_]+\b", "", text)

    # Also remove any stray tokens that look like an image marker (common variants)
    text = re.sub(r"\b(?:image_start|image_end|start_stream|end_stream)\b", "", text)

    # Trim again and strip trailing punctuation
    text = text.strip()
    text = re.sub(r"[.!?]+$", "", text).strip()

    # Remove any remaining non-word punctuation (allow hyphen)
    text = re.sub(r"[^\w\s\-]", "", text).strip()

    # If nothing meaningful left, return generic fallback
    if not text:
        return "object"

    # Some models still return short control-like words; guard against those
    if len(text) <= 20 and re.match(r"^(im_|image|start|end|<.*>$)", text):
        return "object"

    # Remove long non-word garbage and cap words at 3
    words = text.split()
    if not words or len(words) > 4:
        return "object"
    return " ".join(words[:3])


# ─────────────────────────────────────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy" if _model is not None else "no_model",
        "model": _model_name,
        "type": _model_type,
        "version": "3.1.0",
    }


@app.post("/identify", response_model=IdentifyResponse)
async def identify(
    req: IdentifyRequest,
    authorization: Optional[str] = Header(None),
):
    """Identify the single object framed by the green bounding box."""
    server_key = os.environ.get("SERVER_API_KEY")
    if server_key and authorization != f"Bearer {server_key}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        img_bytes = base64.b64decode(req.annotated_image)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}")
    
    # FIX: VLM inference is synchronous and CPU/GPU-bound.  Running it
    # directly inside an async def blocks the entire uvicorn event loop,
    # preventing health-check pings and queue-depth requests from being
    # served while inference is in progress.  Offloading to the executor
    # lets the event loop keep ticking during long inference calls.
    loop = asyncio.get_event_loop()
    try:
        if _model_type == "gguf":
            result = await loop.run_in_executor(_executor, _identify_gguf, image)
        elif _model_type == "safetensors":
            result = await loop.run_in_executor(_executor, _identify_safetensors, image)
        else:
            raise HTTPException(status_code=500, detail="Unknown model type")
    except Exception:
        # Print the full traceback from the worker for debugging
        print("[identify] Inference exception traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal inference error (see server logs)")

    return IdentifyResponse(result=result)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    from concurrent.futures import ThreadPoolExecutor

    p = argparse.ArgumentParser(
        description=(
            "VisionTracker Remote LLM Server\n\n"
            "Recommended model (best quality, <10 GB):\n"
            "  Qwen2-VL-7B-Instruct-Q4_K_M.gguf  (~4.5 GB)\n"
            "  mmproj-Qwen2-VL-7B-Instruct-f16.gguf (~1 GB)\n"
            "  Download from: https://huggingface.co/bartowski/Qwen2-VL-7B-Instruct-GGUF\n\n"
            "Alternative (most stable with llama-cpp):\n"
            "  llava-1.6-mistral-7b.Q4_K_M.gguf   (~4.4 GB)\n"
            "  mmproj-model-f16.gguf               (~631 MB)\n"
            "  Download from: https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf\n\n"
            "Install llama-cpp-python with CUDA 12.1:\n"
            "  pip install llama-cpp-python \\\n"
            "    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-path", required=True,
                   help="Path to .gguf file or safetensors model directory")
    p.add_argument("--mmproj-path", default=None,
                   help="Path to vision-encoder mmproj .gguf file (required for "
                        "LLaVA / Qwen2-VL and most other GGUF VLMs)")
    p.add_argument("--chat-format", default=None,
                   help="llama-cpp chat format override (auto-detected if omitted)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=None,
                   help="Optional bearer token for /identify endpoint")
    p.add_argument("--n-ctx", type=int, default=4096,
                   help="Context length for GGUF models (default: 4096)")
    args = p.parse_args()

    if args.api_key:
        os.environ["SERVER_API_KEY"] = args.api_key

    # One worker: VRAM/shared memory can't safely run two inferences in parallel.
    global _executor
    _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-worker")

    print("=" * 62)
    print("  VisionTracker Remote LLM Server  v3.1.0")
    print("=" * 62)

    load_model(
        model_path=args.model_path,
        mmproj_path=args.mmproj_path,
        chat_format=args.chat_format,
    )

    print("=" * 62)
    print(f"  Listening on http://{args.host}:{args.port}")
    if os.environ.get("SERVER_API_KEY"):
        print("  API key authentication: enabled")
    print("=" * 62)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
