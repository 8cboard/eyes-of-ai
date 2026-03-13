"""
Remote LLM Server for VisionTracker

FastAPI server supporting both GGUF (llama-cpp-python) and Safetensors
(transformers) vision-language models.

Auto-detects model format:
- GGUF: File ends with .gguf extension
- Safetensors: Directory contains model.safetensors or pytorch_model.bin

Model size limit: <10GB
"""

import base64
import io
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from pydantic import BaseModel

# FastAPI imports
from fastapi import FastAPI, HTTPException, Header
import uvicorn

app = FastAPI(title="VisionTracker Remote ID Server", version="2.0.0")

# Global model instance
_model = None
_model_type = None  # 'gguf' or 'safetensors'
_model_name = "unknown"
_MAX_MODEL_SIZE_GB = 10


class IdentifyRequest(BaseModel):
    annotated_image: str  # base64 JPEG
    color_map: dict[str, int]  # {"red": 1, "blue": 2, ...}


class IdentifyResponse(BaseModel):
    results: list[dict]  # [{"track_id": int, "description": str}]


def _check_model_size(model_path: str) -> float:
    """Check total size of model files in GB."""
    path = Path(model_path)
    if path.is_file():
        return path.stat().st_size / (1024**3)
    elif path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024**3)
    return 0.0


def _detect_model_type(model_path: str) -> Optional[str]:
    """Auto-detect if model is GGUF or safetensors format."""
    path = Path(model_path)

    if not path.exists():
        return None

    # Check for GGUF
    if path.is_file() and path.suffix == ".gguf":
        return "gguf"

    # Check for safetensors in directory
    if path.is_dir():
        if (path / "model.safetensors").exists():
            return "safetensors"
        if (path / "pytorch_model.bin").exists():
            return "safetensors"
        # Check for .safetensors files
        if list(path.glob("*.safetensors")):
            return "safetensors"

    return None


def load_gguf_model(model_path: str):
    """Load GGUF model using llama-cpp-python."""
    global _model, _model_type, _model_name

    from llama_cpp import Llama

    print(f"Loading GGUF model: {model_path}")
    print(f"Model size: {_check_model_size(model_path):.2f} GB")

    _model = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,  # Use all GPU layers
        verbose=False,
    )
    _model_type = "gguf"
    _model_name = Path(model_path).stem
    print("GGUF model loaded successfully")


def load_safetensors_model(model_path: str):
    """Load safetensors model using transformers."""
    global _model, _model_type, _model_name

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"Loading Safetensors model: {model_path}")
    print(f"Model size: {_check_model_size(model_path):.2f} GB")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    _model = {"model": model, "processor": processor, "device": device}
    _model_type = "safetensors"
    _model_name = Path(model_path).name
    print("Safetensors model loaded successfully")


def load_model(model_path: str):
    """Load model with auto-detection of format."""
    size_gb = _check_model_size(model_path)
    if size_gb > _MAX_MODEL_SIZE_GB:
        raise ValueError(
            f"Model size ({size_gb:.2f} GB) exceeds limit of {_MAX_MODEL_SIZE_GB} GB"
        )

    model_type = _detect_model_type(model_path)
    if model_type is None:
        raise ValueError(
            f"Could not detect model format for: {model_path}\n"
            "Expected .gguf file or directory with model.safetensors"
        )

    if model_type == "gguf":
        load_gguf_model(model_path)
    elif model_type == "safetensors":
        load_safetensors_model(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def create_prompt(color_map: dict[str, int]) -> str:
    """Create prompt for single common noun responses.

    Example output format:
        1. person
        2. car
        3. dog
    """
    n = len(color_map)

    color_items = sorted(color_map.items(), key=lambda x: x[1])
    color_list = ", ".join([f"{color} (object #{tid})" for color, tid in color_items])

    prompt = (
        f"You see {n} colored boxes on objects in an image. "
        f"The colors and their object numbers are: {color_list}.\n\n"
        "Identify each object with exactly ONE common noun. "
        "Use simple everyday words like: person, car, dog, cat, chair, table, phone, book, cup, bottle, etc.\n\n"
        "Respond in this exact format (one per line):\n"
    )

    for i, (color, tid) in enumerate(color_items, 1):
        prompt += f"{i}. [common noun for the {color} box]\n"

    prompt += (
        "\nBe concise. Use only single words or very short two-word phrases. "
        "Examples: 'person', 'red car', 'dog', 'wooden chair'"
    )

    return prompt


def identify_with_gguf(image: Image.Image, color_map: dict[str, int]) -> dict[int, str]:
    """Run identification using GGUF model."""
    import io

    prompt = create_prompt(color_map)

    # Convert image to base64 for llama-cpp
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="JPEG", quality=90)
    img_b64 = base64.b64encode(img_buffer.getvalue()).decode()

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
    ]

    messages = [{"role": "user", "content": content}]

    response = _model.create_chat_completion(
        messages=messages,
        max_tokens=100 * len(color_map),
        temperature=0.2,
        stop=["</s>"],
    )

    raw_text = response["choices"][0]["message"]["content"].strip()
    return parse_response(raw_text, color_map)


def identify_with_safetensors(image: Image.Image, color_map: dict[str, int]) -> dict[int, str]:
    """Run identification using Safetensors model."""
    import torch

    model = _model["model"]
    processor = _model["processor"]
    device = _model["device"]

    prompt = create_prompt(color_map)

    # Process inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100 * len(color_map),
            temperature=0.2,
            do_sample=True,
        )

    # Decode
    raw_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return parse_response(raw_text, color_map)


def parse_response(text: str, color_map: dict[str, int]) -> dict[int, str]:
    """Parse numbered list response into {track_id: description}."""
    results = {}
    color_items = sorted(color_map.items(), key=lambda x: x[1])

    # Try numbered list pattern: "1. noun" or "1) noun"
    pattern = re.compile(r"^\s*(\d+)[.)]\s*(.+)$")

    numbered = {}
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            idx = int(m.group(1)) - 1
            desc = m.group(2).strip().lower()
            # Clean up: take only first word or short phrase
            desc = re.sub(r'[^\w\s-]', '', desc).strip()
            if desc:
                numbered[idx] = desc

    # Map to track IDs
    for i, (color, tid) in enumerate(color_items):
        if i in numbered:
            results[tid] = numbered[i]
        else:
            # Fallback: extract any noun-like word
            fallback = extract_noun(text, i) or "object"
            results[tid] = fallback

    return results


def extract_noun(text: str, index: int) -> Optional[str]:
    """Extract a noun from text as fallback."""
    # Simple heuristic: find words that look like nouns
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    common_nouns = [
        "person", "people", "man", "woman", "child", "car", "truck", "bus",
        "dog", "cat", "animal", "bird", "chair", "table", "desk", "couch",
        "phone", "laptop", "computer", "screen", "book", "cup", "bottle",
        "bag", "backpack", "box", "ball", "bike", "bicycle", "motorcycle",
        "tree", "plant", "flower", "building", "house", "door", "window"
    ]

    for word in words:
        if word in common_nouns:
            return word

    return None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": _model_name,
        "type": _model_type,
        "version": "2.0.0",
    }


@app.post("/identify", response_model=IdentifyResponse)
async def identify(
    request: IdentifyRequest,
    authorization: Optional[str] = Header(None)
):
    """Identify objects in the provided annotated frame."""
    # Check API key if configured
    server_key = os.environ.get("SERVER_API_KEY")
    if server_key and authorization != f"Bearer {server_key}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.color_map:
        return IdentifyResponse(results=[])

    try:
        # Decode image
        img_bytes = base64.b64decode(request.annotated_image)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Run identification
        if _model_type == "gguf":
            results = identify_with_gguf(image, request.color_map)
        elif _model_type == "safetensors":
            results = identify_with_safetensors(image, request.color_map)
        else:
            raise HTTPException(status_code=500, detail="Unknown model type")

        # Build response
        response_results = [
            {"track_id": tid, "description": desc}
            for tid, desc in results.items()
        ]

        return IdentifyResponse(results=response_results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="VisionTracker Remote LLM Server")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to model (.gguf file or safetensors directory)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key for authentication",
    )

    args = parser.parse_args()

    # Set API key from env if provided via CLI
    if args.api_key:
        os.environ["SERVER_API_KEY"] = args.api_key

    # Load model
    print("=" * 60)
    print("VisionTracker Remote LLM Server")
    print("=" * 60)
    load_model(args.model_path)
    print("=" * 60)
    print(f"Server ready at http://{args.host}:{args.port}")
    if os.environ.get("SERVER_API_KEY"):
        print("API key authentication enabled")
    print("=" * 60)

    # Start server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
