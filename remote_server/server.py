"""
Remote LLM Server for VisionTracker — Single Object Mode

FastAPI server supporting both GGUF (llama-cpp-python) and Safetensors
(transformers) vision-language models.

Each request contains a full frame with a SINGLE bounding box drawn.
The LLM returns a single common noun describing the boxed object.

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

app = FastAPI(title="VisionTracker Remote ID Server", version="2.1.0")

# Global model instance
_model = None
_model_type = None  # 'gguf' or 'safetensors'
_model_name = "unknown"
_MAX_MODEL_SIZE_GB = 10


class IdentifyRequest(BaseModel):
    annotated_image: str  # base64 JPEG (full frame with single box drawn)


class IdentifyResponse(BaseModel):
    result: str  # single common noun (e.g., "person", "car", "chair")


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


SINGLE_OBJECT_PROMPT = """Look at the image. There is a green bounding box drawn around one object.

What object is inside the bounding box?

Answer with exactly ONE common noun. Use simple everyday words like: person, car, dog, cat, chair, table, phone, book, cup, bottle, backpack, laptop, etc.

Respond with only the noun, nothing else. Examples: "person", "car", "wooden chair", "coffee mug"
"""


def identify_with_gguf(image: Image.Image) -> str:
    """Run single-object identification using GGUF model."""
    import io

    # Convert image to base64 for llama-cpp
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="JPEG", quality=90)
    img_b64 = base64.b64encode(img_buffer.getvalue()).decode()

    content = [
        {"type": "text", "text": SINGLE_OBJECT_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
    ]

    messages = [{"role": "user", "content": content}]

    response = _model.create_chat_completion(
        messages=messages,
        max_tokens=50,
        temperature=0.2,
        stop=["</s>", "\n"],
    )

    raw_text = response["choices"][0]["message"]["content"].strip()
    return parse_single_response(raw_text)


def identify_with_safetensors(image: Image.Image) -> str:
    """Run single-object identification using Safetensors model."""
    import torch

    model = _model["model"]
    processor = _model["processor"]
    device = _model["device"]

    # Process inputs
    inputs = processor(text=SINGLE_OBJECT_PROMPT, images=image, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.2,
            do_sample=True,
        )

    # Decode
    raw_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return parse_single_response(raw_text)


def parse_single_response(text: str) -> str:
    """Parse LLM response to extract single common noun."""
    # Clean up the response
    text = text.strip().lower()

    # Remove quotes if present
    text = text.strip('"\'')

    # Take only the first line if multiple lines
    text = text.split('\n')[0].strip()

    # Remove trailing punctuation
    text = re.sub(r'[.!?]+$', '', text).strip()

    # Validate: should be a reasonable noun phrase (1-3 words)
    words = text.split()
    if len(words) == 0 or len(words) > 4:
        return "object"

    # Clean up any remaining special characters
    text = re.sub(r'[^\w\s-]', '', text).strip()

    return text if text else "object"


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": _model_name,
        "type": _model_type,
        "version": "2.1.0",
    }


@app.post("/identify", response_model=IdentifyResponse)
async def identify(
    request: IdentifyRequest,
    authorization: Optional[str] = Header(None)
):
    """Identify the single object in the provided annotated frame."""
    # Check API key if configured
    server_key = os.environ.get("SERVER_API_KEY")
    if server_key and authorization != f"Bearer {server_key}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode image
        img_bytes = base64.b64decode(request.annotated_image)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Run identification
        if _model_type == "gguf":
            result = identify_with_gguf(image)
        elif _model_type == "safetensors":
            result = identify_with_safetensors(image)
        else:
            raise HTTPException(status_code=500, detail="Unknown model type")

        return IdentifyResponse(result=result)

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
