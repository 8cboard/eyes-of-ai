# VisionTracker Remote LLM Server

FastAPI server supporting both GGUF (llama-cpp-python) and Safetensors (transformers) vision-language models for VisionTracker identification.

## Features

- **Auto-detects model format**: GGUF by `.gguf` extension, Safetensors by directory structure
- **Model size limit**: <10GB (enforced on load)
- **Prompt engineering**: Optimized for single common noun responses
- **Secure**: Optional API key authentication

## Quick Start

### 1. Install Dependencies

```bash
cd remote_server
pip install -r requirements.txt
```

### 2. Download a Vision Model

**GGUF (Recommended for smaller GPUs):**
```bash
# Gemma 3 4B IT - Q4_K_M quantized (~3.3GB)
wget https://huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf
```

**Safetensors (Better quality, needs more VRAM):**
```bash
# Will be auto-downloaded on first use
# Or download manually with huggingface-cli
huggingface-cli download google/gemma-3-4b-it
```

### 3. Start Server

**GGUF:**
```bash
python server.py --model-path ./gemma-3-4b-it-Q4_K_M.gguf
```

**Safetensors:**
```bash
python server.py --model-path /path/to/gemma-3-4b-it
```

**With API key:**
```bash
python server.py --model-path ./model.gguf --api-key your-secret-key
```

### 4. Public Access (ngrok)

```bash
# Install ngrok: https://ngrok.com/download
ngrok http 8000

# Use the https URL in VisionTracker:
# python main.py --remote-url https://abc123.ngrok.io
```

## API Reference

### Health Check
```
GET /health

Response:
{
  "status": "healthy",
  "model": "gemma-3-4b-it-Q4_K_M",
  "type": "gguf",
  "version": "2.0.0"
}
```

### Identify Objects
```
POST /identify
Content-Type: application/json
Authorization: Bearer <optional-api-key>

Request:
{
  "annotated_image": "base64_jpeg_string",
  "color_map": {
    "red": 1,
    "blue": 2,
    "green": 3
  }
}

Response:
{
  "results": [
    {"track_id": 1, "description": "person"},
    {"track_id": 2, "description": "car"},
    {"track_id": 3, "description": "dog"}
  ]
}
```

## Supported Models

### GGUF (llama-cpp-python)

| Model | Size | VRAM | Quality |
|-------|------|------|---------|
| gemma-3-4b-it-Q4_K_M | ~3.3GB | ~4GB | Good |
| gemma-3-4b-it-Q5_K_M | ~4.2GB | ~5GB | Better |
| gemma-3-4b-it-Q6_K | ~5.0GB | ~6GB | Best |

Download from: https://huggingface.co/lmstudio-community

### Safetensors (transformers)

| Model | Size | VRAM | Notes |
|-------|------|------|-------|
| google/gemma-3-4b-it | ~8GB | ~10GB | Fits on P100, may OOM on T4 |
| microsoft/Phi-4-multimodal-instruct | ~14GB | ~16GB | Exceeds 10GB limit |

## Cloud Deployment

### Google Colab

1. Open `colab_setup.ipynb` in Google Colab
2. Runtime → Change runtime type → GPU (T4)
3. Run all cells
4. Copy the public URL from output

### Kaggle

1. Create notebook from `kaggle_setup.ipynb`
2. Settings → Accelerator → GPU
3. Add ngrok token to Secrets
4. Run all cells

## Troubleshooting

### Out of Memory

**Reduce GPU layers (GGUF):**
Edit server.py and change `n_gpu_layers=-1` to `n_gpu_layers=20`

**Use CPU:**
Set `n_gpu_layers=0` in server.py

**Use smaller model:**
Download Q4_K_M instead of Q5_K_M or Q6_K

### Slow First Request

This is normal - CUDA kernels are initializing. Subsequent requests are faster.

### Model Format Not Detected

The server auto-detects:
- `.gguf` extension → GGUF format
- Directory with `model.safetensors` or `pytorch_model.bin` → Safetensors

If auto-detection fails, check the path exists and contains expected files.

### Connection Issues

- Keep the server running (don't close notebook/terminal)
- ngrok free tier has 2-hour session limits
- Check firewall settings

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VisionTracker (Local)                    │
│  ┌──────────┐    ┌──────────┐    ┌─────────────────────┐   │
│  │  Edge    │───▶│ Tracker  │───▶│ IDService           │   │
│  │ Detector │    │          │    │  - Draw colored boxes│   │
│  └──────────┘    └──────────┘    │  - POST to server   │   │
│                                   └──────────┬──────────┘   │
│                                              │ HTTPS        │
└──────────────────────────────────────────────┼──────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Remote Server (Your Hardware)                  │
│  ┌──────────┐    ┌──────────────────────────────────────┐  │
│  │  FastAPI │───▶│  LLM (GGUF or Safetensors)           │  │
│  │  Server  │    │  - Vision encoder                    │  │
│  │  :8000   │    │  - Prompt: "Identify by color"       │  │
│  └──────────┘    │  - Response: {1:"person",2:"car"}    │  │
│                  └──────────────────────────────────────┘  │
│                                                             │
│  Endpoints:                                                 │
│    GET /health   - Health check                             │
│    POST /identify - Identify objects by colored boxes       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Prompt Engineering

The server uses carefully engineered prompts to get single common noun responses:

```
You see N colored boxes on objects in an image.
The colors and their object numbers are: red (object #1), blue (object #2), ...

Identify each object with exactly ONE common noun.
Use simple everyday words like: person, car, dog, chair, table, etc.

Respond in this exact format (one per line):
1. [common noun for the red box]
2. [common noun for the blue box]
...
```

This produces responses like:
- `1. person`
- `2. car`
- `3. dog`

## Performance

| Setup | Format | Batch Size | Time per Request |
|-------|--------|------------|------------------|
| Colab T4 | GGUF Q4_K_M | 4 | 2-4s |
| Colab T4 | GGUF Q4_K_M | 8 | 4-6s |
| Kaggle P100 | GGUF Q4_K_M | 8 | 2-3s |
| Kaggle P100 | Safetensors | 4 | 3-5s |

## License

Same as VisionTracker project. Model weights subject to their respective licenses (Gemma, Phi, etc.).
