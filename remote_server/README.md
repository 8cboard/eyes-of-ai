# 🌐 VisionTracker Remote Server

Self-hosted Gemma 3 4B vision-language model server for VisionTracker object identification. Run on Google Colab or Kaggle for private, rate-limit-free identification.

## Why Self-Host?

| Feature | OpenRouter Free | Self-Hosted (This) |
|---------|-----------------|-------------------|
| **Rate Limit** | ~20 RPM | Unlimited (you control it) |
| **Privacy** | Sends images to third-party | Your server, your data |
| **Cost** | Free | Free (Colab/Kaggle tiers) |
| **Latency** | Network variable | 1-3s after warmup |
| **Setup** | API key only | ~5 minute setup |

## Quick Start

### Option 1: Google Colab (Recommended for beginners)

1. Open `colab_setup.ipynb` in Google Colab
2. Runtime → Change runtime type → GPU (T4)
3. Follow the 6 cells:
   - Install dependencies
   - Configure ngrok (get free token at [ngrok.com](https://dashboard.ngrok.com/signup))
   - Download Gemma 3 4B model (~3.3GB)
   - Load model & warmup
   - Create FastAPI server
   - Start tunnel & server

4. Copy the **Public URL** from output
5. Run VisionTracker locally:
   ```bash
   python main.py --use-remote-gemma --remote-gemma-url https://abc123.ngrok.io
   ```

### Option 2: Kaggle (More GPU RAM)

1. Create Kaggle account at [kaggle.com](https://kaggle.com)
2. Open `kaggle_setup.ipynb` in Kaggle
3. Settings → Accelerator → GPU (T4/P100)
4. Settings → Internet → On
5. Add ngrok token to Secrets: Add-ons → Secrets → `NGROK_AUTHTOKEN`
6. Run all cells
7. Use the public URL in VisionTracker

## Model Details

- **Model**: Gemma 3 4B IT (instruction-tuned)
- **Quantization**: Q4_K_M (4-bit, ~3.3GB)
- **Original**: Google Gemma 3 4B
- **Context**: 4096 tokens
- **VRAM**: ~4GB (fits in both Colab T4 and Kaggle P100)

## API Endpoints

### Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "model": "gemma-3-4b-it",
  "version": "1.0.0"
}
```

### Identify Objects
```bash
POST /identify
Content-Type: application/json
Authorization: Bearer <optional-api-key>

Request:
{
  "images": ["base64_jpeg_1", "base64_jpeg_2"],
  "class_names": ["person", "chair"],
  "track_ids": [1, 2]
}

Response:
{
  "results": [
    {"track_id": 1, "description": "a person in a blue jacket"},
    {"track_id": 2, "description": "a wooden chair with cushion"}
  ]
}
```

## Security

### API Key (Optional but Recommended)

Add an API key to prevent unauthorized access:

**Colab**: Click 🔑 in left sidebar → Add secret: `SERVER_API_KEY`

**Kaggle**: Add-ons → Secrets → Add secret: `SERVER_API_KEY`

Then pass it to VisionTracker:
```bash
python main.py --use-remote-gemma \
  --remote-gemma-url https://abc123.ngrok.io \
  --remote-gemma-key your-secret-key
```

### ngrok Considerations

- Free tier: 1 tunnel, ~2hr sessions, 40 conn/min
- URLs change on restart
- Use paid ngrok for persistent URLs
- Consider Cloudflare Tunnel for longer sessions

## Troubleshooting

### Out of Memory
Reduce GPU layers in notebook:
```python
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,        # Reduce from 4096
    n_gpu_layers=20,   # Instead of -1 (all)
    verbose=False
)
```

### Slow First Request
This is normal - CUDA kernels are initializing. Subsequent requests are fast.

### Connection Refused
- Keep the Colab/Kaggle notebook running
- Check ngrok tunnel status
- Verify the URL hasn't changed (free tier rotates URLs)

### Model Download Fails
```bash
# Manual download fallback
!wget https://huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VisionTracker (Local)                    │
│  ┌──────────┐    ┌─────────────────────────────────────┐   │
│  │ Detector │───▶│ GemmaRemoteService (gemma_remote_   │   │
│  │  YOLOv8  │    │ service.py)                         │   │
│  └──────────┘    │  - Batch crops (up to 16)           │   │
│                  │  - HTTP POST to remote server       │   │
│                  │  - No rate limiting (user control)  │   │
│                  │  - Retry with backoff               │   │
│                  └──────────┬──────────────────────────┘   │
│                             │ HTTPS + ngrok tunnel         │
│                             ▼                              │
│                  ┌─────────────────────┐                   │
│                  │  ngrok Cloud        │                   │
│                  │  (tunnel broker)    │                   │
│                  └──────────┬──────────┘                   │
│                             │                              │
└─────────────────────────────┼──────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Remote Server (Colab/Kaggle)                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │  FastAPI │───▶│  Llama   │───▶│ Gemma 3 4B (GGUF)    │  │
│  │  Server  │    │  CPP     │    │ Q4_K_M quantized     │  │
│  │  :8000   │    │  Python  │    │ ~3.3GB, T4 GPU       │  │
│  └──────────┘    └──────────┘    └──────────────────────┘  │
│                                                             │
│  Endpoints:                                                 │
│    GET /health   - Health check                             │
│    POST /identify - Batch object identification             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Files

- `colab_setup.ipynb` - Google Colab notebook
- `kaggle_setup.ipynb` - Kaggle notebook
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Performance Comparison

| Setup | Batch Size | Time per Batch | Effective IDs/min |
|-------|-----------|----------------|-------------------|
| OpenRouter Free | 4 | 5-15s | ~50 |
| Colab T4 | 4 | 2-5s | ~150 |
| Kaggle T4 | 8 | 3-6s | ~300 |
| Kaggle P100 | 8 | 2-4s | ~400 |

## Contributing

The server implementation is in the notebook cells. To modify:

1. Edit the notebook cell containing the FastAPI app
2. Test locally with `uvicorn main:app --reload`
3. Export updated `.ipynb` to this directory

## License

Same as VisionTracker project. Model weights subject to Google's Gemma license.
