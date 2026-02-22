# CosyVoice2 Voice Clone — Deploy Guide

## Pre-requisites
- Docker Desktop running
- Docker Hub login (`docker login`)
- RunPod account with API key
- AWS CLI configured (for Lambda env vars)
- Fly CLI configured (for Fly.io secrets)

## Step 1: Build & Push Docker Image

```bash
cd /Users/yuki/workspace/ai/voice-clone
./deploy.sh
# Or manually:
docker buildx build --platform linux/amd64 \
  -t yukihamada/cosyvoice2-runpod:latest \
  --push .
```

Image includes:
- PyTorch 2.5.1 + CUDA 12.1
- CosyVoice2-0.5B model (baked in, ~1.5GB)
- ffmpeg for WebM/OGG audio conversion

## Step 2: Create RunPod Serverless Endpoint

1. Go to https://www.runpod.io/console/serverless
2. Click "New Endpoint"
3. Settings:
   - **Docker Image**: `yukihamada/cosyvoice2-runpod:latest`
   - **GPU**: RTX 4090 (24GB) — sufficient for CosyVoice2-0.5B
   - **Min Workers**: 0 (scale to zero)
   - **Max Workers**: 2
   - **Idle Timeout**: 120s
   - **Execution Timeout**: 120s
4. Copy the **Endpoint ID** (e.g., `abc123def456`)

## Step 3: Set Environment Variables

### Lambda
```bash
# Get current env vars first
aws lambda get-function-configuration \
  --function-name nanobot-prod \
  --region ap-northeast-1 \
  --query 'Environment.Variables' > /tmp/lambda-env.json

# Add new vars (merge with existing!)
aws lambda update-function-configuration \
  --function-name nanobot-prod \
  --region ap-northeast-1 \
  --environment "Variables={...,RUNPOD_API_KEY=rpa_YOUR_KEY,RUNPOD_COSYVOICE_ENDPOINT_ID=YOUR_ENDPOINT_ID}"
```

### Fly.io
```bash
fly secrets set RUNPOD_API_KEY=rpa_YOUR_KEY -a nanobot-ai
fly secrets set RUNPOD_COSYVOICE_ENDPOINT_ID=YOUR_ENDPOINT_ID -a nanobot-ai
```

## Step 4: Deploy nanobot (Rust fix)

The `"output_format"` → `"format"` fix in http.rs needs to be deployed:

```bash
cd /Users/yuki/workspace/ai/chatweb.ai
LAMBDA_FUNCTION_NAME=nanobot-prod ./infra/deploy-fast.sh
```

## Step 5: Test

### Health check
```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"input":{"text":"__ping__"}}'
```

### SFT mode (no voice clone, basic TTS)
```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"input":{"text":"こんにちは、テストです","mode":"sft","format":"mp3"}}'
```

### Voice Clone (zero_shot)
```bash
AUDIO_B64=$(base64 -i sample.wav)
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"input\":{\"text\":\"こんにちは\",\"prompt_audio\":\"${AUDIO_B64}\",\"prompt_text\":\"テスト\",\"mode\":\"zero_shot\",\"format\":\"mp3\"}}"
```

### Via chatweb.ai API
```bash
AUDIO_B64=$(base64 -i sample.wav)
curl -X POST "https://chatweb.ai/api/v1/voice/clone" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"こんにちは\",\"audio_base64\":\"${AUDIO_B64}\",\"prompt_text\":\"テスト\",\"mode\":\"zero_shot\"}" \
  --output cloned.mp3
```

## Architecture
```
Browser (WebM recording)
  → POST /api/v1/voice/clone (Lambda/Fly.io)
  → handle_voice_clone() [Rust]
  → Fallback chain: Replicate → RunPod CosyVoice2 → OpenAI TTS
  → RunPod Serverless (CosyVoice2-0.5B, RTX 4090)
  → base64 MP3 response
  → Browser playback
```

## Cost Estimate
| Item | Cost |
|------|------|
| RunPod RTX 4090 (Flex) | ~$0.34/hr (per-second billing) |
| Per clone (~3s inference) | ~$0.0003 |
| 100 clones/day | ~$1/month |
| 1000 clones/day | ~$10/month |
| Cold start (model baked in) | ~15-30s (GPU allocation) |
| Warm latency | ~2-5s |
