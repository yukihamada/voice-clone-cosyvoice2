#!/usr/bin/env bash
set -euo pipefail

# CosyVoice2 Voice Clone — RunPod Serverless Deploy
# Usage: ./deploy.sh [tag]
#
# Prerequisites:
#   - Docker (or Podman)
#   - RunPod account + API key
#   - Docker Hub or RunPod registry credentials

IMAGE_NAME="voice-clone-cosyvoice2"
TAG="${1:-latest}"
REGISTRY="${RUNPOD_REGISTRY:-docker.io}"
REPO="${RUNPOD_REPO:-$(whoami)/${IMAGE_NAME}}"
FULL_IMAGE="${REGISTRY}/${REPO}:${TAG}"

echo "=== CosyVoice2 Voice Clone — RunPod Deploy ==="
echo "Image: ${FULL_IMAGE}"
echo ""

# Build (force linux/amd64 for RunPod)
echo ">> Building Docker image..."
docker build \
  --platform linux/amd64 \
  -t "${FULL_IMAGE}" \
  -f Dockerfile \
  .

# Push
echo ""
echo ">> Pushing to registry..."
docker push "${FULL_IMAGE}"

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. Go to https://www.runpod.io/console/serverless"
echo "  2. Create a new Serverless Endpoint"
echo "  3. Set Docker Image: ${FULL_IMAGE}"
echo "  4. GPU: RTX A4000 (16GB) or A10G (24GB)"
echo "  5. Min/Max Workers: 0/1 (scale to zero)"
echo "  6. Copy the Endpoint ID and use it in index.html"
echo ""
echo "Or create via API:"
echo "  curl -s https://api.runpod.ai/v2/endpoint -H 'Authorization: Bearer \${RUNPOD_API_KEY}'"
