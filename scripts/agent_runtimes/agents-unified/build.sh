docker buildx build \
  --builder default \
  --platform linux/amd64 \
  --network host \
  --load \
  --provenance=false \
  --build-arg HTTP_PROXY=http://127.0.0.1:5429 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:5429 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  -t lego/agents-unified:2026-11-20-amd64 \
  scripts/agent_runtimes/agents-unified