#!/usr/bin/env bash
set -euo pipefail

REQUIRED_MODELS=("llama3.2" "nomic-embed-text")
OPTIONAL_MODELS=("qwen3:8b" "mistral")

# ── 1. Ollama health check ────────────────────────────────────────────────────
echo "Checking Ollama..."
if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "ERROR: Ollama is not running. Start it with 'ollama serve' and re-run this script."
  exit 1
fi
echo "  Ollama is running."

# ── 2. Pull required Ollama models ───────────────────────────────────────────
AVAILABLE=$(curl -sf http://localhost:11434/api/tags | python3 -c "
import sys, json
models = json.load(sys.stdin).get('models', [])
print(' '.join(m['name'].split(':')[0] for m in models))
")

for model in "${REQUIRED_MODELS[@]}"; do
  base="${model%%:*}"
  if echo "$AVAILABLE" | grep -qw "$base"; then
    echo "  $model already pulled."
  else
    echo "  Pulling required model: $model"
    ollama pull "$model"
  fi
done

for model in "${OPTIONAL_MODELS[@]}"; do
  base="${model%%:*}"
  if echo "$AVAILABLE" | grep -qw "$base"; then
    echo "  $model already pulled."
  else
    echo "  Skipping optional model $model (pull manually with: ollama pull $model)"
  fi
done

# ── 3. Python virtual environment ────────────────────────────────────────────
echo ""
echo "Setting up Python environment..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "  Created .venv"
fi
# shellcheck source=/dev/null
source .venv/bin/activate
echo "  Activated .venv"

# ── 4. Install dependencies ──────────────────────────────────────────────────
echo "  Installing dependencies..."
pip install -q -r requirements.txt
echo "  Dependencies installed."

# ── 5. Environment file ──────────────────────────────────────────────────────
echo ""
if [ ! -f ".env" ]; then
  cp .env.example .env
  # Fix the default URL for local dev (example file targets Docker)
  sed -i '' 's|OLLAMA_BASE_URL=http://host.docker.internal:11434|OLLAMA_BASE_URL=http://localhost:11434|' .env
  echo "Created .env from .env.example (OLLAMA_BASE_URL set to localhost)."
  echo "  Add your OPENAI_API_KEY to .env if you want to use GPT models."
  echo ""
else
  echo ".env already exists — skipping copy."
fi

# ── 6. Port check ────────────────────────────────────────────────────────────
PORT=7860
EXISTING_PID=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
  echo ""
  echo "Port $PORT is in use by PID $EXISTING_PID — killing it..."
  kill "$EXISTING_PID"
  sleep 1
fi

# ── 7. Launch ────────────────────────────────────────────────────────────────
echo "Starting app at http://localhost:$PORT ..."
echo ""
python3 app.py
