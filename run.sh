#!/usr/bin/env bash
# Run EduScope from the eduscope/ directory
set -e
cd "$(dirname "$0")"

source venv/bin/activate

echo "Starting EduScope server…"
echo "Open http://localhost:8000 in your browser."
echo ""

uvicorn backend.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
