#!/usr/bin/env bash
# Run EduScope from the eduscope/ directory
set -e
cd "$(dirname "$0")"

if [ ! -f "venv/bin/activate" ]; then
    echo ""
    echo "ERROR: Virtual environment not found. Complete the setup steps first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install --upgrade pip"
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
fi

source venv/bin/activate

echo "Starting EduScope server…"
echo "Open http://localhost:8000 in your browser."
echo ""

uvicorn backend.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
