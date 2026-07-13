#!/usr/bin/env bash
# Double-click this file in Finder to set up (first run only) and launch the app.
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv.nosync" ]; then
  echo "Setting up (first run only)..."
  python3 -m venv .venv.nosync
fi

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

source .venv.nosync/bin/activate
pip install -q -r requirements.txt
streamlit run app.py
