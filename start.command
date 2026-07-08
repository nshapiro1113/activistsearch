#!/usr/bin/env bash
# Double-click this file in Finder to set up (first run only) and launch the app.
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Setting up (first run only)..."
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt
streamlit run app.py
