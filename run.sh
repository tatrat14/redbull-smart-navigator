#!/usr/bin/env bash
# ============================================================
#   Astana Smart Navigator - one-click launcher (macOS / Linux)
#   Run with:  ./run.sh   (or double-click on most desktops)
#   First run sets everything up; afterwards it launches instantly.
# ============================================================
set -e
cd "$(dirname "$0")"

# --- Find a working Python --------------------------------------------------
PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[ERROR] Python 3.10+ not found. Install it from https://www.python.org/downloads/"
  exit 1
fi

# --- Create the virtual environment on first run ---------------------------
if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment..."
  "$PY" -m venv .venv
fi
VPY=".venv/bin/python"

# --- Install dependencies once ---------------------------------------------
if [ ! -f ".venv/.deps_installed" ]; then
  echo "Installing dependencies (first run can take a few minutes)..."
  "$VPY" -m pip install --upgrade pip
  "$VPY" -m pip install -r requirements.txt
  touch ".venv/.deps_installed"
fi

# --- Launch the app (opens in your browser) --------------------------------
echo "Starting Astana Smart Navigator... (Ctrl+C to stop)"
exec "$VPY" -m streamlit run app.py
