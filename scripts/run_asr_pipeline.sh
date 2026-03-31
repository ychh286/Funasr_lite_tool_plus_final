#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${FUNASR_LITE_PYTHON:-$ROOT_DIR/.venv/Scripts/python.exe}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${FUNASR_LITE_PYTHON:-$ROOT_DIR/.venv/bin/python}"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found in project-local .venv" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_asr_pipeline.py" "$@"
