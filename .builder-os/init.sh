#!/usr/bin/env bash
# init.sh - GridCast bootstrap + verification gate (BUILDER-OS).
# Commands run from the PROJECT ROOT (parent of .builder-os/).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(basename "$ROOT_DIR")" = ".builder-os" ]; then
  PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$ROOT_DIR")}"
else
  PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"
fi
cd "$PROJECT_ROOT"

export PYTHONDONTWRITEBYTECODE=1

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
PY=.venv/bin/python

INSTALL_CMD="${INSTALL_CMD:-$PY -m pip install -q -r requirements.txt}"
VERIFY_CMD="${VERIFY_CMD:-$PY -m pytest tests/ -q}"
START_CMD="${START_CMD:-$PY -m gridcast run}"

echo "[init] project root: $PROJECT_ROOT"
echo "[init] install: $INSTALL_CMD"
eval "$INSTALL_CMD"
echo "[init] verify: $VERIFY_CMD"
eval "$VERIFY_CMD"
echo "[init] run pipeline: $START_CMD"
eval "$START_CMD"
echo "[init] OK - all gates green"
