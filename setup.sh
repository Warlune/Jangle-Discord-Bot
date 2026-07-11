#!/usr/bin/env sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.11}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=${PYTHON_FALLBACK:-python3}
fi

if [ ! -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    "$PYTHON_BIN" -m venv "$PROJECT_ROOT/.venv"
fi

"$PROJECT_ROOT/.venv/bin/python" -m pip install --upgrade pip
"$PROJECT_ROOT/.venv/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"

if [ ! -f "$PROJECT_ROOT/.env" ]; then
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    printf '%s\n' "Created .env from .env.example. Add your Discord token and model settings."
fi

printf '%s\n' "Setup complete. Run .venv/bin/python bot.py after configuring .env."
