#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python3}"
VENV_DIR="${2:-.venv}"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  exit 0
fi

mkdir -p "${VENV_DIR}"

# Do not rely on ensurepip being present on the host.
"${PYTHON_BIN}" -m venv --without-pip "${VENV_DIR}"

GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"
GET_PIP_PATH="${VENV_DIR}/get-pip.py"

"${PYTHON_BIN}" - <<PY
import ssl
import urllib.request
from pathlib import Path

url = "${GET_PIP_URL}"
dst = Path("${GET_PIP_PATH}")
dst.parent.mkdir(parents=True, exist_ok=True)

ctx = ssl.create_default_context()
with urllib.request.urlopen(url, context=ctx) as r:
    data = r.read()
dst.write_bytes(data)
print(f"Downloaded {url} -> {dst}")
PY

"${VENV_DIR}/bin/python" "${GET_PIP_PATH}" --disable-pip-version-check

# Basic build tooling; keep minimal and pinned by requirements.txt for the rest.
"${VENV_DIR}/bin/pip" install --disable-pip-version-check --upgrade pip==24.0 wheel==0.43.0 setuptools==69.5.1

