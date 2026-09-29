#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec python3 -m uvicorn dxa.api:app --host 127.0.0.1 --port "${DXA_PORT:-8095}"
