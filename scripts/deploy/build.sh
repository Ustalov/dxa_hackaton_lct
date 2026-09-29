#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 scripts/deploy/verify_runtime.py
docker build -t imagelab-dxa:final .
