#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 scripts/deploy/init_auth.py
mkdir -p .runtime/jobs .runtime/annotations
exec docker run --rm --name "${DXA_CONTAINER_NAME:-imagelab-dxa}" \
  --user "$(id -u):$(id -g)" \
  -p "127.0.0.1:${DXA_PORT:-8095}:8095" \
  --mount "type=bind,src=$PWD/.runtime/auth.json,dst=/run/secrets/dxa-auth.json,readonly" \
  --mount "type=bind,src=$PWD/.runtime/jobs,dst=/runtime/jobs" \
  --mount "type=bind,src=$PWD/.runtime/annotations,dst=/runtime/annotations" \
  imagelab-dxa:final
