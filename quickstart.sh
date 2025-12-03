#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker-compose >/dev/null; then
  echo "docker-compose is required" >&2
  exit 1
fi

echo "Building containers..."
make build

echo "Starting stack..."
make up

echo "Tailing logs (Ctrl+C to stop)"
make logs
