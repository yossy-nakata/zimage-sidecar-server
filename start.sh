#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 <config.json>" >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ -f "$1" ]] || { echo "ERROR: config not found: $1" >&2; exit 1; }

revision="$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["base_revision"])' "$1")"
snapshot="${HF_HOME:-/workspace/hf}/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/${revision}"
[[ -d "$snapshot" ]] || { echo "ERROR: Clean Turbo snapshot not found: $snapshot" >&2; exit 1; }

exec python -u server.py --config "$1" --snapshot "$snapshot"
