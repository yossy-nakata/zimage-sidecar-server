#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${R2_KEY:?R2_KEY is required}"
: "${R2_SECRET:?R2_SECRET is required}"
: "${R2_URL:?R2_URL is required}"
: "${SIDECAR_API_KEY:?SIDECAR_API_KEY is required}"

export HF_HOME="${HF_HOME:-/workspace/hf}"
export UV_SYSTEM_PYTHON=1
export UV_PYTHON_DOWNLOADS=0
export UV_TORCH_BACKEND=cu126
export PIP_REQUIRE_VIRTUALENV=true

[[ "$(command -v python)" == '/usr/local/bin/python' ]] || {
  echo 'ERROR: expected /usr/local/bin/python' >&2
  exit 1
}
for cmd in curl tar rclone uv; do
  command -v "$cmd" >/dev/null || { echo "ERROR: missing $cmd" >&2; exit 1; }
done
uv pip install bitsandbytes peft datasets
python - <<'PY'
import torch, diffusers, transformers, safetensors, huggingface_hub
assert torch.cuda.is_available(), 'CUDA is required'
print('GPU:', torch.cuda.get_device_name(0))
print('torch:', torch.__version__, 'diffusers:', diffusers.__version__, 'transformers:', transformers.__version__)
PY

# On a fresh Pod, fetch the public GitHub source archive (no git required).
if [[ ! -f /workspace/sidecar-server/server.py ]]; then
  mkdir -p /workspace/sidecar-server
  curl -fsSL https://github.com/yossy-nakata/zimage-sidecar-server/archive/refs/heads/main.tar.gz \
    | tar -xz -C /workspace/sidecar-server --strip-components=1
fi
for f in server.py runtime.py sidecar.py config.json; do
  [[ -f "/workspace/sidecar-server/$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done
python -m py_compile /workspace/sidecar-server/{server,runtime,sidecar}.py

mkdir -p /root/.config/rclone /workspace/identity /workspace/models /workspace/loras "$HF_HOME"
cat > /root/.config/rclone/rclone.conf <<EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_KEY}
secret_access_key = ${R2_SECRET}
endpoint = ${R2_URL}
EOF
chmod 600 /root/.config/rclone/rclone.conf

if [[ ! -f /workspace/identity/character-v3.safetensors ]]; then
  rclone copyto r2:nana-storage/zimage-sidecar/identity/character-v3.safetensors \
    /workspace/identity/character-v3.safetensors --retries 3 --low-level-retries 10
fi
if [[ ! -f /workspace/models/sidecar-step-006000.safetensors ]]; then
  rclone copyto r2:nana-storage/zimage-sidecar/runs/sidecar-n40-turbo-mix-v1/sidecar-step-006000.safetensors \
    /workspace/models/sidecar-step-006000.safetensors --retries 3 --low-level-retries 10
fi

# Download the LoRA files from R2. Existing unchanged files are skipped.
rclone copy r2:nana-storage/zimage-sidecar/loras /workspace/loras \
  --retries 3 --low-level-retries 10

# No Training Adapter and no train-mix: inference always uses Clean Turbo.
export SIDECAR_CLEAN_SNAPSHOT="$(python - <<'PY'
import json, os, sys
from pathlib import Path
from huggingface_hub import snapshot_download
config = json.loads(Path('/workspace/sidecar-server/config.json').read_text(encoding='utf-8'))
revision = config['base_revision']
path = Path(snapshot_download(
    repo_id='Tongyi-MAI/Z-Image-Turbo', revision=revision, token=os.environ['HF_TOKEN'],
    allow_patterns=[
        'model_index.json', 'scheduler/*', 'text_encoder/*',
        'tokenizer/*', 'transformer/*', 'vae/*',
    ],
))
if path.name != revision:
    raise SystemExit('unexpected snapshot revision')
for name in ('model_index.json', 'scheduler', 'text_encoder', 'tokenizer', 'transformer', 'vae'):
    if not (path / name).exists():
        raise SystemExit(f'missing snapshot component: {name}')
if any(p.is_symlink() and not p.exists() for p in path.rglob('*')):
    raise SystemExit('broken symlink in snapshot')
print(path)
PY
)"

cd /workspace/sidecar-server
exec python -u server.py
