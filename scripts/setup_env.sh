#!/usr/bin/env bash
# Create the project-local environment (.venv) and verify CUDA.
# Requires: sudo apt install python3.12-venv  (or: wsl -u root apt install python3.12-venv)
set -euo pipefail
cd "$(dirname "$0")/.."
echo "==> creating .venv with $(python3 --version)"
[ -x .venv/bin/python ] || python3 -m venv .venv
echo "==> upgrading pip"
.venv/bin/pip install --upgrade pip
echo "==> installing requirements (PyTorch CUDA 12.4 wheels, ~3 GB)"
.venv/bin/pip install -r requirements.txt
echo "==> verifying PyTorch + CUDA"
.venv/bin/python - <<'EOF'
import torch, torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF
echo "==> environment ready"
