#!/usr/bin/env bash
# Rebuild the hosted-app environment from app/requirements.txt (CPU-only, as on Streamlit Cloud) and serve it.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf /tmp/eggplant-app-venv
python3 -m venv /tmp/eggplant-app-venv
/tmp/eggplant-app-venv/bin/pip install --upgrade pip
/tmp/eggplant-app-venv/bin/pip install -r app/requirements.txt
/tmp/eggplant-app-venv/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA build:', torch.version.cuda)"
exec /tmp/eggplant-app-venv/bin/streamlit run app/streamlit_app.py --server.headless true --server.port 8502 --server.address localhost
