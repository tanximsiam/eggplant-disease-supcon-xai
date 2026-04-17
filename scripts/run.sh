#!/usr/bin/env bash
# Run a pipeline stage in the foreground with live progress, also teed to logs/<stage>_<time>.log
#   bash scripts/run.sh download | audit | split | smoke | train [--tag pilot ...] | ledger | evaluate | xai | export
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
log="logs/$1_$(date +%Y%m%d_%H%M%S).log"
echo "==> python -m eggplant $*   (log: $log)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # deterministic cuBLAS
PYTHONUNBUFFERED=1 .venv/bin/python -m eggplant "$@" 2>&1 | tee "$log"
exit "${PIPESTATUS[0]}"
