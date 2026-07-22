#!/bin/zsh
# Keep a portable Apple-Silicon long-training process awake for the utility's lifetime.
set -euo pipefail

PORTABLE_SCRIPT_DIR="${0:A:h}"
PORTABLE_ROOT="${PORTABLE_SCRIPT_DIR:h}"
PORTABLE_PYTHON="${PYTHON_BIN:-/opt/homebrew/Caskroom/miniconda/base/envs/torchenv/bin/python}"

cd "$PORTABLE_ROOT"

if [[ ! -x "$PORTABLE_PYTHON" ]]; then
  print -u2 "[run_long_train_mps] Python is not executable: $PORTABLE_PYTHON"
  exit 1
fi

unset PYTORCH_ENABLE_MPS_FALLBACK
export PYTHONUNBUFFERED=1

# -i: prevent idle system sleep; -m: prevent disk idle sleep;
# -s: prevent system sleep while on AC. Display sleep remains enabled.
exec /usr/bin/caffeinate -ims "$PORTABLE_PYTHON" \
  scripts/long_train_portable_mps.py "$@"
