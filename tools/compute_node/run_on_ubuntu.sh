#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${OSMO_NODE_RESULT_DIR:-$SCRIPT_DIR/RESULTS}"
KEY_FILE="${1:-$SCRIPT_DIR/WORKSTATION_PUBLIC_KEY.pub}"
mkdir -p "$RESULT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
HOST="$(hostname)"
LOG="$RESULT_DIR/onboarding-${HOST}-${STAMP}.log"
exec > >(tee "$LOG") 2>&1
bash "$SCRIPT_DIR/collect_node_info.sh" "$RESULT_DIR/node-before-${HOST}-${STAMP}.txt"
bash "$SCRIPT_DIR/enable_ssh_node.sh" "$KEY_FILE"
bash "$SCRIPT_DIR/collect_node_info.sh" "$RESULT_DIR/node-after-${HOST}-${STAMP}.txt"
python3 "$SCRIPT_DIR/gpu_smoke_test.py" | tee "$RESULT_DIR/gpu-smoke-${HOST}-${STAMP}.json"
printf '\nFiles to return:\n%s\n%s\n%s\n' \
  "$RESULT_DIR/node-after-${HOST}-${STAMP}.txt" \
  "$RESULT_DIR/gpu-smoke-${HOST}-${STAMP}.json" \
  "$LOG"
sync
