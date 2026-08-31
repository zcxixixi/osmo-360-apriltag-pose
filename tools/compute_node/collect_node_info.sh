#!/usr/bin/env bash
set -u
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${OSMO_NODE_RESULT_DIR:-$SCRIPT_DIR/RESULTS}"
mkdir -p "$RESULT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${1:-$RESULT_DIR/ubuntu-node-report-$(hostname)-$STAMP.txt}"
exec > >(tee "$OUT") 2>&1
section() { printf '\n===== %s =====\n' "$1"; }
run() { printf '\n$ %s\n' "$*"; "$@" 2>&1 || true; }
section "IDENTITY"
run date --iso-8601=seconds
run whoami
run hostname
run hostnamectl
section "OS AND KERNEL"
run uname -a
if [ -r /etc/os-release ]; then cat /etc/os-release; fi
section "CPU"
run lscpu
run nproc
section "MEMORY"
run free -h
section "DISKS"
run lsblk -o NAME,MODEL,SERIAL,SIZE,TYPE,FSTYPE,MOUNTPOINTS
run df -hT
section "NETWORK"
run ip -brief address
run ip route
run hostname -I
run ss -ltn
for dev in /sys/class/net/*; do
  [ -e "$dev" ] || continue
  name="$(basename "$dev")"
  printf '%s speed=' "$name"
  cat "$dev/speed" 2>/dev/null || echo unknown
done
section "NVIDIA GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  run nvidia-smi
  run nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader
else
  echo "nvidia-smi: NOT FOUND"
fi
run lspci -nnk
section "CUDA AND VIDEO LIBRARIES"
if command -v nvcc >/dev/null 2>&1; then run nvcc --version; else echo "nvcc: NOT FOUND"; fi
if command -v ldconfig >/dev/null 2>&1; then
  ldconfig -p 2>/dev/null | grep -E 'libcuda|libnvcuvid|libnvidia-encode' || true
fi
section "PYTHON"
run which python3
run python3 --version
python3 - <<'PY'
import importlib.util, json
mods = ['numpy', 'cv2', 'cupy', 'torch', 'av', 'PyNvVideoCodec']
print(json.dumps({name: bool(importlib.util.find_spec(name)) for name in mods}, indent=2))
PY
section "SSH SERVER"
run systemctl is-enabled ssh
run systemctl is-active ssh
run ssh -V
run sshd -T
section "CONNECTION CANDIDATES"
for ip_addr in $(hostname -I 2>/dev/null); do
  case "$ip_addr" in *:*) continue;; esac
  echo "ssh $(whoami)@${ip_addr}"
done
section "REPORT"
echo "REPORT_PATH=$OUT"
