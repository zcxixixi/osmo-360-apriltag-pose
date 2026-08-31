#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KEY_FILE="${1:-$SCRIPT_DIR/WORKSTATION_PUBLIC_KEY.pub}"
if [ ! -f "$KEY_FILE" ]; then
  echo "Missing public key file: $KEY_FILE" >&2
  echo "Pass an OpenSSH public-key file as the first argument." >&2
  exit 2
fi
PUBLIC_KEY="$(head -n 1 "$KEY_FILE")"
case "$PUBLIC_KEY" in
  ssh-ed25519\ *|ssh-rsa\ *|ecdsa-sha2-nistp256\ *|ecdsa-sha2-nistp384\ *|ecdsa-sha2-nistp521\ *) ;;
  *) echo "Unsupported or malformed OpenSSH public key" >&2; exit 2;;
esac
sudo apt-get update
sudo apt-get install -y openssh-server
sudo ssh-keygen -A
sudo sshd -t
sudo systemctl enable --now ssh
sudo systemctl restart ssh
install -d -m 700 "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys"
chmod 600 "$HOME/.ssh/authorized_keys"
if ! grep -qxF "$PUBLIC_KEY" "$HOME/.ssh/authorized_keys"; then
  printf '%s\n' "$PUBLIC_KEY" >> "$HOME/.ssh/authorized_keys"
fi
if command -v ufw >/dev/null 2>&1 && sudo ufw status | grep -q '^Status: active'; then
  sudo ufw allow OpenSSH
fi
sudo systemctl --no-pager --full status ssh || true
if ! hostname -I | grep -Eq '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+'; then
  echo "NO_IPV4_ADDRESS: connect Ethernet or Wi-Fi, then rerun." >&2
fi
for ip_addr in $(hostname -I 2>/dev/null); do
  case "$ip_addr" in *:*) continue;; esac
  echo "ssh $(whoami)@${ip_addr}"
done
