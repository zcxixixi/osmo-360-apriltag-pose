# Ubuntu GPU compute-node kit

This kit inventories an Ubuntu workstation, enables public-key SSH, and verifies CUDA/NVDEC Python packages. Reports stay under the kit's `RESULTS/` directory, allowing the kit to run directly from removable storage.

Build a node-specific removable-drive package without committing the public key:

```bash
python tools/compute_node/package_node_kit.py \
  --public-key-file ~/.ssh/id_ed25519.pub \
  --output /media/$USER/DRIVE/OSMO_GPU_NODE_ONBOARDING
```

On the target Ubuntu node, connect Ethernet or Wi-Fi, then run:

```bash
bash RUN_ON_UBUNTU.sh
```

The generated desktop launcher is optional and may require “Allow Launching” on first use. The package installs `openssh-server`, generates missing SSH host keys, authorizes only the supplied public key, and opens the OpenSSH UFW profile only when UFW is already active. It does not install NVIDIA drivers or vendor SDK binaries.

After network connectivity is available, install the project GPU environment with:

```bash
uv sync --extra node-gpu
```

Insta360 vendor binaries are intentionally excluded from the public repository. Verify a locally licensed copy with `tools/verify_vendor_sdk.py` and `config/sdk_revisions/insta360_linux_sdk_20260831.json`.
