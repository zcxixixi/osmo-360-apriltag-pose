# Visualization platform security

The LAN review platform deliberately keeps read-only review URLs public inside
the trusted network. Every state-changing request is authenticated before its
body, project ID, or destination is processed:

| Method and route | Authentication |
|---|---|
| `GET /`, `/healthz`, `/api/capabilities`, projects, views, assets | public read |
| `PUT /api/devices` | Bearer token |
| `POST /api/projects` | Bearer token |
| `PUT /api/projects/:id/{timeline,video,scene}` | Bearer token |
| `POST /api/projects/:id/publish` | Bearer token |

The server refuses to start without a URL-safe token containing at least 256
bits of random material. Prefer a private file instead of an inline environment
variable:

```bash
install -d -m 0700 ~/.config/osmo360
umask 077
openssl rand -hex 32 > ~/.config/osmo360/platform-write-token
OSMO_PLATFORM_WRITE_TOKEN_FILE="$HOME/.config/osmo360/platform-write-token" \
  node dual_gripper_3d/platform_server.mjs \
  --data-dir /srv/osmo-visualization/data \
  --mesh-dir assets/gripper_v52_new_r1/meshes \
  --host 0.0.0.0 --port 7865
```

The file must be mode `0600`; the server and Python clients reject a file that
is accessible by the group or other users. Never commit, print, log, or place a
token in a query string. Keep the configured file outside the repository.

The upload and inventory clients read
`~/.config/osmo360/platform-write-token` by default. They also support an
explicit private path:

```bash
./umi devices sync --write-token-file /run/secrets/osmo-platform-write-token

python -m tools.upload_visualization_bundle \
  --timeline timeline.json --video front-video.mp4 --scene scene.html \
  --write-token-file /run/secrets/osmo-platform-write-token
```

For rotation, create a new file with `umask 077`, atomically replace the old
file, restart the service, and update authorized clients. Verify that public
reads still work and that an unauthenticated write fails before sending a body:

```bash
curl -fsS http://127.0.0.1:7865/healthz
curl -i -X POST http://127.0.0.1:7865/api/projects
```

The second command must return `401` with `WWW-Authenticate: Bearer`; it must
not create a project. Valid authenticated uploads remain subject to the
existing JSON, video, scene, and inventory size limits.
