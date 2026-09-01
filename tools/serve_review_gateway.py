"""Serve the legacy alignment-review UI behind authenticated session control."""

from __future__ import annotations

import argparse
from pathlib import Path

from osmo360.pipeline.review_gateway import create_review_gateway_server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7869)
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=7870)
    parser.add_argument("--public-origin", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()
    server = create_review_gateway_server(
        host=args.host,
        port=args.port,
        backend_host=args.backend_host,
        backend_port=args.backend_port,
        public_origin=args.public_origin,
        token_file=args.token_file,
        max_workers=args.max_workers,
    )
    print(f"OSMO_REVIEW_GATEWAY_READY {args.public_origin}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
