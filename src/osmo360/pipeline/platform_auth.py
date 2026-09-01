from __future__ import annotations

import os
import re
from pathlib import Path


DEFAULT_WRITE_TOKEN_FILE = Path.home() / ".config/osmo360/platform-write-token"
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._~-]{43,256}$")


def load_platform_write_token(token_file: Path | None = None) -> str:
    """Load a platform write token without ever placing it in a URL."""
    inline = os.environ.get("OSMO_PLATFORM_WRITE_TOKEN", "").strip()
    configured_file = os.environ.get("OSMO_PLATFORM_WRITE_TOKEN_FILE", "").strip()
    if token_file is not None:
        if inline or configured_file:
            raise RuntimeError(
                "--write-token-file cannot be combined with OSMO_PLATFORM_WRITE_TOKEN "
                "or OSMO_PLATFORM_WRITE_TOKEN_FILE"
            )
        path = token_file.expanduser().resolve(strict=True)
        token = _read_private_token_file(path)
    elif inline:
        if configured_file:
            raise RuntimeError(
                "configure only one of OSMO_PLATFORM_WRITE_TOKEN or "
                "OSMO_PLATFORM_WRITE_TOKEN_FILE"
            )
        token = inline
    else:
        path = (
            Path(configured_file).expanduser()
            if configured_file
            else DEFAULT_WRITE_TOKEN_FILE
        )
        if not path.is_file():
            raise RuntimeError(
                "platform write token is not configured; set "
                "OSMO_PLATFORM_WRITE_TOKEN_FILE or create "
                f"{DEFAULT_WRITE_TOKEN_FILE} with mode 0600"
            )
        token = _read_private_token_file(path.resolve(strict=True))
    if not TOKEN_PATTERN.fullmatch(token):
        raise RuntimeError(
            "platform write token must contain at least 256 bits of random "
            "URL-safe text"
        )
    return token


def platform_authorization_headers(
    token_file: Path | None = None,
) -> dict[str, str]:
    return {"Authorization": f"Bearer {load_platform_write_token(token_file)}"}


def _read_private_token_file(path: Path) -> str:
    if os.name == "posix" and path.stat().st_mode & 0o077:
        raise RuntimeError(
            f"platform write-token file must have mode 0600: {path}"
        )
    return path.read_text(encoding="utf-8").strip()
