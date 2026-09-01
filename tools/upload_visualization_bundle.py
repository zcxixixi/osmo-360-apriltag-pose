#!/usr/bin/env python3
"""Upload a processed visualization bundle and print the animation result as JSON."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from osmo360.pipeline.platform_auth import (
    open_http_no_redirect,
    platform_authorization_headers,
    validate_http_url,
)
from tools._root import ROOT

DEFAULT_SERVER = os.environ.get("OSMO_VISUALIZATION_URL", "http://192.168.111.62:7865")
DEFAULT_SCENE = ROOT / "dual_gripper_3d/single_gripper_scene.html"
PROJECT_ID_PATTERN = re.compile(r"^[a-z0-9-]{1,64}$")


def request_json(
    url: str,
    method: str = "GET",
    payload: dict | None = None,
    authorization: dict[str, str] | None = None,
) -> dict:
    validate_http_url(url)
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    headers.update(authorization or {})
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with open_http_no_redirect(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        try:
            message = json.load(error).get("error", error.reason)
        except (json.JSONDecodeError, AttributeError):
            message = error.reason
        raise RuntimeError(f"{method} {url}: HTTP {error.code}: {message}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"{method} {url}: {error.reason}") from error


def put_file(
    url: str,
    file: Path,
    content_type: str,
    authorization: dict[str, str] | None = None,
) -> dict:
    validate_http_url(url)
    parsed = urlparse(url)
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=120)
    target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
    size = file.stat().st_size
    try:
        connection.putrequest("PUT", target)
        connection.putheader("Content-Type", content_type)
        connection.putheader("Content-Length", str(size))
        for key, value in (authorization or {}).items():
            connection.putheader(key, value)
        connection.endheaders()
        with file.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                connection.send(chunk)
        response = connection.getresponse()
        body = response.read()
        try:
            result = json.loads(body)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"PUT {url}: invalid JSON response") from error
        if not 200 <= response.status < 300:
            raise RuntimeError(f"PUT {url}: HTTP {response.status}: {result.get('error', response.reason)}")
        return result
    finally:
        connection.close()


def project_upload_urls(server: str, project: dict) -> dict[str, str]:
    """Build upload URLs locally instead of trusting absolute links in a response."""
    validate_http_url(server)
    project_id = str(project.get("id", ""))
    if not PROJECT_ID_PATTERN.fullmatch(project_id):
        raise RuntimeError("platform returned an invalid project id")
    base = f"{server.rstrip('/')}/api/projects/{project_id}"
    return {
        "timeline": f"{base}/timeline",
        "video": f"{base}/video",
        "scene": f"{base}/scene",
        "publish": f"{base}/publish",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload timeline.json, front-video.mp4, and a versioned scene to OSMO Motion Studio."
    )
    parser.add_argument("--timeline", required=True, type=Path, help="Processed WebGL timeline JSON")
    parser.add_argument("--video", required=True, type=Path, help="Synchronized front-lens MP4")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="Versioned single-gripper renderer")
    parser.add_argument("--name", help="Animation name; defaults to the timeline filename")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"Platform base URL (default: {DEFAULT_SERVER})")
    parser.add_argument(
        "--write-token-file",
        type=Path,
        help=(
            "private bearer-token file; otherwise use OSMO_PLATFORM_WRITE_TOKEN_FILE "
            "or ~/.config/osmo360/platform-write-token"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timeline = args.timeline.resolve()
    video = args.video.resolve()
    scene = args.scene.resolve()
    for file in (timeline, video, scene):
        if not file.is_file():
            raise RuntimeError(f"input file not found: {file}")
    server = args.server.rstrip("/")
    capabilities = request_json(f"{server}/api/capabilities")
    if capabilities.get("api_version") != "v1":
        raise RuntimeError(f"unsupported platform API: {capabilities.get('api_version')!r}")
    authorization = platform_authorization_headers(args.write_token_file)
    created = request_json(
        f"{server}/api/projects",
        "POST",
        {"name": args.name or timeline.stem},
        authorization,
    )["project"]
    upload_urls = project_upload_urls(server, created)
    put_file(
        upload_urls["timeline"], timeline, "application/json", authorization
    )
    put_file(upload_urls["video"], video, "video/mp4", authorization)
    put_file(upload_urls["scene"], scene, "text/html; charset=utf-8", authorization)
    published = request_json(
        upload_urls["publish"], "POST", authorization=authorization
    )["project"]
    output = {
        "api_version": "v1",
        "project_id": published["id"],
        "status": published["status"],
        "view_url": published["view_url"],
        "summary": published["summary"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
