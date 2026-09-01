from __future__ import annotations

import socket
import threading
from contextlib import contextmanager
from http import HTTPStatus
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlencode

from osmo360.pipeline.review_gateway import (
    SESSION_COOKIE,
    create_review_gateway_server,
)

TOKEN = "r" * 64


class BackendHandler(BaseHTTPRequestHandler):
    calls: ClassVar[list[dict]] = []

    def do_GET(self):
        type(self).calls.append({
            "method": "GET",
            "path": self.path,
            "cookie": self.headers.get("Cookie"),
            "authorization": self.headers.get("Authorization"),
            "range": self.headers.get("Range"),
        })
        if self.path == "/":
            body = b"<!doctype html><html><head><title>Review</title></head><body>ok</body></html>"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        elif self.headers.get("Range"):
            body = b"data"
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Range", "bytes 0-3/4")
        else:
            body = b'{"items":[]}'
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        type(self).calls.append({
            "method": "POST",
            "path": self.path,
            "cookie": self.headers.get("Cookie"),
            "authorization": self.headers.get("Authorization"),
            "body": body,
        })
        response = b'{"saved":true}'
        self.send_response(HTTPStatus.CREATED)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *args):
        pass


@contextmanager
def running_gateway(tmp_path: Path):
    BackendHandler.calls = []
    token_file = tmp_path / "review-token"
    token_file.write_text(TOKEN + "\n", encoding="utf-8")
    token_file.chmod(0o600)
    backend = ThreadingHTTPServer(("127.0.0.1", 0), BackendHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    gateway = create_review_gateway_server(
        host="127.0.0.1",
        port=0,
        backend_host="127.0.0.1",
        backend_port=backend.server_port,
        public_origin="http://review.test:7869",
        token_file=token_file,
        max_workers=4,
    )
    gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
    gateway_thread.start()
    try:
        yield gateway
    finally:
        gateway.shutdown()
        backend.shutdown()
        gateway.server_close()
        backend.server_close()
        gateway_thread.join(timeout=2)
        backend_thread.join(timeout=2)


def request(gateway, method, path, *, headers=None, body=None):
    connection = HTTPConnection("127.0.0.1", gateway.server_port, timeout=3)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    result = response.status, dict(response.getheaders()), payload
    connection.close()
    return result


def login(gateway):
    body = urlencode({"token": TOKEN}).encode()
    status, headers, _ = request(
        gateway,
        "POST",
        "/login",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Content-Length": str(len(body)),
        },
        body=body,
    )
    assert status == HTTPStatus.SEE_OTHER
    return headers["Set-Cookie"].split(";", 1)[0]


def test_unauthenticated_requests_do_not_reach_backend(tmp_path):
    with running_gateway(tmp_path) as gateway:
        root_status, root_headers, _ = request(gateway, "GET", "/")
        api_status, api_headers, _ = request(gateway, "GET", "/api/items")

    assert root_status == HTTPStatus.SEE_OTHER
    assert root_headers["Location"] == "/login"
    assert api_status == HTTPStatus.UNAUTHORIZED
    assert api_headers["WWW-Authenticate"].startswith("Bearer ")
    assert api_headers["Connection"] == "close"
    assert BackendHandler.calls == []


def test_login_rejects_bad_token_and_sets_hardened_session_cookie(tmp_path):
    with running_gateway(tmp_path) as gateway:
        bad = urlencode({"token": "wrong"}).encode()
        bad_status, _, _ = request(
            gateway,
            "POST",
            "/login",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(bad)),
            },
            body=bad,
        )
        body = urlencode({"token": TOKEN}).encode()
        status, headers, _ = request(
            gateway,
            "POST",
            "/login",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
            },
            body=body,
        )

    assert bad_status == HTTPStatus.UNAUTHORIZED
    assert status == HTTPStatus.SEE_OTHER
    assert headers["Location"] == "/"
    assert SESSION_COOKIE in headers["Set-Cookie"]
    assert "HttpOnly" in headers["Set-Cookie"]
    assert "SameSite=Strict" in headers["Set-Cookie"]
    assert TOKEN not in headers["Set-Cookie"]


def test_authenticated_page_gets_csrf_injection_and_security_headers(tmp_path):
    with running_gateway(tmp_path) as gateway:
        cookie = login(gateway)
        status, headers, body = request(
            gateway, "GET", "/", headers={"Cookie": cookie}
        )

    assert status == HTTPStatus.OK
    assert b"X-CSRF-Token" in body
    assert TOKEN.encode() not in body
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert BackendHandler.calls[0]["cookie"] is None
    assert BackendHandler.calls[0]["authorization"] is None


def test_cookie_post_requires_csrf_but_bearer_can_write(tmp_path):
    with running_gateway(tmp_path) as gateway:
        cookie = login(gateway)
        rejected, _, _ = request(
            gateway,
            "POST",
            "/api/items/test/review",
            headers={
                "Cookie": cookie,
                "Content-Type": "application/json",
                "Content-Length": "2",
            },
            body=b"{}",
        )
        csrf = gateway.config.csrf_value
        accepted, _, payload = request(
            gateway,
            "POST",
            "/api/items/test/review",
            headers={
                "Cookie": cookie,
                "Content-Type": "application/json",
                "Content-Length": "2",
                "X-CSRF-Token": csrf,
                "Origin": gateway.config.public_origin,
                "Sec-Fetch-Site": "same-origin",
            },
            body=b"{}",
        )
        bearer, _, _ = request(
            gateway,
            "POST",
            "/api/items/test/review",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
                "Content-Length": "2",
            },
            body=b"{}",
        )

    assert rejected == HTTPStatus.FORBIDDEN
    assert accepted == HTTPStatus.CREATED
    assert payload == b'{"saved":true}'
    assert bearer == HTTPStatus.CREATED
    posts = [call for call in BackendHandler.calls if call["method"] == "POST"]
    assert len(posts) == 2
    assert all(call["cookie"] is None for call in posts)
    assert all(call["authorization"] is None for call in posts)


def test_range_is_forwarded_and_large_body_is_rejected_before_backend(tmp_path):
    with running_gateway(tmp_path) as gateway:
        status, headers, body = request(
            gateway,
            "GET",
            "/api/items/test/video?role=left",
            headers={"Authorization": f"Bearer {TOKEN}", "Range": "bytes=0-3"},
        )
        too_large, _, _ = request(
            gateway,
            "POST",
            "/api/items/test/review",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
                "Content-Length": "65537",
            },
        )

    assert status == HTTPStatus.PARTIAL_CONTENT
    assert headers["Content-Range"] == "bytes 0-3/4"
    assert body == b"data"
    assert too_large == HTTPStatus.BAD_REQUEST
    assert [call["method"] for call in BackendHandler.calls] == ["GET"]


def test_expect_continue_is_rejected_before_unauthenticated_body(tmp_path):
    with (
        running_gateway(tmp_path) as gateway,
        socket.create_connection(
            ("127.0.0.1", gateway.server_port), timeout=3
        ) as client,
    ):
        client.sendall(
            b"POST /api/items/test/review HTTP/1.1\r\n"
            b"Host: review.test:7869\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 1000000000\r\n"
            b"Expect: 100-continue\r\n\r\n"
        )
        response = client.recv(4096)

    assert b" 100 Continue" not in response
    assert b" 400 " in response
    assert BackendHandler.calls == []
