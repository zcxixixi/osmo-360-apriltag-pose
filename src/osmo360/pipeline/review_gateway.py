"""Authenticated, bounded reverse gateway for the legacy alignment-review UI."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.client import HTTPConnection, HTTPException
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
from urllib.parse import parse_qs, urlsplit

from .platform_auth import load_platform_write_token, validate_http_url

SESSION_COOKIE = "osmo_alignment_review_session"
MAX_LOGIN_BODY_BYTES = 4096
MAX_PROXY_BODY_BYTES = 65536
MAX_HTML_BYTES = 1024 * 1024
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
FORWARDED_REQUEST_HEADERS = {
    "accept",
    "accept-language",
    "cache-control",
    "content-type",
    "if-modified-since",
    "if-none-match",
    "range",
    "user-agent",
}


def _derived_secret(token: str, purpose: str) -> str:
    return hmac.new(
        token.encode("utf-8"), purpose.encode("ascii"), hashlib.sha256
    ).hexdigest()


def _validated_origin(value: str) -> str:
    validate_http_url(value)
    parsed = urlsplit(value)
    if parsed.path not in {"", "/"} or parsed.query:
        raise ValueError("public origin must not include a path or query")
    return f"{parsed.scheme}://{parsed.netloc}"


@dataclass(frozen=True)
class ReviewGatewayConfig:
    backend_host: str
    backend_port: int
    public_origin: str
    token: str
    backend_timeout_s: float = 30.0
    secure_cookie: bool = False

    @property
    def session_value(self) -> str:
        return _derived_secret(self.token, "alignment-review-session-v1")

    @property
    def csrf_value(self) -> str:
        return _derived_secret(self.token, "alignment-review-csrf-v1")


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server with a hard request-worker ceiling."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, max_workers: int = 16, **kwargs):
        if not 1 <= max_workers <= 128:
            raise ValueError("max_workers must be between 1 and 128")
        self._worker_slots = threading.BoundedSemaphore(max_workers)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address) -> None:
        self._worker_slots.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._worker_slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._worker_slots.release()


class ReviewGatewayServer(BoundedThreadingHTTPServer):
    config: ReviewGatewayConfig


class ReviewGatewayHandler(BaseHTTPRequestHandler):
    server: ReviewGatewayServer
    server_version = "OsmoReviewGateway/1.0"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.request.settimeout(30.0)

    def _standard_headers(self, *, html: bool = False) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if html:
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "media-src 'self'; connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
            )

    def _send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        *,
        html: bool = False,
        extra_headers: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self._standard_headers(html=html)
        for name, value in extra_headers:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, value: dict[str, str]) -> None:
        self._send_bytes(
            status,
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            extra_headers=(
                ("WWW-Authenticate", 'Bearer realm="alignment-review"'),
            )
            if status == HTTPStatus.UNAUTHORIZED
            else (),
        )

    def _redirect(self, location: str, *, clear_cookie: bool = False) -> None:
        headers = [("Location", location)]
        if clear_cookie:
            headers.append(
                (
                    "Set-Cookie",
                    f"{SESSION_COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict",
                )
            )
        self._send_bytes(
            HTTPStatus.SEE_OTHER,
            b"",
            "text/plain; charset=utf-8",
            extra_headers=tuple(headers),
        )

    def _login_page(self, *, failed: bool = False) -> None:
        error = "<p class=error>令牌不正确</p>" if failed else ""
        body = f"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content=\"width=device-width,initial-scale=1\">
<title>InstaUMI 审核登录</title><style>
:root{{font-family:system-ui,sans-serif;background:#f3f6f8;color:#17202a}}
main{{max-width:420px;margin:12vh auto;background:#fff;padding:28px;border-radius:16px;box-shadow:0 10px 30px #0001}}
input,button{{width:100%;box-sizing:border-box;padding:12px;margin-top:12px;border-radius:9px;border:1px solid #bccbd4;font:inherit}}
button{{background:#173d52;color:#fff;font-weight:700}}.error{{color:#a33220;font-weight:700}}
</style></head><body><main><h1>InstaUMI 人工审核</h1>
<p>请输入服务器管理员提供的审核令牌。</p>{error}
<form method=post action=/login><input type=password name=token required autofocus autocomplete=current-password>
<button type=submit>登录</button></form></main></body></html>""".encode()
        self._send_bytes(
            HTTPStatus.UNAUTHORIZED if failed else HTTPStatus.OK,
            body,
            "text/html; charset=utf-8",
            html=True,
        )

    def _authorization_mode(self) -> str | None:
        authorization = self.headers.get("Authorization", "")
        expected_bearer = f"Bearer {self.server.config.token}"
        if authorization and hmac.compare_digest(authorization, expected_bearer):
            return "bearer"
        raw_cookie = self.headers.get("Cookie", "")
        try:
            cookie = SimpleCookie(raw_cookie)
        except CookieError:
            return None
        morsel = cookie.get(SESSION_COOKIE)
        if morsel and hmac.compare_digest(
            morsel.value, self.server.config.session_value
        ):
            return "cookie"
        return None

    def _body_length(self, maximum: int) -> int:
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("transfer-encoded request bodies are not accepted")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if not 0 <= length <= maximum:
            raise ValueError("request body is too large")
        return length

    def _read_body(self, maximum: int) -> bytes:
        return self.rfile.read(self._body_length(maximum))

    def handle_expect_100(self) -> bool:
        path = urlsplit(self.path).path
        maximum = MAX_LOGIN_BODY_BYTES if path == "/login" else MAX_PROXY_BODY_BYTES
        try:
            self._body_length(maximum)
        except ValueError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return False
        if path != "/login":
            mode = self._authorization_mode()
            if mode is None:
                self._send_json(
                    HTTPStatus.UNAUTHORIZED, {"error": "authentication required"}
                )
                return False
            if mode == "cookie" and not self._csrf_is_valid():
                self._send_json(
                    HTTPStatus.FORBIDDEN, {"error": "CSRF validation failed"}
                )
                return False
        self.send_response_only(HTTPStatus.CONTINUE)
        self.end_headers()
        return True

    def _handle_login(self) -> None:
        try:
            if self.headers.get_content_type() != "application/x-www-form-urlencoded":
                raise ValueError("login requires form encoding")
            payload = parse_qs(
                self._read_body(MAX_LOGIN_BODY_BYTES).decode("utf-8"),
                keep_blank_values=True,
                strict_parsing=True,
                max_num_fields=2,
            )
            supplied = payload.get("token", [""])
            if len(supplied) != 1 or not hmac.compare_digest(
                supplied[0], self.server.config.token
            ):
                self._login_page(failed=True)
                return
        except (UnicodeDecodeError, ValueError):
            self._login_page(failed=True)
            return
        cookie = (
            f"{SESSION_COOKIE}={self.server.config.session_value}; Path=/; "
            "Max-Age=28800; HttpOnly; SameSite=Strict"
        )
        if self.server.config.secure_cookie:
            cookie += "; Secure"
        self._send_bytes(
            HTTPStatus.SEE_OTHER,
            b"",
            "text/plain; charset=utf-8",
            extra_headers=(("Location", "/"), ("Set-Cookie", cookie)),
        )

    def _csrf_is_valid(self) -> bool:
        if not hmac.compare_digest(
            self.headers.get("X-CSRF-Token", ""), self.server.config.csrf_value
        ):
            return False
        origin = self.headers.get("Origin")
        if origin and not hmac.compare_digest(origin, self.server.config.public_origin):
            return False
        fetch_site = self.headers.get("Sec-Fetch-Site")
        return fetch_site in {None, "same-origin", "none"}

    def _proxy_headers(self, body: bytes | None) -> dict[str, str]:
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() in FORWARDED_REQUEST_HEADERS
        }
        headers["Host"] = (
            f"{self.server.config.backend_host}:{self.server.config.backend_port}"
        )
        headers["Connection"] = "close"
        if body is not None:
            headers["Content-Length"] = str(len(body))
        return headers

    @staticmethod
    def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                return
            destination.write(chunk)

    def _inject_csrf(self, body: bytes) -> bytes:
        marker = b"</head>"
        if marker not in body:
            raise ValueError("review page has no head element")
        token = json.dumps(self.server.config.csrf_value)
        script = (
            "<script>(()=>{const f=window.fetch.bind(window);window.fetch=(u,o={})=>{"
            "const m=String(o.method||'GET').toUpperCase();const x=new URL(u,location.href);"
            "if(x.origin===location.origin&&!['GET','HEAD','OPTIONS'].includes(m)){"
            "const h=new Headers(o.headers||{});h.set('X-CSRF-Token',"
            + token
            + ");o={...o,headers:h}}return f(u,o)}})();</script>"
        ).encode("utf-8")
        return body.replace(marker, script + marker, 1)

    def _proxy(self, body: bytes | None = None) -> None:
        target = urlsplit(self.path)
        if target.scheme or target.netloc or not target.path.startswith("/"):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid request target"})
            return
        connection = HTTPConnection(
            self.server.config.backend_host,
            self.server.config.backend_port,
            timeout=self.server.config.backend_timeout_s,
        )
        response_started = False
        try:
            connection.request(
                self.command,
                self.path,
                body=body,
                headers=self._proxy_headers(body),
            )
            response = connection.getresponse()
            response_headers = response.getheaders()
            content_type = response.getheader("Content-Type", "")
            is_root_html = target.path == "/" and content_type.startswith("text/html")
            buffered: bytes | None = None
            if is_root_html:
                declared = response.getheader("Content-Length")
                if declared is not None and int(declared) > MAX_HTML_BYTES:
                    raise ValueError("review page exceeds gateway HTML limit")
                buffered = response.read(MAX_HTML_BYTES + 1)
                if len(buffered) > MAX_HTML_BYTES:
                    raise ValueError("review page exceeds gateway HTML limit")
                buffered = self._inject_csrf(buffered)

            self.send_response(response.status, response.reason)
            response_started = True
            for name, value in response_headers:
                lowered = name.lower()
                if lowered in HOP_BY_HOP_HEADERS or lowered in {
                    "content-length",
                    "content-security-policy",
                    "cross-origin-resource-policy",
                    "permissions-policy",
                    "referrer-policy",
                    "server",
                    "set-cookie",
                    "x-content-type-options",
                    "x-frame-options",
                }:
                    continue
                self.send_header(name, value)
            if buffered is not None:
                self.send_header("Content-Length", str(len(buffered)))
            else:
                declared = response.getheader("Content-Length")
                if declared is not None:
                    self.send_header("Content-Length", declared)
            self.send_header("Connection", "close")
            self._standard_headers(html=is_root_html)
            self.end_headers()
            if self.command != "HEAD":
                if buffered is not None:
                    self.wfile.write(buffered)
                else:
                    self._copy_stream(response, self.wfile)
        except (OSError, HTTPException, ValueError) as error:
            if not response_started and not self.wfile.closed:
                try:
                    self._send_json(
                        HTTPStatus.BAD_GATEWAY,
                        {"error": f"review backend unavailable: {type(error).__name__}"},
                    )
                except (BrokenPipeError, ConnectionError):
                    pass
        finally:
            connection.close()

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        mode = self._authorization_mode()
        if path == "/login":
            if mode:
                self._redirect("/")
            else:
                self._login_page()
            return
        if path == "/logout":
            self._redirect("/login", clear_cookie=True)
            return
        if mode is None:
            if path == "/":
                self._redirect("/login")
            else:
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "authentication required"})
            return
        self._proxy()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path == "/login":
            self._handle_login()
            return
        mode = self._authorization_mode()
        if mode is None:
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "authentication required"})
            return
        if mode == "cookie" and not self._csrf_is_valid():
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "CSRF validation failed"})
            return
        try:
            body = self._read_body(MAX_PROXY_BODY_BYTES)
        except ValueError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._proxy(body)

    def log_message(self, format: str, *args) -> None:
        return


def create_review_gateway_server(
    *,
    host: str,
    port: int,
    backend_host: str,
    backend_port: int,
    public_origin: str,
    token_file: Path,
    max_workers: int = 16,
    backend_timeout_s: float = 30.0,
) -> ReviewGatewayServer:
    if not 0 <= port <= 65535 or not 1 <= backend_port <= 65535:
        raise ValueError(
            "gateway port must be 0..65535 and backend port must be 1..65535"
        )
    if backend_host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("review backend must be bound to loopback")
    origin = _validated_origin(public_origin)
    token = load_platform_write_token(token_file)
    server = ReviewGatewayServer(
        (host, port), ReviewGatewayHandler, max_workers=max_workers
    )
    server.config = ReviewGatewayConfig(
        backend_host=backend_host,
        backend_port=backend_port,
        public_origin=origin,
        token=token,
        backend_timeout_s=backend_timeout_s,
        secure_cookie=urlsplit(origin).scheme == "https",
    )
    return server
