"""Read-only file server for ./runs that requires a secret token.

Auth, either of:
  * header `X-Token: <token>` (scripts / curl)
  * cookie set by the login form: open /login in a browser, paste the token once.
Only sha256(token) is stored (tools/token.sha256), so the public repo never contains the
secret. Unauthenticated requests get 404 (except the bare /login form).
"""
import hashlib
import hmac
import http.cookies
import http.server
import os
import sys
import urllib.parse
from functools import partial

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "runs")
HASH = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "token.sha256")).read().strip()
COOKIE = "ladis_tok"
MAX_BODY = 4096

LOGIN_PAGE = b"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Login</title></head>
<body style="font-family:sans-serif;max-width:420px;margin:15vh auto;padding:0 16px">
<form method="post" action="/login">
<label>Token<br><input name="token" type="password" autocomplete="off" style="width:100%;padding:8px"></label>
<p><button type="submit" style="padding:8px 16px">Log in</button></p></form></body></html>"""


def _valid(tok: str) -> bool:
    return bool(tok) and hmac.compare_digest(hashlib.sha256(tok.encode()).hexdigest(), HASH)


class H(http.server.SimpleHTTPRequestHandler):
    def _send(self, code, body=b"", headers=()):
        self.send_response(code)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def _authed(self) -> bool:
        if _valid(self.headers.get("X-Token", "")):
            return True
        c = http.cookies.SimpleCookie()
        try:
            c.load(self.headers.get("Cookie", ""))
        except http.cookies.CookieError:
            return False
        return COOKIE in c and _valid(c[COOKIE].value)

    def _path(self):
        return urllib.parse.urlparse(self.path).path

    def do_GET(self):
        if self._path() == "/login":
            return self._send(200, LOGIN_PAGE, [("Content-Type", "text/html; charset=utf-8")])
        if self._path() == "/logout":
            return self._send(302, headers=[("Location", "/login"),
                                            ("Set-Cookie", f"{COOKIE}=; Max-Age=0; Path=/; Secure; HttpOnly; SameSite=Strict")])
        if not self._authed():
            return self._send(404)
        super().do_GET()

    def do_HEAD(self):
        if not self._authed():
            return self._send(404)
        super().do_HEAD()

    def do_POST(self):
        if self._path() != "/login":
            return self._send(404)
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n > MAX_BODY:
            return self._send(413)
        form = urllib.parse.parse_qs(self.rfile.read(n).decode(errors="replace"))
        tok = form.get("token", [""])[0].strip()
        if not _valid(tok):
            return self._send(302, headers=[("Location", "/login")])
        cookie = f"{COOKIE}={tok}; Max-Age=2592000; Path=/; Secure; HttpOnly; SameSite=Strict"
        self._send(302, headers=[("Location", "/"), ("Set-Cookie", cookie)])

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", 8000))),
                                    partial(H, directory=ROOT)).serve_forever()
