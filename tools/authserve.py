"""Read-only file server for ./runs that requires a secret token.

Clients send header `X-Token: <token>`. Only sha256(token) is stored (tools/token.sha256),
so the public repo never contains the secret. Requests without a valid token get 404.
"""
import hashlib
import hmac
import http.server
import os
import sys
from functools import partial

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "runs")
HASH = open(os.path.join(os.path.dirname(__file__), "token.sha256")).read().strip()


class H(http.server.SimpleHTTPRequestHandler):
    def _ok(self):
        tok = self.headers.get("X-Token", "")
        good = hmac.compare_digest(hashlib.sha256(tok.encode()).hexdigest(), HASH)
        if not good:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
        return good

    def do_GET(self):
        if self._ok():
            super().do_GET()

    def do_HEAD(self):
        if self._ok():
            super().do_HEAD()

    def log_message(self, *a):
        pass


http.server.ThreadingHTTPServer(("0.0.0.0", 8000), partial(H, directory=ROOT)).serve_forever()
