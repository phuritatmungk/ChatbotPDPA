"""Path-routing reverse proxy.

Forwards:
  /v1/*               -> http://localhost:8080  (llama-server)
  /rerank, /health    -> http://localhost:8090  (rerank_server)

Listens on PROXY_PORT (default 8000). Intended to sit behind a single ngrok tunnel
so the public URL stays stable while the static domain is fixed.
"""
import os
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

PROXY_PORT = int(os.getenv("PROXY_PORT", "8000"))
LLAMA_UPSTREAM = os.getenv("LLAMA_UPSTREAM", "http://localhost:8080")
RERANK_UPSTREAM = os.getenv("RERANK_UPSTREAM", "http://localhost:8090")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("proxy")

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


def pick_upstream(path: str) -> str | None:
    if path.startswith("/v1/") or path == "/v1":
        return LLAMA_UPSTREAM
    if path in ("/rerank", "/health", "/"):
        return RERANK_UPSTREAM
    return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self, method: str):
        upstream = pick_upstream(self.path)
        if upstream is None:
            self.send_error(404, "no route")
            return

        url = upstream + self.path
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length > 0 else None

        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        req = Request(url, data=body, method=method, headers=headers)

        try:
            with urlopen(req, timeout=300) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() in HOP_BY_HOP:
                        continue
                    self.send_header(k, v)
                payload = resp.read()
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except HTTPError as e:
            payload = e.read() if e.fp else b""
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() in HOP_BY_HOP:
                    continue
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except URLError as e:
            log.warning(f"upstream unreachable for {url}: {e}")
            self.send_error(502, f"upstream unreachable: {e}")

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def log_message(self, format, *args):
        log.info("%s %s", self.address_string(), format % args)


if __name__ == "__main__":
    log.info(f"proxy on :{PROXY_PORT}  →  /v1/* → {LLAMA_UPSTREAM}, /rerank|/health → {RERANK_UPSTREAM}")
    ThreadingHTTPServer(("0.0.0.0", PROXY_PORT), Handler).serve_forever()
