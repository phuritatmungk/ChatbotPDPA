"""Minimal HTTP server that exposes BGE reranker via POST /rerank.

Body: {"pairs": [[query, text], ...]}
Returns: {"scores": [float, ...]}
"""
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sentence_transformers import CrossEncoder

MODEL_NAME = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
PORT = int(os.getenv("RERANK_PORT", "8090"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("rerank_server")

log.info(f"Loading reranker: {MODEL_NAME}")
reranker = CrossEncoder(MODEL_NAME)
log.info("Reranker ready")


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, body: dict):
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._send_json(200, {"status": "ok", "model": MODEL_NAME})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/rerank":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            pairs = payload.get("pairs") or []
            if not isinstance(pairs, list) or not all(isinstance(p, list) and len(p) == 2 for p in pairs):
                self._send_json(400, {"error": "pairs must be a list of [query, text] pairs"})
                return
            scores = reranker.predict(pairs)
            scores_list = [float(s) for s in scores]
            self._send_json(200, {"scores": scores_list})
        except Exception as e:
            log.exception("rerank failed")
            self._send_json(500, {"error": str(e)})

    def log_message(self, format, *args):
        log.info("%s - %s", self.address_string(), format % args)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    log.info(f"Listening on http://0.0.0.0:{PORT}")
    server.serve_forever()
