"""Simple CORS proxy for Replicate API."""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import urllib.request
import urllib.error
import sys
import os


class ProxyHandler(SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/predictions":
            self._proxy_replicate("POST")
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path.startswith("/api/predictions/"):
            self._proxy_replicate("GET")
        else:
            # Serve static files (index.html etc.)
            super().do_GET()

    def _proxy_replicate(self, method):
        # Read request body
        body = None
        if method == "POST":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else None

        # Build Replicate URL
        if self.path == "/api/predictions":
            url = "https://api.replicate.com/v1/predictions"
        else:
            # /api/predictions/<id> -> https://api.replicate.com/v1/predictions/<id>
            url = "https://api.replicate.com/v1" + self.path[4:]

        auth = self.headers.get("Authorization", "")

        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Authorization", auth)
        req.add_header("Content-Type", "application/json")
        if method == "POST":
            req.add_header("Prefer", "wait")

        try:
            resp = urllib.request.urlopen(req, timeout=300)
            data = resp.read()
            self.send_response(resp.status)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, Prefer")

    def log_message(self, format, *args):
        print(f"[proxy] {args[0]}" if args else "")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = HTTPServer(("127.0.0.1", port), ProxyHandler)
    print(f"Proxy running at http://localhost:{port}")
    print(f"Open http://localhost:{port}/index.html")
    server.serve_forever()
