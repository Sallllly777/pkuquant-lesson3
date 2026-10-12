from __future__ import annotations

import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .data import parse_time
from .engine import Replay


def serve(archive, symbol, port):
    token = secrets.token_hex(24)
    lock = threading.RLock()
    replay = Replay(archive.load(symbol), initial_position=1000)
    replay.advance_to(parse_time("09:30:05"))
    html = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8").replace("__TOKEN__", token)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data, content_type="application/json; charset=utf-8"):
            body = data.encode("utf-8") if isinstance(data, str) else json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def allowed_host(self):
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def do_GET(self):
            if not self.allowed_host():
                self.reply(403, {"error": "Local access only"}); return
            with lock:
                if self.path == "/":
                    self.reply(200, html, "text/html; charset=utf-8")
                elif self.path == "/api/state":
                    self.reply(200, {**replay.state(), "symbols": archive.symbols})
                else:
                    self.reply(404, {"error": "Not found"})

        def do_POST(self):
            nonlocal replay
            if not self.allowed_host() or self.headers.get("X-Session-Token") != token:
                self.reply(403, {"error": "Invalid local session"}); return
            if self.path != "/api/action":
                self.reply(404, {"error": "Not found"}); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 8192:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("Expected JSON object")
                with lock:
                    op = request.get("op")
                    if op == "reset":
                        # Build a replacement first: failed resets leave the current account intact.
                        new = Replay(archive.load(int(request["symbol"])), cash=request.get("cash", 1000000), initial_position=request.get("position", 1000))
                        target = parse_time(request.get("time", "09:30:05"))
                        new.advance_to(target)
                        replay = new
                    elif op == "advance":
                        replay.advance_to(parse_time(request["time"]))
                    elif op == "step":
                        replay.step(request.get("groups", 1))
                    elif op == "order":
                        replay.submit(request["side"], request["price"], request["quantity"])
                    elif op == "cancel":
                        replay.cancel(request["id"])
                    else:
                        raise ValueError("Unknown action")
                    self.reply(200, {**replay.state(), "symbols": archive.symbols})
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {"error": str(exc)})

        def log_message(self, fmt, *args):
            if args and str(args[1]) != "200":
                super().log_message(fmt, *args)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Paper trading dashboard: http://127.0.0.1:{port}", flush=True)
    print("Local simulation only. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
