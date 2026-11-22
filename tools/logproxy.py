"""Recording proxy between an agent and the engine: saves every request body, passes the stream through.

    python logproxy.py LISTEN_PORT ENGINE_PORT OUT_DIR

Each POST body goes to OUT_DIR\\NNNN.json (in arrival order) with a line in OUT_DIR\\index.txt. Use it to see what
the agent really sends, e.g. where two consecutive prompts of one conversation start to differ (promptdiff.py).
"""
import http.client, http.server, os, sys, threading, time

LISTEN, ENGINE, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.makedirs(OUT, exist_ok=True)
LOCK = threading.Lock()
COUNT = [0]


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def relay(self, body):
        conn = http.client.HTTPConnection("127.0.0.1", ENGINE, timeout=3600)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        conn.request(self.command, self.path, body, headers)
        resp = conn.getresponse()
        self.send_response(resp.status)
        length = resp.getheader("Content-Length")
        for k, v in resp.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        if length is not None:
            self.send_header("Content-Length", length)
            self.end_headers()
            self.wfile.write(resp.read())
        else:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    piece = resp.read1(65536)
                    if not piece:
                        break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionError):
                pass                                  # the client stopped: closing the engine's connection stops the request
        conn.close()

    def do_GET(self):
        self.relay(None)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        with LOCK:
            COUNT[0] += 1
            n = COUNT[0]
            open(os.path.join(OUT, f"{n:04d}.json"), "wb").write(body)
            open(os.path.join(OUT, "index.txt"), "a").write(f"{n:04d} {time.strftime('%H:%M:%S')} {self.path} {len(body)}\n")
        self.relay(body)


http.server.ThreadingHTTPServer(("127.0.0.1", LISTEN), Handler).serve_forever()
