"""A redirect must never receive an MCP credential, session ID, or tool arguments."""
import contextlib
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from serve.mcp import HttpTransport, McpError


class Redirects(unittest.TestCase):
    def test_redirects_do_not_receive_headers_or_tool_arguments(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if self.path.startswith("/redirect/"):
                    self.send_response(int(self.path.rsplit("/", 1)[1]))
                    self.send_header("Location", f"http://127.0.0.1:{target.server_port}/destination")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    self.reply(body)

            def do_GET(self):
                self.reply(b"")

            do_DELETE = do_POST

            def reply(self, body):
                seen.append((dict(self.headers), body))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

        with contextlib.ExitStack() as stack:
            source = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            target = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            for httpd in (source, target):
                threading.Thread(target=httpd.serve_forever, daemon=True).start()
                stack.callback(httpd.server_close)
                stack.callback(httpd.shutdown)
            for code in (301, 302, 303, 307, 308):
                with self.subTest(code=code):
                    seen.clear()
                    transport = HttpTransport("test", {"url": f"http://127.0.0.1:{source.server_port}/redirect/{code}",
                                                       "headers": {"Authorization": "Bearer fake-secret",
                                                                   "X-Api-Key": "fake-api-key"}})
                    transport.session = "fake-session"
                    with self.assertRaises(McpError):
                        transport.notify("notifications/test", {"text": "private tool arguments"})
                    transport.close()
                    self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
