"""Network access and request framing regressions, with a mock engine and local sockets only."""
from __future__ import annotations

import contextlib
import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve import server
from serve.access import api_key_of, require_key_for_bind
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]


def service():
    tok = ByteTokenizer()
    return Service(MockEngine(tok, "</think>\n\nok", max_context=4096), tok,
                   ChatTemplate(ROOT / "serve/chat_template.jinja"))


class BindPolicy(unittest.TestCase):
    def test_only_loopback_is_allowed_without_a_key(self):
        for host in ("127.0.0.1", "127.0.0.2", "::1", "localhost"):
            require_key_for_bind(host, "")
        for host in ("0.0.0.0", "::", "192.168.1.20", "localhost.example.com", ""):
            with self.subTest(host=host), self.assertRaises(ValueError):
                require_key_for_bind(host, "")

    def test_explicit_keys_override_config_without_empty_fallback(self):
        with mock.patch.dict(os.environ, {"STRATA_API_KEY": "env-test-key"}):
            self.assertEqual(api_key_of(None, {"api_key": "saved-test-key"}), "env-test-key")
            self.assertEqual(api_key_of("cli-test-key", {"api_key": "saved-test-key"}), "cli-test-key")
            with self.assertRaises(ValueError):
                api_key_of("", {"api_key": "saved-test-key"})

    def test_wildcard_bind_requires_a_key(self):
        httpd = None
        try:
            with self.assertRaisesRegex(ValueError, "API key"):
                httpd = serve(service(), host="0.0.0.0", port=0)
        finally:
            if httpd:
                httpd.shutdown()
                httpd.server_close()

    def test_wildcard_with_a_key_still_authenticates_requests(self):
        svc = service()
        svc.api_key = "test-secret"
        httpd = serve(svc, host="0.0.0.0", port=0)
        try:
            for headers, expected in (({}, 401), ({"Authorization": "Bearer test-secret"}, 200)):
                with contextlib.closing(http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=2)) as c:
                    c.request("GET", "/v1/models", headers=headers)
                    r = c.getresponse()
                    self.assertEqual(r.status, expected)
                    r.read()
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_bad_access_settings_fail_before_loading_a_model(self):
        cases = [({"host": "0.0.0.0"}, [], {}),
                 ({"host": "127.0.0.1", "api_key": "   "}, [], {}),
                 ({"host": "0.0.0.0", "api_key": 123}, [], {}),
                 ({"host": "0.0.0.0", "api_key": "saved-key"}, ["--api-key", ""], {}),
                 ({"host": "0.0.0.0", "api_key": "saved-key"}, [], {"STRATA_API_KEY": ""})]
        for cfg, args, extra in cases:
            with self.subTest(cfg=cfg, args=args, env=extra), tempfile.TemporaryDirectory() as d:
                p = Path(d) / "config.json"
                p.write_text(json.dumps(cfg), encoding="utf-8")
                env = {k: v for k, v in os.environ.items() if k != "STRATA_API_KEY"}
                run = subprocess.run([sys.executable, "-m", "serve.server", "--engine", "strata", "--port", "0",
                                      "--config", str(p), *args], cwd=ROOT, env={**env, **extra},
                                     capture_output=True, text=True, timeout=5)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("API key", run.stderr)
                self.assertNotIn("loading the", run.stdout)


class RequestBodies(unittest.TestCase):
    def setUp(self):
        self.svc = service()
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_port

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, headers, body=b"{}", finish=True):
        with socket.create_connection(("127.0.0.1", self.port), timeout=3) as s:
            wire = f"POST {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
            wire += "".join(f"{k}: {v}\r\n" for k, v in headers)
            s.sendall(wire.encode() + b"\r\n" + body)
            if finish:
                s.shutdown(socket.SHUT_WR)
            with http.client.HTTPResponse(s) as response:
                try:
                    response.begin()
                except socket.timeout:
                    self.fail("server waited for the body instead of rejecting the request")
                return response.status, json.loads(response.read())

    def test_negative_length_does_not_read_until_eof(self):
        self.assertEqual(self.request("/settings", [("Content-Length", "-1")])[0], 400)

    def test_bad_length_is_a_json_error(self):
        status, body = self.request("/settings", [("Content-Length", "garbage")])
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_oversized_bodies_are_rejected_without_waiting_for_bytes(self):
        for path, length in (("/v1/chat/completions", 33554433), ("/v1/messages", 33554433),
                             ("/v1/responses", 33554433), ("/settings", 65537), ("/config", 65537),
                             ("/load", 65537), ("/v1/load", 65537)):
            with self.subTest(path=path):
                self.assertEqual(self.request(path, [("Content-Length", str(length))], body=b"", finish=False)[0], 413)

    def test_ambiguous_framing_is_rejected(self):
        for headers in ([('Content-Length', '2'), ('Content-Length', '2')],
                        [('Transfer-Encoding', 'chunked')],
                        [('Transfer-Encoding', 'chunked'), ('Content-Length', '2')]):
            with self.subTest(headers=headers):
                self.assertEqual(self.request("/settings", headers)[0], 400)

    def test_truncated_valid_json_does_not_change_settings(self):
        body = b'{"defaults":{"temperature":0.7}}'
        status, _ = self.request("/settings", [("Content-Length", str(len(body) + 10))], body)
        self.assertEqual(status, 400)
        self.assertEqual(self.svc.shared, {})

    def test_body_deadline_returns_a_timeout(self):
        with mock.patch.object(server, "REQUEST_BODY_TIMEOUT_S", 0.15):
            status, _ = self.request("/settings", [("Content-Length", "20")], body=b"{", finish=False)
        self.assertEqual(status, 408)

    def test_dripping_bytes_does_not_extend_the_deadline(self):
        done = threading.Event()
        with mock.patch.object(server, "REQUEST_BODY_TIMEOUT_S", 0.3), \
                socket.create_connection(("127.0.0.1", self.port), timeout=3) as s:
            s.sendall(b"POST /settings HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                      b"Content-Length: 100\r\n\r\n")

            def drip():
                while not done.wait(0.04):
                    try:
                        s.sendall(b" ")
                    except OSError:
                        break

            sender = threading.Thread(target=drip, daemon=True)
            sender.start()
            try:
                with http.client.HTTPResponse(s) as response:
                    response.begin()
                    self.assertEqual(response.status, 408)
                    response.read()
            finally:
                done.set()
                sender.join(1)

    def test_normal_chat_still_works(self):
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}).encode()
        code, data = self.request("/v1/chat/completions", [("Content-Length", str(len(body)))], body)
        self.assertEqual(code, 200)
        self.assertEqual(data["choices"][0]["message"]["content"], "ok")


if __name__ == "__main__":
    unittest.main()
