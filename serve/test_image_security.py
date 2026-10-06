"""Image input boundary tests: generated pictures and local fixtures, no GPU or external requests."""
import base64
import contextlib
import io
import json
import http.client
import socket
import ssl
import subprocess
import shutil
import time
import struct
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from serve import image_input

from PIL import Image
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, Vision, serve

ROOT = Path(__file__).resolve().parents[1]


def picture(fmt="PNG", size=(2, 2)):
    out = io.BytesIO()
    Image.new("RGB", size, "red").save(out, format=fmt)
    return out.getvalue()


def data_url(data, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


class ImageBoundary(unittest.TestCase):
    def test_local_paths_and_file_uris_are_not_server_reads(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "private.png"
            path.write_bytes(picture())
            for source in (str(path), path.as_uri()):
                with self.subTest(source=source), self.assertRaises(ValueError):
                    Vision.load(source)

    def test_default_rejects_urls_without_contacting_destination(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                seen.append(self.path)
                body = picture()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            with self.assertRaises(ValueError):
                Vision.load(f"http://127.0.0.1:{httpd.server_port}/private.png")
            self.assertEqual(seen, [])
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_only_strict_base64_image_data_is_accepted(self):
        for source in ("data:text/plain;base64,eA==", "data:image/png;base64,%%%%",
                       "data:image/png,not-base64", "data:image/png;base64,", "data:image/png;base64,YQ==junk"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                Vision.load(source)

    def test_uploaded_image_bytes_still_work(self):
        png = picture()
        self.assertEqual(Vision.load(data_url(png)), png)
        with Image.open(io.BytesIO(Vision.normalize(png))) as image:
            self.assertEqual(image.size, (2, 2))
            self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 0, 0))

    def test_native_format_magic_does_not_bypass_validation(self):
        for bad in (b"\x89PNG\r\n\x1a\ntruncated", b"\xff\xd8\xffnot-jpeg", b"BMnot-bmp", b"GIF89anot-gif"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Vision.normalize(bad)

    def test_declared_pixel_bomb_rejected_before_decoding(self):
        data = bytearray(picture("BMP"))
        struct.pack_into("<ii", data, 18, 20000, 20000)
        with self.assertRaises(ValueError):
            Vision.normalize(bytes(data))

    def test_large_inline_image_is_rejected(self):
        with self.assertRaises(ValueError):
            Vision.load(data_url(b"x" * (16 * 1024 * 1024 + 1)))

    def test_image_count_is_checked_before_encoding(self):
        class NoEncode:
            def encode(self, source):
                raise AssertionError("too many images reached the encoder")

        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=NoEncode())
        messages = [{"role": "user", "content": [{"type": "image", "source": data_url(picture())}] * 17}]
        with self.assertRaisesRegex(ValueError, "images"):
            svc.prepare(messages, None, {})


class ImageAPI(unittest.TestCase):
    def test_both_apis_return_400_for_denied_sources(self):
        class ValidatingVision:
            def encode(self, source):
                Vision.normalize(Vision.load(source))
                raise AssertionError("a denied image reached the native encoder")

        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=ValidatingVision())
        server = serve(svc, port=0)
        try:
            for source in ("/etc/private.png", "file:///etc/private.png", "http://127.0.0.1/private.png", 42):
                for path, part in (("/v1/chat/completions", {"type": "image_url", "image_url": {"url": source}}),
                                   ("/v1/messages", {"type": "image", "source": {"type": "url", "url": source}})):
                    with self.subTest(source=source, path=path):
                        body = {"model": "x", "max_tokens": 4,
                                "messages": [{"role": "user", "content": [part]}]}
                        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                        try:
                            conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
                            response = conn.getresponse()
                            payload = response.read()
                            self.assertEqual(response.status, 400, payload)
                        finally:
                            conn.close()
        finally:
            server.shutdown()
            server.server_close()


class URLPolicy(unittest.TestCase):
    def test_malformed_origins_fail_before_encoder_spawn(self):
        for value in ("https://example.com", ["http://example.com"], ["https://*.example.com"],
                      ["https://example.com/path"], ["https://user:pass@example.com"],
                      ["https://example.com?"], ["https://example.com#"], ["https://example.com:0"],
                      ["https://example.com."], ["https://example.com\\@evil.com"], [None]):
            with self.subTest(value=value), mock.patch.object(Vision, "_start") as start:
                with self.assertRaises(ValueError):
                    Vision({"image_origins": value})
                start.assert_not_called()

    def test_unlisted_origins_and_malformed_urls_never_resolve(self):
        allowed = image_input.image_origins_of(["https://images.example.test"])
        for url in ("https://evil.test/a", "https://images.example.test.evil/a",
                    "https://images.example.test:444/a", "https://images.example.test@evil.test/a",
                    "https://user:pass@images.example.test/a", "https://images.example.test/a#x",
                    "https://images.example.test/a\r\nInjected: yes", "http://images.example.test/a"):
            with self.subTest(url=url), mock.patch.object(socket, "getaddrinfo") as dns:
                with self.assertRaises(ValueError):
                    Vision.load(url, allowed)
                dns.assert_not_called()

    def test_private_mixed_and_translation_addresses_never_connect(self):
        allowed = image_input.image_origins_of(["https://images.example.test"])
        for ip in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "192.168.1.1", "100.64.0.1",
                   "0.0.0.0", "224.0.0.1", "::1", "fe80::1", "fec0::1", "fc00::1", "ff02::1",
                   "::ffff:127.0.0.1", "64:ff9b::7f00:1", "2002:7f00:1::1"):
            family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            addr = (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 443))
            public = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443))
            for results in ([addr], [public, addr]):
                with self.subTest(ip=ip, mixed=len(results) == 2), \
                        mock.patch.object(socket, "getaddrinfo", return_value=results), \
                        mock.patch.object(socket.socket, "connect") as connect:
                    with self.assertRaises(ValueError):
                        Vision.load("https://images.example.test/a", allowed)
                    connect.assert_not_called()

    def test_inline_validation_strips_metadata_and_flattens_transparency(self):
        from PIL.PngImagePlugin import PngInfo
        out = io.BytesIO()
        meta = PngInfo()
        meta.add_text("private", "should not reach the encoder")
        Image.new("RGBA", (2, 2), (255, 0, 0, 0)).save(out, "PNG", pnginfo=meta)
        with Image.open(io.BytesIO(Vision.normalize(out.getvalue()))) as im:
            self.assertEqual(im.mode, "RGB")
            self.assertEqual(im.getpixel((0, 0)), (255, 255, 255))
            self.assertNotIn("private", im.info)

    def test_postscript_does_not_invoke_an_external_decoder(self):
        eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 2 2\nshowpage\n"
        with mock.patch("subprocess.Popen") as process, self.assertRaises(ValueError):
            Vision.normalize(eps)
        process.assert_not_called()

    def test_malformed_anthropic_sources_raise_value_error(self):
        from serve.frontend import anthropic_to_messages
        for source in (["bad"], {"type": "base64", "data": {"not": "text"}}):
            with self.subTest(source=source), self.assertRaises(ValueError):
                anthropic_to_messages({"messages": [{"role": "user", "content": [
                    {"type": "image", "source": source}]}]})


class HTTPSImages(unittest.TestCase):
    """Real TLS/HTTP fixture, with only DNS and the network route mapped to a local server."""
    @classmethod
    def setUpClass(cls):
        openssl = shutil.which("openssl")
        if not openssl:
            raise unittest.SkipTest("TLS fixture requires the openssl CLI")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cert, cls.key = (str(Path(cls.tmp.name) / name) for name in ("cert.pem", "key.pem"))
        subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                        "-subj", "/CN=images.example.test", "-addext", "subjectAltName=DNS:images.example.test",
                        "-keyout", cls.key, "-out", cls.cert], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @contextlib.contextmanager
    def endpoint(self, mode="ok", trusted=True, byte_limit=None, deadline=2):
        seen, names, connections = [], [], []
        body = picture()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                seen.append((self.path, self.headers.get("Host"), self.headers.get("Authorization")))
                try:
                    if mode == "headers_drip":
                        for c in b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\n\r\n":
                            self.wfile.write(bytes([c]))
                            self.wfile.flush()
                            time.sleep(0.03)
                        return
                    self.send_response(302 if mode == "redirect" else 200)
                    self.send_header("Content-Type", "image/png")
                    if mode == "redirect":
                        self.send_header("Location", "http://127.0.0.1/private")
                    if mode in ("chunked_large", "ambiguous"):
                        self.send_header("Transfer-Encoding", "chunked")
                    if mode == "ambiguous":
                        self.send_header("Content-Length", str(len(body)))
                    if mode == "duplicate":
                        self.send_header("Content-Length", str(len(body)))
                    if mode not in ("chunked_large", "ambiguous", "no_length_large"):
                        self.send_header("Content-Length", str(999999999 if mode == "large_header" else len(body)))
                    if mode == "compressed":
                        self.send_header("Content-Encoding", "gzip")
                    self.end_headers()
                    if mode == "chunked_large":
                        self.wfile.write(b"80\r\n" + b"x" * 128 + b"\r\n0\r\n\r\n")
                    elif mode == "no_length_large":
                        self.wfile.write(b"x" * 128)
                    elif mode == "body_drip":
                        for c in body:
                            self.wfile.write(bytes([c]))
                            self.wfile.flush()
                            time.sleep(0.03)
                    elif mode == "truncated":
                        self.wfile.write(body[:4])
                    else:
                        self.wfile.write(body)
                except (OSError, ssl.SSLError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(self.cert, self.key)
        server_context.set_servername_callback(lambda sock, name, context: names.append(name))
        server.socket = server_context.wrap_socket(server.socket, server_side=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        # Keep certificate and hostname verification enabled; only this fixture CA is trusted.
        context = ssl.create_default_context(cafile=self.cert) if trusted else ssl.create_default_context()
        port = server.server_port
        origin = f"https://images.example.test:{port}"
        route = ("93.184.216.34", port)
        address = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", route)
        original_connect = socket.socket.connect

        def route_connect(sock, destination):
            connections.append(destination)
            if destination != route:
                raise AssertionError(f"connection to unvalidated destination: {destination}")
            return original_connect(sock, ("127.0.0.1", port))

        def resolve(*args, **kwargs):
            # A second DNS lookup would rebind to loopback; the implementation must never perform it.
            return [address] if dns.call_count == 1 else [
                (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port))]

        try:
            with mock.patch.object(socket, "getaddrinfo", side_effect=resolve) as dns, \
                    mock.patch.object(socket.socket, "connect", route_connect), \
                    mock.patch.object(image_input.ssl, "create_default_context", return_value=context), \
                    mock.patch.object(image_input, "FETCH_TIMEOUT_S", deadline), \
                    mock.patch.object(image_input, "MAX_IMAGE_BYTES", byte_limit or image_input.MAX_IMAGE_BYTES), \
                    mock.patch.dict("os.environ", {"HTTPS_PROXY": "http://127.0.0.1:1", "HTTP_PROXY": "http://127.0.0.1:1"}):
                yield origin, image_input.image_origins_of([origin]), seen, names, connections, dns
        finally:
            server.shutdown()
            server.server_close()

    def test_allowed_fetch_pins_address_and_verifies_original_tls_host(self):
        with self.endpoint() as (origin, allowed, seen, names, connections, dns):
            self.assertEqual(Vision.load(origin + "/a?x=1", allowed), picture())
            self.assertEqual(dns.call_count, 1)
            self.assertEqual(names, ["images.example.test"])
            self.assertEqual(seen, [("/a?x=1", origin[8:], None)])
            self.assertEqual(len(connections), 1)
            self.assertEqual(connections[0][0], "93.184.216.34")

    def test_untrusted_tls_certificate_is_rejected(self):
        with self.endpoint(trusted=False) as (origin, allowed, seen, *_):
            with self.assertRaises(ValueError):
                Vision.load(origin + "/a", allowed)
            self.assertEqual(seen, [])

    def test_redirects_are_rejected_without_following(self):
        with self.endpoint("redirect") as (origin, allowed, seen, _, connections, dns):
            with self.assertRaises(ValueError):
                Vision.load(origin + "/a", allowed)
            self.assertEqual(len(seen), 1)
            self.assertEqual(len(connections), 1)
            self.assertEqual(dns.call_count, 1)

    def test_bounded_and_unambiguous_responses(self):
        for mode in ("large_header", "chunked_large", "no_length_large", "duplicate", "ambiguous",
                     "compressed", "truncated"):
            with self.subTest(mode=mode), self.endpoint(mode, byte_limit=100) as (origin, allowed, *_):
                with self.assertRaises(ValueError):
                    Vision.load(origin + "/a", allowed)

    def test_deadline_stops_trickled_headers_and_body(self):
        for mode in ("headers_drip", "body_drip"):
            with self.subTest(mode=mode), self.endpoint(mode, deadline=0.18) as (origin, allowed, *_):
                started = time.monotonic()
                with self.assertRaises(ValueError):
                    Vision.load(origin + "/a", allowed)
                self.assertLess(time.monotonic() - started, 1.0)


if __name__ == "__main__":
    unittest.main()
