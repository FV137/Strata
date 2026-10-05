"""Bounded image uploads and opt-in HTTPS fetches; never read a request's local path."""
from __future__ import annotations

import base64
import binascii
import http.client
import io
import ipaddress
import re
import socket
import ssl
import threading
import time
import warnings
from urllib.parse import urlsplit

MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_IMAGE_DIMENSION = 16_000
MAX_IMAGES = 16
FETCH_TIMEOUT_S = 15.0
_TRANSITION_NETS = tuple(ipaddress.ip_network(n) for n in
                         ("64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32"))


def _https_url(value: str):
    if not isinstance(value, str) or not value or len(value) > 8192 or not value.isascii() or \
            any(ord(c) <= 32 or ord(c) == 127 for c in value) or "\\" in value:
        raise ValueError("image URLs must be ASCII HTTPS URLs without spaces or control characters")
    try:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username is not None or \
                parsed.password is not None or parsed.fragment or "#" in value:
            raise ValueError()
        host = parsed.hostname.lower()
        port = parsed.port if parsed.port is not None else 443
        if not 1 <= port <= 65535 or "%" in host or host.endswith("."):
            raise ValueError()
        try:
            host = str(ipaddress.ip_address(host))
        except ValueError:
            if len(host) > 253 or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                                      for label in host.split(".")):
                raise ValueError()
        return parsed, host, port
    except (ValueError, UnicodeError):
        raise ValueError("image URLs require an HTTPS host, without credentials or fragments") from None


def image_origins_of(value) -> frozenset[tuple[str, int]]:
    """Validate administrator config before starting an encoder; an omitted list denies all URLs."""
    if value is None:
        return frozenset()
    if not isinstance(value, list):
        raise ValueError("vision.image_origins must be a list of exact HTTPS origins")
    origins = set()
    for origin in value:
        parsed, host, port = _https_url(origin)
        if parsed.path not in ("", "/") or parsed.query or "?" in origin:
            raise ValueError("vision.image_origins entries must contain only an HTTPS origin")
        origins.add((host, port))
    return frozenset(origins)


def _public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if not ip.is_global or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.is_site_local or ip.ipv4_mapped is not None or any(ip in net for net in _TRANSITION_NETS):
            return False
    return True


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect only to a checked literal address, keeping the original host for verified TLS and Host."""
    def __init__(self, host, port, address, deadline):
        super().__init__(host, port, timeout=FETCH_TIMEOUT_S, context=ssl.create_default_context())
        self.address, self.deadline = address, deadline

    def connect(self):
        family, kind, proto, _, sockaddr = self.address
        sock = socket.socket(family, kind, proto)
        self.sock = sock
        try:
            sock.settimeout(max(0.001, self.deadline - time.monotonic()))
            sock.connect(sockaddr)  # No second DNS lookup, proxy, or redirect.
            sock.settimeout(max(0.001, self.deadline - time.monotonic()))
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


def _fetch(source: str, origins) -> bytes:
    parsed, host, port = _https_url(source)
    if (host, port) not in origins:
        raise ValueError("remote images are disabled unless their exact HTTPS origin is in vision.image_origins")
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
        if not addresses or any(a[0] not in (socket.AF_INET, socket.AF_INET6) or
                                not _public_address(a[4][0]) for a in addresses):
            raise ValueError("image URLs must resolve only to public unicast addresses")
        deadline = time.monotonic() + FETCH_TIMEOUT_S
        conn = _PinnedHTTPSConnection(host, port, addresses[0], deadline)
        expired = threading.Event()
        # The watchdog interrupts slow headers and trickled bodies too. A socket inactivity timeout alone
        # can be kept alive indefinitely by sending one byte before each timeout.
        sockets = []

        def expire():
            expired.set()
            for sock in sockets + [conn.sock]:
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

        timer = threading.Timer(FETCH_TIMEOUT_S, expire)
        timer.daemon = True
        timer.start()
        try:
            conn.connect()
            sockets.append(conn.sock)  # HTTPConnection can clear .sock for a Connection: close response.
            if expired.is_set() or time.monotonic() >= deadline:
                raise ValueError("image download deadline exceeded")
            conn.request("GET", (parsed.path or "/") + ("?" + parsed.query if parsed.query else ""),
                         headers={"User-Agent": "strata", "Accept": "image/*", "Accept-Encoding": "identity"})
            with conn.getresponse() as response:
                if response.status != 200:
                    raise ValueError("image URL must return HTTP 200; redirects are not followed")
                encoding = response.headers.get_all("Content-Encoding", [])
                if encoding and encoding != ["identity"]:
                    raise ValueError("compressed HTTP image responses are not accepted")
                if not response.headers.get("Content-Type", "").lower().startswith("image/"):
                    raise ValueError("image URL must return an image Content-Type")
                lengths = response.headers.get_all("Content-Length", [])
                transfer = response.headers.get_all("Transfer-Encoding", [])
                if transfer and (lengths or len(transfer) != 1 or transfer[0].lower() != "chunked"):
                    raise ValueError("ambiguous image response framing")
                if lengths and (len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,12}", lengths[0]) or
                                int(lengths[0]) > MAX_IMAGE_BYTES):
                    raise ValueError("invalid or oversized image Content-Length")
                data = bytearray()
                while True:
                    if expired.is_set() or time.monotonic() >= deadline:
                        raise ValueError("image download deadline exceeded")
                    chunk = response.read1(min(65536, MAX_IMAGE_BYTES + 1 - len(data)))
                    if not chunk:
                        break
                    data.extend(chunk)
                    if len(data) > MAX_IMAGE_BYTES:
                        raise ValueError("image exceeds the 16 MiB limit")
                if expired.is_set() or time.monotonic() >= deadline:
                    raise ValueError("image download deadline exceeded")
                if not data or (lengths and len(data) != int(lengths[0])):
                    raise ValueError("empty or truncated image response")
                return bytes(data)
        finally:
            timer.cancel()
            conn.close()
    except (OSError, http.client.HTTPException) as exc:
        raise ValueError("image HTTPS download failed") from exc


def load(source: str, origins=()) -> bytes:
    if not isinstance(source, str):
        raise ValueError("an image source must be a string")
    if source.startswith("data:"):
        header, separator, encoded = source.partition(",")
        if not separator or not re.fullmatch(r"data:image/[A-Za-z0-9.+-]+;base64", header) or \
                not encoded or len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise ValueError("an uploaded image must be a base64 image data URL of at most 16 MiB")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("invalid image base64 data") from None
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds the 16 MiB limit")
        return data
    if source.startswith(("https:", "http:")):
        return _fetch(source, origins)
    raise ValueError("upload images as base64 data URLs; server-local paths and file: URLs are not accepted")


def normalize(data: bytes) -> bytes:
    """Decode every format within pixel limits and give the native encoder a fresh RGB PNG."""
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("an image must contain 1 byte to 16 MiB")
    try:
        from PIL import Image
    except ImportError:
        raise ValueError("image validation requires Pillow (install requirements.txt)") from None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=("JPEG", "PNG", "BMP", "GIF", "WEBP", "TIFF", "AVIF")) as im:
                width, height = im.size
                if width < 1 or height < 1 or width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION or \
                        width * height > MAX_IMAGE_PIXELS:
                    raise ValueError("image exceeds 16 million pixels or 16,000 pixels per dimension")
                im.load()  # The first frame of an animated image, as before.
                rgba = im.convert("RGBA")
                clean = Image.new("RGB", im.size, (255, 255, 255))
                clean.paste(rgba, mask=rgba.getchannel("A"))
                out = io.BytesIO()
                clean.save(out, format="PNG")  # No source metadata, paths, or profiles reach the native codec.
                result = out.getvalue()
                if len(result) > MAX_IMAGE_BYTES:
                    raise ValueError("normalized image exceeds the 16 MiB limit")
                return result
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError("image is invalid or exceeds the image byte/dimension/pixel limits") from None
