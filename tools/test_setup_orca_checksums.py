"""Pinned gated-model metadata checks, using tiny LFS pointers instead of weights."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from tools.test_setup_unsloth import setup
from tools.test_setup_pins import Response


class Checksums(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name)
        self.sha = hashlib.sha256(b"model").hexdigest()
        self.pointer = (f"version https://git-lfs.github.com/spec/v1\n"
                        f"oid sha256:{self.sha}\nsize 5\n").encode()
        self.entry = {"name": "Qwen3.8-Flash-Next-Uncensored-Q8_0-00001-of-00001.gguf",
                      "size": 5, "pointer_size": len(self.pointer),
                      "pointer_oid": self.oid(self.pointer)}
        self.catalog = {"repository": setup.ORCA_REPO, "revision": setup.HF_REVISIONS[setup.ORCA_REPO],
                        "variants": {"Q8_0": [self.entry]}}
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(setup, "ORCA_CATALOG", self.catalog, create=True).start()

    @staticmethod
    def oid(content):
        return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()

    def resolve(self, body=None):
        with mock.patch.object(urllib.request, "urlopen", return_value=Response(
                self.pointer if body is None else body)) as opener:
            result = setup.orca_checksums("Q8_0", self.cache)
        return result, opener

    def test_authenticated_pointer_pins_weight_sha_and_cache_works_offline(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "private-test-value"}):
            result, opener = self.resolve()
        self.assertEqual(result, {self.entry["name"]: (5, self.sha)})
        req = opener.call_args.args[0]
        self.assertEqual(req.full_url, f"https://huggingface.co/{setup.ORCA_REPO}/raw/"
                         f"{self.catalog['revision']}/{self.entry['name']}")
        self.assertEqual(req.get_header("Authorization"), "Bearer private-test-value")
        redirect = urllib.request.HTTPRedirectHandler().redirect_request(
            req, None, 302, "Found", {}, "https://cdn.example.net/file")
        self.assertIsNone(redirect.get_header("Authorization"))
        with mock.patch.object(urllib.request, "urlopen", side_effect=AssertionError("network")):
            self.assertEqual(setup.orca_checksums("Q8_0", self.cache), result)
        self.assertNotIn(b"private-test-value", next(self.cache.iterdir()).read_bytes())

    def test_wrong_pointer_bytes_fail_before_any_hash_is_trusted(self):
        for body in (self.pointer.replace(self.sha.encode(), b"0" * 64), b"x" * 2048):
            with self.subTest(body=len(body)), contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaises(SystemExit):
                self.resolve(body)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_verified_git_blob_still_requires_strict_lfs_syntax_and_size(self):
        for body in (self.pointer.replace(b"size 5", b"size 6"),
                     self.pointer.replace(b"sha256:", b"sha512:"),
                     self.pointer + b"unexpected extension\n"):
            self.entry.update(pointer_size=len(body), pointer_oid=self.oid(body))
            with self.subTest(body=body), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                self.resolve(body)

    def test_tampered_cache_is_not_accepted_or_silently_refreshed(self):
        self.resolve()
        next(self.cache.iterdir()).write_bytes(b"forged")
        with mock.patch.object(urllib.request, "urlopen", side_effect=AssertionError("network")), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            setup.orca_checksums("Q8_0", self.cache)

    def test_gate_and_missing_revision_do_not_fall_back_or_print_secrets(self):
        for code in (401, 403, 404, 500):
            output = io.StringIO()
            error = urllib.error.HTTPError("https://huggingface.co/redacted", code, "private-test-value", {}, None)
            with mock.patch.object(urllib.request, "urlopen", side_effect=error) as opener, \
                    mock.patch.dict(os.environ, {"HF_TOKEN": "private-test-value"}), \
                    contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
                setup.orca_checksums("Q8_0", self.cache)
            self.assertEqual(opener.call_count, 1)
            self.assertNotIn("private-test-value", output.getvalue())
            self.assertEqual(list(self.cache.iterdir()), [])

    def test_q4_known_hashes_match_the_committed_public_pointer_ids(self):
        catalog = json.loads((Path(setup.__file__).parent / "orca-quants.json").read_text())
        for entry in catalog["variants"]["Q4_K_M"]:
            size, sha = setup.ORCA_Q4_K_M_SHARDS[entry["name"]]
            body = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {size}\n".encode()
            self.assertEqual(len(body), entry["pointer_size"])
            self.assertEqual(self.oid(body), entry["pointer_oid"])

    def test_raw_pointer_token_has_same_restricted_origin_and_repository_scope(self):
        for url in ("http://huggingface.co/", "https://mirror.example/", "https://huggingface.co.evil/",
                    "https://huggingface.co:443/", "https://huggingface.co/other/repo/"):
            with mock.patch.dict(os.environ, {"HF_TOKEN": "secret"}):
                req = setup.download_request(url + f"{setup.ORCA_REPO}/raw/main/file")
            self.assertIsNone(req.get_header("Authorization"))


class WeightReceipts(unittest.TestCase):
    def test_modified_weight_is_not_trusted_by_an_old_done_marker(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / "weight.gguf"
            path.write_bytes(b"original")
            digest = hashlib.sha256(b"original").hexdigest()
            setup.verify_sha256(path, 8, digest)
            path.write_bytes(b"modified")
            with self.assertRaises(SystemExit):
                setup.verify_sha256(path, 8, digest)

    def test_old_marker_does_not_bypass_size_or_first_identity_check(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / "weight.gguf"
            digest = hashlib.sha256(b"original").hexdigest()
            for content in (b"short", b"modified"):
                path.write_bytes(content)
                setup.mark(path, f"sha256 {digest}")
                with self.assertRaises(SystemExit):
                    setup.verify_sha256(path, 8, digest)


if __name__ == "__main__":
    unittest.main()
