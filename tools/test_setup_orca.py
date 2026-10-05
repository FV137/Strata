"""Orca Q4_K_M installer regression tests; no model downloads or GPU required."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from tools.test_setup_unsloth import Base, setup
from tools.test_setup_pins import Response

REV = "e43d00f4e2b8b40b89f75e9adeb1045ac34c8acc"
BASE = f"https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/resolve/{REV}/"
NAMES = [f"Qwen3.8-Flash-Next-Uncensored-Q4_K_M-{i:05d}-of-00003.gguf" for i in range(1, 4)]
PROJECTOR = "mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf"


class Install(Base):
    def install(self, argv=(), version="0.1.39"):
        code, out, _ = self.main(["--family", "orca", "--context", "32768", *argv],
                                 m="Q4_K_M", family=False, version=version)
        path = self.t / "strata-orca-q4_k_m.json"
        return code, out, json.loads(path.read_text()) if path.exists() else None

    def test_install_keeps_orca_weights_tokenizer_and_compat_pack_together(self):
        code, out, cfg = self.install()
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads, [BASE + name for name in NAMES])
        self.assertEqual([v[0] for v in self.verified], NAMES)
        self.assertEqual(sum(v[1] for v in self.verified), 119150722944)
        packs = [r for r in self.runs if r[1].endswith("iq_pack.py")]
        self.assertEqual(len(packs), 1)
        self.assertIn("--compat-bf16", packs[0])
        self.assertNotIn("--experts-bin", packs[0])
        self.assertTrue(cfg["tokenizer"].endswith("packs/orca-q4_k_m/tokenizer"))
        self.assertEqual(cfg["model_name"], "orcarouter-qwen3.8-flash-next-uncensored-q4_k_m")
        self.assertIn("--resident-budget-gib", cfg["args"])
        self.assertNotIn("--control-vector-scaled", cfg["args"])
        self.assertNotIn("vision", cfg)
        self.assertEqual(cfg.get("host", "127.0.0.1"), "127.0.0.1")
        self.assertEqual(setup.choices_from_config(self.t / "strata-orca-q4_k_m.json")["family"], "orca")

    def test_vision_uses_and_checks_orcas_own_projector(self):
        code, out, cfg = self.install(["--vision", "cpu"])
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads[-1], BASE + PROJECTOR)
        self.assertEqual(self.verified[-1], (PROJECTOR, 907543296,
                         "f0f352a97a62a057f3aecdb597cac664762cea2ca23f7b16ec92eee28c5572d9"))
        self.assertTrue(cfg["vision"]["mmproj"].endswith(PROJECTOR))

    def test_engine_without_q5_0_support_stops_before_model_download(self):
        code, out, cfg = self.install(version="0.1.37")
        self.assertEqual(code, 1, out)
        self.assertEqual(self.downloads, [])
        self.assertIsNone(cfg)


class Downloads(unittest.TestCase):
    def download(self, url, opener, token="local-test-token"):
        with tempfile.TemporaryDirectory() as t, mock.patch.dict(os.environ, {"HF_TOKEN": token}), \
                mock.patch.object(urllib.request, "urlopen", side_effect=opener), \
                mock.patch.object(setup.time, "sleep", side_effect=AssertionError("unexpected retry")), \
                contextlib.redirect_stdout(io.StringIO()):
            setup.download(url, Path(t) / "test.gguf")
            self.assertEqual((Path(t) / "test.gguf").read_bytes(), b"model")

    def test_token_on_head_and_get_is_not_forwarded_to_redirects(self):
        methods = []
        def opener(req, **kwargs):
            methods.append(req.get_method())
            self.assertEqual(req.get_header("Authorization"), "Bearer local-test-token")
            redirected = urllib.request.HTTPRedirectHandler().redirect_request(
                req, None, 302, "Found", {}, "https://cdn.example.net/signed-model")
            self.assertIsNone(redirected.get_header("Authorization"))
            return Response(b"model")
        self.download(BASE + NAMES[0], opener)
        self.assertEqual(methods, ["HEAD", "GET"])

    def test_token_is_not_sent_to_mirrors_plain_http_or_other_downloads(self):
        for url in (BASE.replace("https://huggingface.co", "https://mirror.example.net") + NAMES[0],
                    BASE.replace("https://", "http://") + NAMES[0],
                    "https://huggingface.co/someone/other-model/resolve/main/model.gguf",
                    "https://github.com/Niko1221/Strata/releases/download/v0.1.39/engine.zip"):
            def opener(req, **kwargs):
                self.assertIsNone(req.get_header("Authorization"))
                return Response(b"model")
            with self.subTest(url=url):
                self.download(url, opener)

    def test_denied_access_and_missing_pin_stop_without_retry_or_secret_output(self):
        for status in (401, 403, 404):
            seen = []
            def opener(req, **kwargs):
                seen.append(req.full_url)
                raise urllib.error.HTTPError(req.full_url, status, "Refused", {}, None)
            output = io.StringIO()
            with tempfile.TemporaryDirectory() as t, mock.patch.dict(os.environ, {"HF_TOKEN": "private-test-value"}), \
                    mock.patch.object(urllib.request, "urlopen", side_effect=opener), \
                    mock.patch.object(setup.time, "sleep", side_effect=AssertionError("unexpected retry")), \
                    contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
                setup.download(BASE + NAMES[0], Path(t) / "model.gguf")
            self.assertEqual(seen, [BASE + NAMES[0]])
            self.assertNotIn("private-test-value", output.getvalue())
            self.assertIn("HF_TOKEN" if status != 404 else "pinned", output.getvalue())

    def test_local_shards_require_the_published_names(self):
        self.assertIn("orca", setup.FAMILIES)
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t)
            (folder / "other-Q4_K_M-00001-of-00002.gguf").touch()
            paths = setup.gguf_dir_shards(folder, setup.FAMILIES["orca"], "Q4_K_M")
            self.assertEqual([p.name for p in paths], NAMES)


if __name__ == "__main__":
    unittest.main()
