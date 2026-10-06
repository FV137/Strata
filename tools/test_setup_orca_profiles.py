"""All published Orca profiles: installer/config round trips, memory isolation and source-engine gates.
No GPU, compiler, gated requests or large model files are used.
"""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.test_setup_unsloth import Base, quiet, setup

QUANTS = ("IQ2_M", "IQ2_XXS", "IQ3_M", "IQ3_XXS", "IQ4_XS", "Q2_K", "Q3_K_L", "Q3_K_M", "Q3_K_S",
          "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "Q8_0-MTP")
SOURCE_QUANTS = {"Q2_K", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q6_K"}


def checksums(model, cache_dir=None):
    return {s["name"]: (s["size"], "1" * 64) for s in setup.ORCA_CATALOG["variants"][model]}


class PublishedProfiles(Base):
    def test_all_quant_profiles_keep_an_explicit_split_when_ram_fits(self):
        for model in QUANTS:
            with self.subTest(model=model):
                code, out, cfg, _ = self.install(model, ["--gpus", "all"], ram=320, n_gpus=2)
                self.assertEqual(code, 0, out)
                self.assertEqual(cfg["gpu"], [0, 1])
                self.assertEqual(cfg["layer_split"], "auto")
                self.assertNotIn("--resident-budget-gib", cfg["args"])

    def test_orca_iq3_split_uses_orca_memory_bound_and_does_not_fall_back(self):
        code, out, cfg, _ = self.install("IQ3_XXS", ["--gpus", "all"], ram=104, n_gpus=2)
        self.assertEqual(code, 1, out)
        self.assertIn("~109 GB", out)
        self.assertIsNone(cfg)
        self.assertEqual(self.downloads, [])

    def test_windows_amd_source_quant_stops_with_actionable_platform_limit(self):
        card = {"index": 0, "name": "AMD Radeon RX 7900 XTX", "vram_gb": 24, "arch": "gfx1100"}
        with mock.patch.object(setup, "WIN", True), \
                mock.patch.object(setup, "get_prebuilt_hip") as prebuilt:
            code, out, cfg, built = self.install("Q2_K", ["--backend", "hip"], amd=[card])
        self.assertEqual(code, 1, out)
        self.assertIn("cannot yet build for Windows AMD", out)
        self.assertIn("Linux HIP", out)
        self.assertEqual(self.downloads, [])
        self.assertIsNone(cfg)
        prebuilt.assert_not_called()
        built.assert_not_called()

    def test_orca_local_files_cannot_impersonate_the_selected_qwen_profile(self):
        folder = self.t / "shards"
        folder.mkdir()
        for entry in setup.ORCA_CATALOG["variants"]["Q2_K"]:
            path = folder / entry["name"]
            path.touch()
            setup.mark(path)
        code, out, _ = self.main(["--family", "qwen", "--gguf-dir", str(folder)],
                                 m="IQ3_XXS", family=False, version="0.1.39")
        self.assertEqual(code, 1, out)
        self.assertIn("--family orca --model Q2_K", out)
        self.assertFalse(any(len(cmd) > 1 and cmd[1].endswith("iq_pack.py") for cmd in self.runs))
        self.assertFalse((self.t / "strata-iq3_xxs.json").exists())

    def install(self, quant="Q4_K_M", argv=(), **kwargs):
        model = quant
        self.downloads.clear()
        self.verified.clear()
        self.runs.clear()

        def build(*a, **k):
            eng = self.t / "engine"
            (eng / "BUILD.json").write_text(json.dumps({"version": "0.1.39", "source": "local",
                                                        "src": setup.source_hash(setup.ENGINE_SOURCES)}))
            return eng

        with mock.patch.object(setup, "build_engine", side_effect=build) as built, \
                mock.patch.object(setup, "orca_checksums", side_effect=checksums, create=True), \
                mock.patch.dict(setup.FAMILIES["orca"]["sha256"], checksums(model)):
            code, out, _ = self.main(["--family", "orca", "--context", "32768", *argv], m=model,
                                     family=False, version="0.1.39", **kwargs)
        path = self.t / f"strata-orca-{model.lower()}.json"
        return code, out, json.loads(path.read_text()) if path.exists() else None, built

    def test_all_published_options_keep_exact_shards_names_budgets_and_unique_configs(self):
        self.assertEqual(set(setup.family_models("orca")), set(QUANTS))
        generated = set()
        for model in QUANTS:
            with self.subTest(model=model):
                code, out, cfg, built = self.install(model)
                self.assertEqual(code, 0, out)
                names = list(checksums(model))
                self.assertEqual([url.rsplit("/", 1)[-1] for url in self.downloads], names)
                self.assertEqual(self.verified, [(name, size, sha) for name, (size, sha) in checksums(model).items()])
                self.assertEqual(cfg["model_name"], f"orcarouter-qwen3.8-flash-next-uncensored-{model.lower()}")
                self.assertTrue(cfg["tokenizer"].endswith(f"packs/orca-{model.lower()}/tokenizer"))
                self.assertEqual(cfg["args"][cfg["args"].index("--native") + 1].split("/")[-1], names[0])
                self.assertIn("--resident-budget-gib", cfg["args"])
                self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "40")
                self.assertTrue(cfg["args"][cfg["args"].index("--mtp") + 1].endswith("mtp/rt"))
                pack = next(r for r in self.runs if r[1].endswith("iq_pack.py"))
                self.assertIn("--compat-bf16", pack)
                self.assertNotIn("--experts-bin", pack)
                self.assertEqual(built.call_count, int(model in SOURCE_QUANTS))
                self.assertEqual(cfg.get("engine_requirements", []),
                                 [setup.ORCA_ENGINE_REQUIREMENT] if model in SOURCE_QUANTS else [])
                path = self.t / f"strata-orca-{model.lower()}.json"
                choices = setup.choices_from_config(path)
                self.assertEqual((choices["family"], choices["model"]), ("orca", model))
                self.assertEqual(setup.config_model_choice(cfg), ("orca", model))
                generated.add(path)
        self.assertEqual(len(generated), 16)
        self.assertTrue(all(p.exists() for p in generated))

    def test_q4_k_m_remains_default_at_small_and_large_ram(self):
        for ram in (48, 64, 256):
            with self.subTest(ram=ram):
                code, out, cfg, _ = self.install(ram=ram, model=False)
                self.assertEqual(code, 0, out)
                self.assertTrue(cfg["model_name"].endswith("-q4_k_m"))

    def test_iq3_xxs_repeated_main_calls_do_not_change_qwen_swift_profiles(self):
        original = dict(setup.MODELS["IQ3_XXS"])
        for family in ("orca", "qwen", "swift", "orca", "qwen"):
            with self.subTest(family=family):
                if family == "orca":
                    code, out, cfg, _ = self.install("IQ3_XXS", ram=96)
                else:
                    code, out, _ = self.main(["--family", family, "--context", "32768"], m="IQ3_XXS",
                                             family=False, version="0.1.39", ram=96)
                    tag = "" if family == "qwen" else "swift-"
                    cfg = json.loads((self.t / f"strata-{tag}iq3_xxs.json").read_text())
                self.assertEqual(code, 0, out)
                self.assertEqual("--resident-budget-gib" in cfg["args"], family == "orca")
                self.assertEqual(setup.config_model_choice(cfg), (family, "IQ3_XXS"))
                self.assertEqual(setup.MODELS["IQ3_XXS"], original)
        self.assertEqual(original["arena_gb"], 42.9)
        self.assertEqual(setup.model_profile("IQ3_XXS", "orca")["arena_gb"], 85.202668032)
        self.assertEqual(setup.resident_budget_gib("IQ3_XXS", 256, family="orca"), 79)
        self.assertEqual(setup.resident_budget_gib("IQ3_XXS", 256, family="qwen"), 39)
        self.assertIsNone(setup.ctx_ram_need("IQ3_XXS", 262144, family="orca"))
        self.assertIsNotNone(setup.ctx_ram_need("IQ3_XXS", 262144, family="qwen"))

    def test_exact_local_shard_lists_for_all_variants_and_five_shard_files(self):
        folder = self.t / "local"
        folder.mkdir()
        (folder / "other-Q4_K_M-00001-of-00002.gguf").touch()
        for model in QUANTS:
            with self.subTest(model=model):
                expected = list(checksums(model))
                paths = setup.gguf_dir_shards(folder, setup.FAMILIES["orca"], model)
                self.assertEqual([p.name for p in paths], expected)
                for path in paths:
                    path.touch()
                    setup.mark(path)
                code, out, cfg, _ = self.install(model, ["--gguf-dir", str(folder)])
                self.assertEqual(code, 0, out)
                self.assertEqual(self.downloads, [])
                self.assertEqual([name for name, _, _ in self.verified], expected)
                self.assertEqual(setup.model_shards(setup.FAMILIES["orca"], model), len(expected))
        for model in ("Q6_K", "Q8_0", "Q8_0-MTP"):
            self.assertEqual(setup.model_shards(setup.FAMILIES["orca"], model), 5)

    def test_check_uses_orca_profiles_including_iq3_xxs_collision(self):
        code, out, cfg, _ = self.install("IQ3_XXS", ["--check"])
        self.assertEqual(code, 0, out)
        self.assertIsNone(cfg)
        self.assertIn("IQ3_XXS  48 GB RAM planning threshold (unmeasured)", out)
        self.assertIn("40 GiB", out)
        self.assertNotIn("IQ3_XXS  needs ~60", out)

    def test_other_quant_shards_are_not_substituted(self):
        folder = self.t / "wrong"
        folder.mkdir()
        (folder / "model-Q4_K_M-00001-of-00002.gguf").touch()
        code, out, cfg, _ = self.install("Q4_K_M", ["--gguf-dir", str(folder)])
        self.assertEqual(code, 1, out)
        self.assertIsNone(cfg)
        self.assertIn("has no", out)
        self.assertEqual(self.downloads, [])
        for model in QUANTS:
            if model != "IQ3_XXS":  # the original family still permits its historical renamed-shard workflow
                self.assertEqual(setup.gguf_unsupported(f"other-{model}-00001-of-00002.gguf"), model)

    def test_conservative_sizes_and_experimental_notes_for_every_variant(self):
        for model in QUANTS:
            with self.subTest(model=model):
                profile = setup.model_profile(model, "orca")
                total = sum(size for size, _ in checksums(model).values()) / 1e9
                self.assertEqual(profile["arena_gb"], total)
                self.assertEqual(profile["download_gb"], total)
                self.assertEqual(profile["ram_gb"], 48)
                self.assertTrue(profile["budget"])
                self.assertTrue(profile["experimental"])
                self.assertIn("not a measured minimum", profile["experimental_note"])
                self.assertIn("original Qwen draft", profile["experimental_note"])
                self.assertEqual(setup.resident_budget_gib(model, 64, family="orca"), 40)
                self.assertEqual(setup.resident_budget_gib(model, 64, 2.1, "orca"), 37)


class ResumeAndEngine(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.engine = self.root / "engine"
        self.engine.mkdir()
        self.root_patch = mock.patch.object(setup, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def cfg(self, model):
        name = setup.model_file(setup.FAMILIES["orca"], model, 1)
        return {"exe": str(self.engine / setup.EXE), "args": ["--native", "C:\\models\\" + name,
                                                               "--resident-budget-gib", "40"]}

    def stamp(self, source="local", src=None):
        (self.engine / "BUILD.json").write_text(json.dumps({"source": source, "version": "0.1.39",
                                                           "src": src or setup.source_hash(setup.ENGINE_SOURCES)}))

    def test_budget_inference_retains_family_and_mtp_variant(self):
        for model in QUANTS:
            cfg = self.cfg(model)
            self.assertEqual(setup.budget_model(cfg), model)
            self.assertEqual(setup.budget_family(cfg), "orca")
            self.assertEqual(setup.unsloth_split_need_gb(model, setup.budget_family(cfg)),
                             setup.ORCA_MODELS[model]["download_gb"] + 24)
        self.assertNotEqual(setup.unsloth_split_need_gb("Q8_0"), setup.unsloth_split_need_gb("Q8_0-MTP"))
        with mock.patch.object(setup, "ram_gb", return_value=104):
            code, out = quiet(setup.split_budget, self.cfg("IQ3_XXS"))
        self.assertEqual(code, 1, out)
        self.assertIn("109 GB", out)

    def test_source_requirements_inferred_from_old_configs_and_prebuilt_rejected(self):
        for model in QUANTS:
            cfg = self.cfg(model)
            self.assertEqual(bool(setup.model_engine_requirements(cfg)), model in SOURCE_QUANTS)
        for source, src in (("prebuilt", None), ("local", "old-source")):
            self.stamp(source, src)
            for model in SOURCE_QUANTS:
                code, out = quiet(setup.require_model_engine, self.cfg(model))
                self.assertEqual(code, 1, (model, source, out))
                self.assertIn("--setup --build", out)
        self.stamp()
        for model in SOURCE_QUANTS:
            self.assertIsNone(setup.require_model_engine(self.cfg(model)))

    def test_start_and_update_refuse_an_incompatible_prebuilt(self):
        self.stamp("prebuilt")
        cfg = self.cfg("Q2_K")
        path = self.root / "strata-orca-q2_k.json"
        path.write_text(json.dumps(cfg))
        with mock.patch.object(setup, "upgrade_config", side_effect=lambda p, c: c), \
                mock.patch.object(setup, "require_verified_engine"), \
                mock.patch.object(setup, "pip_install"), \
                mock.patch.object(setup, "update_installed_engine"):
            code, out = quiet(setup.start, path, None)
            self.assertEqual(code, 1, out)
            code, out = quiet(setup.update_install, [path], SimpleNamespace(build=False, prebuilt="test"))
            self.assertEqual(code, 1, out)
            self.assertNotIn("up to date", out)

    def test_hardware_switch_requires_a_local_cuda12_build(self):
        cfg = self.cfg("Q6_K")
        self.stamp()
        path = self.root / "strata-orca-q6_k.json"
        card = {"index": 0, "arch": "61", "name": "Pascal"}
        with mock.patch.object(setup, "gpu_info", return_value=card), \
                mock.patch.object(setup, "get_cuda12_engine", return_value=self.engine) as engine, \
                mock.patch.object(setup, "engine_lib_dirs", return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()):
            result = setup.use_cuda12([card], path, cfg, True)
        self.assertTrue(engine.call_args.kwargs["build"])
        self.assertEqual(result["cuda"], 12)
        self.assertEqual(setup.model_engine_requirements(result), {setup.ORCA_ENGINE_REQUIREMENT})

    def test_generated_scripts_preflight_before_starting_the_server(self):
        cfg = self.cfg("Q2_K")
        cfg["engine_requirements"] = [setup.ORCA_ENGINE_REQUIREMENT]
        path = self.root / "strata-orca-q2_k.json"
        path.write_text(json.dumps(cfg))
        (self.root / "setup.py").write_text(Path(setup.__file__).read_text())
        (self.root / "orca-quants.json").write_text(json.dumps(setup.ORCA_CATALOG))
        server = self.root / "serve"
        server.mkdir()
        (server / "gpu_devices.py").write_text(
            (Path(setup.__file__).parent / "serve" / "gpu_devices.py").read_text())
        (server / "server.py").write_text("from pathlib import Path; Path(__file__).with_name('started').touch()")
        with mock.patch.object(setup, "WIN", False):
            script = setup.write_run_script("orca-Q2_K", path, 8080, False)
        self.stamp("prebuilt")
        refused = subprocess.run(["sh", str(script)], capture_output=True, text=True)
        self.assertEqual(refused.returncode, 1, refused.stdout + refused.stderr)
        self.assertIn("--setup --build", refused.stdout)
        self.assertFalse((server / "started").exists())
        self.stamp()
        accepted = subprocess.run(["sh", str(script)], capture_output=True, text=True)
        self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
        self.assertTrue((server / "started").exists())
        with mock.patch.object(setup, "WIN", True):
            windows = setup.write_run_script("orca-Q2_K", path, 8080, False).read_text()
        self.assertLess(windows.index("require_model_engine"), windows.index("server.py"))
        self.assertIn("exit /b 1", windows)

    def test_saved_requirement_survives_custom_config_names(self):
        cfg = {"exe": str(self.engine / setup.EXE), "args": [],
               "engine_requirements": [setup.ORCA_ENGINE_REQUIREMENT]}
        self.stamp("prebuilt")
        code, out = quiet(setup.require_model_engine, cfg)
        self.assertEqual(code, 1, out)


if __name__ == "__main__":
    unittest.main()
