"""Exercise the actual entry point with a tiny setup substitute: no Docker, GPU, downloads or secrets."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(os.name == "nt", "the container entry point runs on Linux")
class ContainerAccess(unittest.TestCase):
    def run_entry(self, extra=None, saved=None):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / "data"
            (data / "config").mkdir(parents=True)
            if saved is not None:
                (data / "config/strata-iq2_xs.json").write_text(json.dumps(saved))
            (root / ".venv/bin").mkdir(parents=True)
            (root / "setup.py").write_text('''import importlib.util, json, os, pathlib, sys
root = pathlib.Path(__file__).parent
config_path = root / "strata-iq2_xs.json"
saved_path = pathlib.Path(os.environ["STRATA_DATA"]) / "config/strata-iq2_xs.json"
with (root / "calls.jsonl").open("a") as f:
    f.write(json.dumps({"args": sys.argv[1:], "key": os.environ.get("STRATA_API_KEY"),
                       "config": json.loads(config_path.read_text()) if config_path.exists() else {},
                       "saved": json.loads(saved_path.read_text()) if saved_path.exists() else {}}) + "\\n")
if "--setup" in sys.argv:
    spec = importlib.util.spec_from_file_location("real_setup", REPO_SETUP_PATH)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    setup.write_setup_config(config_path, {}, source=saved_path)
'''.replace("REPO_SETUP_PATH", repr(str(ROOT / "setup.py"))))
            python = root / ".venv/bin/python"
            python.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} \"$@\"\n")
            python.chmod(0o755)
            entry = root / "entry.sh"
            entry.write_text((ROOT / "docker-entrypoint.sh").read_text().replace("/opt/strata", str(root)))
            names = {"API_KEY", "STRATA_API_KEY", "HOST", "PORT", "FAMILY", "MODEL", "REINSTALL", "GPUS", "GPU",
                     "KV", "LOW_RAM", "LAYER_SPLIT", "CONTEXT", "VISION"}
            env = {k: v for k, v in os.environ.items() if k not in names}
            env.update(STRATA_DATA=str(data), PYTHONPATH=str(ROOT))
            run = subprocess.run(["sh", str(entry)], env={**env, **(extra or {})}, capture_output=True,
                                 text=True, timeout=5)
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()] \
                if (root / "calls.jsonl").exists() else []
            return run, calls

    def test_new_unkeyed_container_stops_before_setup(self):
        run, calls = self.run_entry()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("API key", run.stderr)
        self.assertEqual(calls, [])

    def test_reused_unkeyed_volume_also_stops(self):
        run, calls = self.run_entry(saved={"host": "0.0.0.0"})
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(calls, [])

    def test_key_rotation_reaches_server_without_reinstall(self):
        run, calls = self.run_entry({"API_KEY": "new-test-key"}, saved={"api_key": "old-test-key"})
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["key"], "new-test-key")
        self.assertEqual(calls[0]["args"], ["--host", "0.0.0.0", "--port", "8080"])
        self.assertNotIn("new-test-key", run.stdout + run.stderr)

    def test_new_key_stays_out_of_setup_arguments(self):
        run, calls = self.run_entry({"STRATA_API_KEY": "new-test-key"})
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual(call["key"], "new-test-key")
            self.assertNotIn("--api-key", call["args"])
            self.assertNotIn("new-test-key", call["args"])

    def test_saved_key_can_be_reused(self):
        run, calls = self.run_entry(saved={"api_key": "saved-test-key"})
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(calls), 1)

    def test_reinstall_preserves_saved_key_in_both_configs(self):
        run, calls = self.run_entry({"REINSTALL": "1"}, saved={"api_key": "saved-test-key"})
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1]["config"].get("api_key"), "saved-test-key")
        self.assertEqual(calls[-1]["saved"].get("api_key"), "saved-test-key")
        for call in calls:
            self.assertNotIn("saved-test-key", call["args"])
        self.assertNotIn("saved-test-key", run.stdout + run.stderr)

    def test_conflicting_or_empty_environment_keys_stop(self):
        for env in ({"API_KEY": ""}, {"STRATA_API_KEY": ""},
                    {"API_KEY": "one-test-key", "STRATA_API_KEY": "different-test-key"}):
            with self.subTest(env=env):
                run, calls = self.run_entry(env, saved={"api_key": "saved-test-key"})
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
