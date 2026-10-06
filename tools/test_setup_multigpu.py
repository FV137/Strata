"""GPU discovery, explicit split choices and CUDA child environments, without GPU hardware."""
import contextlib
import ctypes
import io
import json
import os
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup
from serve.server import child_env, engine_args, vision_env
from serve import gpu_devices


# NVIDIA management indices may be sparse inside a container. UUIDs identify the
# same devices even when CUDA enumerates this pair as devices 0 and 1.
SMI = ("2, GPU-22222222-aaaa-bbbb-cccc-111111111111, NVIDIA RTX 3090, 24576, 8.6, 580.97\n"
       "5, GPU-55555555-aaaa-bbbb-cccc-111111111111, NVIDIA RTX 4090, 24576, 8.9, 580.97\n")


class Visibility(unittest.TestCase):
    def test_no_nvidia_inventory_does_not_probe_cuda_before_amd_selection(self):
        with mock.patch.object(setup, "nvidia_gpus", return_value=[]), \
                mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                mock.patch.object(gpu_devices, "cuda_mask_uuids", side_effect=AssertionError("CUDA queried")):
            self.assertEqual(setup.gpus(), [])

    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        def query(cmd, **kwargs):
            if cmd[0] == sys.executable:
                # CUDA's FASTEST_FIRST order is deliberately different from
                # nvidia-smi's management indices (5,2 instead of 2,5).
                devices = [SMI.splitlines()[i].split(", ")[1] for i in (1, 0)]
                if kwargs["env"].get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID":
                    devices.reverse()
                selected = []
                for token in kwargs["env"]["CUDA_VISIBLE_DEVICES"].split(","):
                    if not token.isdigit() or int(token) >= len(devices):
                        break
                    selected.append(devices[int(token)])
                return SimpleNamespace(stdout=json.dumps(selected), returncode=0)
            self.assertEqual(cmd[0], "nvidia-smi")
            output = SMI if "uuid" in cmd[1] else "\n".join(
                ", ".join(line.split(", ")[:1] + line.split(", ")[2:]) for line in SMI.splitlines())
            return SimpleNamespace(stdout=output, returncode=0)
        self.smi = mock.patch("subprocess.run", side_effect=query)
        self.smi.start()
        self.addCleanup(self.smi.stop)

    def test_all_discovers_sparse_container_devices(self):
        self.assertEqual(setup.parse_gpus("all", setup.gpus()), [5, 2])

    def test_cuda_mask_filters_discovery_but_keeps_management_ids(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-22222222"}):
            self.assertEqual([g["index"] for g in setup.gpus()], [2])

    def test_empty_mask_means_no_gpus(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            self.assertEqual(setup.gpus(), [])

    def test_integer_mask_stops_at_invalid_identifier(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,-1,0"}):
            self.assertEqual([g["index"] for g in setup.gpus()], [2])

    def test_integer_mask_uses_cuda_order_instead_of_management_ids(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            self.assertEqual([g["index"] for g in setup.gpus()], [5])
            self.assertEqual(child_env({"gpu": 5})["CUDA_VISIBLE_DEVICES"],
                             "GPU-55555555-aaaa-bbbb-cccc-111111111111")

    def test_numeric_mask_with_pci_order_is_resolved_by_cuda(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}):
            self.assertEqual([g["index"] for g in setup.gpus()], [2])

    def test_numeric_mask_never_guesses_if_cuda_probe_fails(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                mock.patch("subprocess.run", return_value=SimpleNamespace(stdout=SMI, returncode=0)):
            with self.assertRaisesRegex(ValueError, "UUID"):
                child_env({"gpu": 2})

    def test_mig_mask_is_not_replaced_with_a_whole_gpu(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "MIG-GPU-22222222/1/0"}):
            with self.assertRaisesRegex(ValueError, "MIG"):
                child_env({"gpu": 2})

    def test_all_never_adds_a_hidden_card(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-22222222"}), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            setup.parse_gpus("all", setup.gpus())

    def test_explicit_hidden_gpu_stops_before_launch(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-55555555"}):
            with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                child_env({"gpu": [5, 2]})

    def test_sparse_ids_become_ordered_uuids_for_cuda(self):
        cfg = {"gpu": [5, 2], "args": []}
        env = child_env(cfg)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"],
                         "GPU-55555555-aaaa-bbbb-cccc-111111111111,GPU-22222222-aaaa-bbbb-cccc-111111111111")
        self.assertEqual(engine_args(cfg), ["--layer-split", "auto"])

    def test_saved_gpu_missing_from_container_stops(self):
        with self.assertRaisesRegex(ValueError, "GPU 0"):
            child_env({"gpu": [0, 2]})

    def test_config_env_cannot_silently_reduce_the_selected_split(self):
        with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
            child_env({"gpu": [5, 2], "env": {"CUDA_VISIBLE_DEVICES": "GPU-55555555"}})

    def test_config_env_cannot_widen_the_launcher_mask(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-22222222"}):
            with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                child_env({"gpu": [5, 2], "env": {"CUDA_VISIBLE_DEVICES": "GPU-55555555,GPU-22222222"}})

    def test_spare_vision_gpu_uses_parent_visibility(self):
        cfg = {"gpu": 5, "vision": {"cuda_device": 2}}
        env = child_env(cfg)
        image_env = vision_env(cfg, env)
        self.assertEqual(image_env["CUDA_VISIBLE_DEVICES"], "GPU-22222222-aaaa-bbbb-cccc-111111111111")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-55555555-aaaa-bbbb-cccc-111111111111")


class ExplicitSplit(unittest.TestCase):
    def test_conflicting_single_and_multiple_gpu_flags_are_refused(self):
        # Previously setup silently chose --gpus and discarded --gpu.
        with mock.patch.object(sys, "argv", ["setup.py", "--gpu", "0", "--gpus", "1,2"]), \
                mock.patch.object(setup, "data_folder", side_effect=AssertionError("must validate flags first")), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            setup.main()
        self.assertEqual(stopped.exception.code, 2)

    def test_all_does_not_fall_back_when_budget_model_cannot_split(self):
        cards = [{"index": i, "name": "RTX 3090", "arch": "86", "vram_gb": 24} for i in (0, 1)]
        args = SimpleNamespace(gpus="all", resident_budget_gib=None, yes=True)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            setup.unsloth_together(args, "Q4_K_M", 64, cards[0], cards)


class DriverProbe(unittest.TestCase):
    def test_cuda_handles_are_resolved_to_uuids_without_creating_contexts(self):
        # ctypes callbacks exercise pointer/byte marshalling at the actual CUDA
        # API boundary. The fake driver exposes no context or allocation API.
        result_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_int))
        get_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int)
        uuid_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
        ids = ["55555555-aaaa-bbbb-cccc-111111111111", "22222222-aaaa-bbbb-cccc-111111111111"]
        def count(out):
            out[0] = 2
            return 0
        def device(out, ordinal):
            out[0] = ordinal + 10
            return 0
        def identity(out, handle):
            ctypes.memmove(out, uuid.UUID(ids[handle - 10]).bytes, 16)
            return 0
        driver = SimpleNamespace(cuInit=lambda flags: 0, cuDeviceGetCount=result_t(count),
                                 cuDeviceGet=get_t(device), cuDeviceGetUuid_v2=uuid_t(identity))
        with mock.patch.object(gpu_devices.os, "name", "posix"), \
                mock.patch.object(gpu_devices.ctypes, "CDLL", return_value=driver):
            self.assertEqual(gpu_devices.cuda_driver_uuids(), ["GPU-" + value for value in ids])

    def test_driver_reports_no_devices(self):
        with mock.patch.object(gpu_devices.os, "name", "posix"), \
                mock.patch.object(gpu_devices.ctypes, "CDLL", return_value=SimpleNamespace(cuInit=lambda _: 100)):
            self.assertEqual(gpu_devices.cuda_driver_uuids(), [])


if __name__ == "__main__":
    unittest.main()
