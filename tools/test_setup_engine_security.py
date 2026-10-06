"""Engine/source supply-chain regressions. Synthetic archives only; no GPU or network."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup


class EngineSecurity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dist = self.root / 'dist'
        self.dist.mkdir()
        (self.root / 'CMakeLists.txt').write_text('project(strata VERSION 0.1.39)')
        self.asset = 'strata-windows-x64.zip'
        self.archive = self.dist / self.asset
        self.manifest = self.root / 'trusted.json'
        self.gpu = {'arch': '89', 'name': 'test GPU'}
        self.meta = {'source': 'prebuilt', 'version': '0.1.39', 'archs': [89], 'ptx': False,
                     'cuda': '13.0', 'vision': 'gpu'}
        for patch in (mock.patch.object(setup, 'ROOT', self.root),
                      mock.patch.object(setup, 'WIN', True),
                      mock.patch.object(setup, 'EXE', 'strata.exe'),
                      mock.patch.object(setup, 'VEXE', 'strata-vision.exe'),
                      mock.patch.object(setup, 'PREBUILT_ASSET', self.asset),
                      mock.patch.dict(os.environ, {'STRATA_ARTIFACT_MANIFEST': str(self.manifest)}),
                      mock.patch.object(setup.urllib.request, 'urlopen', side_effect=OSError('network disabled')),
                      contextlib.redirect_stdout(io.StringIO())):
            patch.__enter__()
            self.addCleanup(patch.__exit__, None, None, None)

    def make_archive(self, *, meta=None, extra=(), executable=True, vision=True):
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr('BUILD.json', json.dumps(self.meta if meta is None else meta))
            if executable:
                z.writestr('strata.exe', b'trusted engine')
            if vision:
                z.writestr('strata-vision.exe', b'trusted encoder')
            for name, data in extra:
                z.writestr(name, data)
        self.trust()

    def trust(self):
        self.manifest.write_text(json.dumps({'artifacts': {self.asset: {
            'size': self.archive.stat().st_size,
            'sha256': hashlib.sha256(self.archive.read_bytes()).hexdigest(), 'version': '0.1.39'}}}))

    def install(self, **kw):
        return setup.get_prebuilt(str(self.dist), self.gpu, 'gpu', **kw)

    def old_engine(self):
        eng = self.root / 'engine'
        eng.mkdir(exist_ok=True)
        (eng / 'strata.exe').write_bytes(b'previous engine')
        (eng / 'BUILD.json').write_text(json.dumps({**self.meta, 'version': '0.1.38'}))
        return eng

    def test_default_has_no_latest_fallback(self):
        self.assertEqual(setup.prebuilt_bases(setup.PREBUILT_URL),
                         ['https://github.com/Niko1221/Strata/releases/download/v0.1.39/'])

    def test_custom_archive_requires_explicit_trusted_manifest(self):
        self.make_archive()
        self.manifest.unlink()
        self.assertIsNone(self.install())
        self.assertFalse((self.root / 'engine' / 'strata.exe').exists())

    def test_appended_tampering_is_rejected_before_activation(self):
        self.make_archive()
        with self.archive.open('ab') as f:
            f.write(b'untrusted bytes')
        self.assertIsNone(self.install())
        self.assertFalse((self.root / 'engine' / 'strata.exe').exists())

    def test_wrong_same_size_digest_is_rejected(self):
        self.make_archive()
        data = json.loads(self.manifest.read_text())
        data['artifacts'][self.asset]['sha256'] = '0' * 64
        self.manifest.write_text(json.dumps(data))
        self.assertIsNone(self.install())

    def test_cached_done_marker_cannot_authorize_modified_archive(self):
        self.make_archive()
        eng = self.old_engine()
        cached = eng / self.asset
        cached.write_bytes(self.archive.read_bytes() + b'forged')
        setup.mark(cached)
        # A second untrusted copy at the origin must not rescue the forged cache.
        self.archive.write_bytes(cached.read_bytes())
        self.assertIsNone(self.install(updating=True))
        self.assertEqual((eng / 'strata.exe').read_bytes(), b'previous engine')

    def test_path_traversal_rejected_before_any_extraction(self):
        self.make_archive(extra=[('../outside.txt', b'escape')])
        self.assertIsNone(self.install())
        self.assertFalse((self.root / 'outside.txt').exists())
        self.assertFalse((self.root / 'engine' / 'outside.txt').exists())

    def test_windows_absolute_and_ambiguous_paths_are_rejected(self):
        for name in ('C:/escape.dll', '/escape.dll', r'..\escape.dll', 'file:stream',
                     'CON.dll', 'strata.exe.', 'STRATA.EXE'):
            with self.subTest(name=name):
                self.make_archive(extra=[(name, b'escape')])
                self.assertIsNone(self.install())

    def test_symlink_entry_rejected(self):
        link = zipfile.ZipInfo('outside-link')
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.make_archive(extra=[(link, b'../outside')])
        self.assertIsNone(self.install())

    def test_missing_binary_never_replaces_previous_engine(self):
        old = self.old_engine()
        self.make_archive(executable=False)
        self.assertIsNone(self.install(updating=True))
        self.assertEqual((old / 'strata.exe').read_bytes(), b'previous engine')
        self.assertEqual(json.loads((old / 'BUILD.json').read_text())['version'], '0.1.38')

    def test_missing_requested_encoder_is_rejected(self):
        self.make_archive(vision=False)
        self.assertIsNone(self.install())

    def test_malformed_metadata_is_rejected_without_exception(self):
        for meta in ({}, [], {'version': 'garbage'}, {**self.meta, 'archs': []},
                     {**self.meta, 'archs': ['bad']}, {**self.meta, 'source': 'local'},
                     {**self.meta, 'lib_dirs': ['../outside']}, {**self.meta, 'cuda': '12.9'}):
            with self.subTest(meta=meta):
                self.make_archive(meta=meta)
                try:
                    installed = self.install()
                except (ValueError, TypeError, AttributeError, FileNotFoundError) as error:
                    installed = error
                self.assertIsNone(installed)

    def test_unsupported_architecture_is_rejected(self):
        self.make_archive(meta={**self.meta, 'archs': [86]})
        self.assertIsNone(self.install())

    def test_self_reported_installed_version_is_not_integrity_evidence(self):
        eng = self.old_engine()
        (eng / 'BUILD.json').write_text(json.dumps({**self.meta, 'version': '99.99.99'}))
        self.assertIsNone(setup.get_prebuilt('', self.gpu, 'gpu'))
        self.assertEqual((eng / 'strata.exe').read_bytes(), b'previous engine')

    def test_installed_bytes_rechecked_against_archive(self):
        self.make_archive()
        eng = self.install()
        self.assertIsNotNone(eng)
        (eng / 'strata.exe').write_bytes(b'tampered installed engine')
        self.assertIsNone(setup.get_prebuilt('', self.gpu, 'gpu'))

    def test_good_archive_installs_and_is_reused_offline(self):
        self.make_archive()
        eng = self.install()
        self.assertEqual((eng / 'strata.exe').read_bytes(), b'trusted engine')
        self.archive.unlink()
        self.assertEqual(setup.get_prebuilt('', self.gpu, 'gpu'), eng)

    def test_cached_verified_archive_is_rehashed_even_with_done_marker(self):
        self.make_archive()
        eng = self.install()
        cached = next((self.root / '.cache').rglob(self.asset))
        cached.write_bytes(cached.read_bytes() + b'forged')
        setup.mark(cached)
        self.assertIsNone(setup.get_prebuilt('', self.gpu, 'gpu'))
        self.assertEqual((eng / 'strata.exe').read_bytes(), b'trusted engine')

    def test_verified_previous_engine_survives_failed_update(self):
        self.make_archive()
        eng = self.install()
        self.archive.unlink()
        # The new trusted release entry is unavailable, while the old receipt still has its trusted bytes.
        with mock.patch.object(setup, 'MIN_ENGINE', (0, 1, 40)), \
                mock.patch.object(setup, 'gpu_info', return_value=self.gpu):
            setup.update_installed_engine(str(self.dist), 13)
        self.assertEqual((eng / 'strata.exe').read_bytes(), b'trusted engine')
        self.assertTrue(setup.installed_prebuilt_verified(eng))

    def test_extra_dll_in_installed_folder_is_not_trusted(self):
        self.make_archive()
        eng = self.install()
        (eng / 'injected.dll').write_bytes(b'injected')
        self.assertIsNone(setup.get_prebuilt('', self.gpu, 'gpu'))

    def test_failed_activation_restores_previous_directory(self):
        self.make_archive()
        eng = self.old_engine()
        original = Path.replace
        def fail_activation(path, destination):
            if '_engine-' in str(path) and Path(destination) == eng:
                raise PermissionError('locked on Windows')
            return original(path, destination)
        with mock.patch.object(Path, 'replace', fail_activation):
            self.assertIsNone(self.install(updating=True))
        self.assertEqual((eng / 'strata.exe').read_bytes(), b'previous engine')
        self.assertFalse(eng.with_name('engine.previous').exists())

    def test_legacy_install_cannot_start_offline(self):
        eng = self.old_engine()
        (eng / 'BUILD.json').write_text(json.dumps({**self.meta, 'version': '99.99.99'}))
        with mock.patch.object(setup, 'gpu_info', return_value=self.gpu):
            with self.assertRaises(SystemExit):
                setup.update_installed_engine('', 13)
        self.assertEqual((eng / 'strata.exe').read_bytes(), b'previous engine')

    def test_unverified_hip_probe_is_not_executed(self):
        eng = self.old_engine()
        (eng / 'BUILD.json').write_text(json.dumps({**self.meta, 'backend': 'hip'}))
        (eng / 'strata-device.exe').write_bytes(b'untrusted probe')
        with mock.patch.object(setup.subprocess, 'run', side_effect=AssertionError('untrusted probe ran')):
            self.assertIsNone(setup.hip_devices(eng / 'strata-device.exe'))

    def test_hip_archive_and_runtime_copy_are_verified(self):
        self.asset = setup.WIN_HIP_ASSET
        self.archive = self.dist / self.asset
        self.make_archive(meta={**self.meta, 'backend': 'hip', 'archs': ['gfx1100'], 'lib_dirs': ['rocm/bin']},
                          extra=[('strata-device.exe', b'probe'), ('rocm/bin/amdhip64_7.dll', b'runtime')])
        gpu = {'arch': 'gfx1100'}
        eng = setup.get_prebuilt_hip(str(self.dist), gpu)
        self.assertIsNotNone(eng)
        self.assertEqual((eng / 'amdhip64_7.dll').read_bytes(), b'runtime')
        self.assertEqual(setup.get_prebuilt_hip('', gpu), eng)
        (eng / 'amdhip64_7.dll').write_bytes(b'tamper!')
        self.assertIsNone(setup.get_prebuilt_hip('', gpu))

    def test_declared_zip_bomb_is_rejected_before_extraction(self):
        import struct
        self.make_archive()
        data = bytearray(self.archive.read_bytes())
        index = data.index(b'PK\x01\x02')
        struct.pack_into('<I', data, index + 24, 0xffffffff)
        self.archive.write_bytes(data)
        self.trust()
        self.assertIsNone(self.install())
        self.assertFalse((self.root / 'engine').exists())

    def test_reserved_receipt_in_archive_is_rejected(self):
        self.make_archive(extra=[('.artifact.json', b'{}')])
        self.assertIsNone(self.install())

    def test_source_tree_is_checked_before_reuse(self):
        asset = f'llama.cpp-{setup.LLAMA_CPP_COMMIT}.zip'
        archive = self.root / asset
        prefix = 'llama.cpp-' + setup.LLAMA_CPP_COMMIT + '/'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(prefix + 'ggml/CMakeLists.txt', 'trusted build script')
            z.writestr(prefix + 'gguf-py/__init__.py', '')
        manifest = self.root / 'source-trusted.json'
        manifest.write_text(json.dumps({'artifacts': {asset: {
            'commit': setup.LLAMA_CPP_COMMIT, 'size': archive.stat().st_size,
            'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}}}))
        with mock.patch.object(setup, 'ARTIFACT_MANIFEST', manifest), \
                mock.patch.object(setup, 'LLAMA_CPP_ZIP', str(archive)):
            tree = setup.get_llama_cpp()
            (tree / 'ggml/CMakeLists.txt').write_text('tampered build script')
            archive.unlink()
            self.assertEqual(setup.get_llama_cpp(), tree)
            self.assertEqual((tree / 'ggml/CMakeLists.txt').read_text(), 'trusted build script')

    def test_unchecked_hash_bytecode_cannot_override_verified_source(self):
        import importlib.util
        import py_compile
        import subprocess

        asset = f'llama.cpp-{setup.LLAMA_CPP_COMMIT}.zip'
        archive = self.root / asset
        prefix = 'llama.cpp-' + setup.LLAMA_CPP_COMMIT + '/'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(prefix + 'ggml/CMakeLists.txt', 'trusted build script')
            z.writestr(prefix + 'gguf-py/gguf/__init__.py', "ORIGIN = 'trusted'\n")
        manifest = self.root / 'source-trusted.json'
        manifest.write_text(json.dumps({'artifacts': {asset: {
            'commit': setup.LLAMA_CPP_COMMIT, 'size': archive.stat().st_size,
            'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}}}))
        marker = self.root / 'attacker-executed'
        attacker = self.root / 'attacker.py'
        attacker.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nORIGIN = 'attacker'\n")
        with mock.patch.object(setup, 'ARTIFACT_MANIFEST', manifest), \
                mock.patch.object(setup, 'LLAMA_CPP_ZIP', str(archive)):
            tree = setup.get_llama_cpp()
            module = tree / 'gguf-py/gguf/__init__.py'
            py_compile.compile(str(attacker), cfile=importlib.util.cache_from_source(str(module)),
                               invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
            archive.unlink()
            verified = setup.get_llama_cpp()
            tools = Path(__file__).resolve().parent
            result = subprocess.run([sys.executable, '-c',
                                     f"import sys; sys.path.insert(0, {str(tools)!r}); "
                                     "from _paths import add_gguf_py; add_gguf_py(); import gguf; print(gguf.ORIGIN)"],
                                    env={**os.environ, 'STRATA_GGUF_PY': str(verified / 'gguf-py')},
                                    check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), 'trusted')
        self.assertFalse(marker.exists(), 'unverified Python bytecode was executed')

    def test_locally_compiled_hip_engine_passes_start_integrity_guard(self):
        eng = self.old_engine()
        (eng / 'BUILD.json').write_text(json.dumps({**self.meta, 'source': 'local-hip', 'backend': 'hip'}))
        with mock.patch.object(setup, 'WIN', False):
            try:
                setup.require_verified_engine(eng / 'strata.exe')
                accepted = True
            except SystemExit:
                accepted = False
        self.assertTrue(accepted, 'locally compiled HIP engine was refused as an unverified download')

    @unittest.skipIf(os.name == 'nt', 'POSIX executable probe fixture')
    def test_locally_compiled_hip_probe_can_execute(self):
        eng = self.old_engine()
        (eng / 'BUILD.json').write_text(json.dumps({**self.meta, 'source': 'local-hip', 'backend': 'hip'}))
        probe = eng / 'strata-device'
        probe.write_text(f"#!{sys.executable}\nprint('device 0: Test AMD GPU\\narch gfx1100, 24 GiB')\n")
        probe.chmod(0o755)
        with mock.patch.object(setup, 'WIN', False):
            devices = setup.hip_devices(probe)
        self.assertIsNotNone(devices)
        self.assertEqual([(d['arch'], d['vram_gb']) for d in devices], [('gfx1100', 24.0)])

    def test_compile_fallback_replaces_unverified_encoder_and_runtime(self):
        eng = self.old_engine()
        (eng / setup.VEXE).write_bytes(b'unverified encoder')
        (eng / 'unverified.dll').write_bytes(b'unverified runtime')
        (eng / 'rocm/bin').mkdir(parents=True)
        (eng / 'rocm/bin/stale.dll').write_bytes(b'unverified nested runtime')
        self.assertIsNone(self.install())  # No archive: fallback must not relabel old files as local.
        built = []

        def compile_fixture(src, bdir, target, defs, vcvars, bat):
            built.append(target)
            dest = bdir / setup.EXE if target == 'strata' else bdir / 'bin' / setup.VEXE
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b'new local ' + target.encode())

        with mock.patch.object(setup, 'install_build_tools', return_value=('/usr/bin/nvcc', None)), \
                mock.patch.object(setup, 'cmake_build', side_effect=compile_fixture), \
                mock.patch.object(setup, 'cpu_floor', return_value=''), \
                mock.patch.object(setup, 'cpu_info', return_value=('test', set())):
            result = setup.build_engine(self.gpu, 'gpu', False, self.root)
        self.assertEqual(result, eng)
        self.assertEqual(built, ['strata', 'strata-vision'])
        self.assertEqual({p.name for p in eng.iterdir()}, {setup.EXE, setup.VEXE, 'BUILD.json'})
        self.assertEqual((eng / setup.VEXE).read_bytes(), b'new local strata-vision')
        setup.require_verified_engine(eng / setup.EXE)

    def test_failed_compile_fallback_keeps_previous_tree_untrusted(self):
        eng = self.old_engine()
        (eng / setup.VEXE).write_bytes(b'previous encoder')
        (eng / 'old.dll').write_bytes(b'previous runtime')
        original = {p.name: p.read_bytes() for p in eng.iterdir()}

        def compile_fixture(src, bdir, target, defs, vcvars, bat):
            if target == 'strata-vision':
                raise RuntimeError('compiler failed')
            bdir.mkdir(parents=True, exist_ok=True)
            (bdir / setup.EXE).write_bytes(b'new local engine')

        with mock.patch.object(setup, 'install_build_tools', return_value=('/usr/bin/nvcc', None)), \
                mock.patch.object(setup, 'cmake_build', side_effect=compile_fixture), \
                mock.patch.object(setup, 'cpu_floor', return_value=''), \
                mock.patch.object(setup, 'cpu_info', return_value=('test', set())):
            with self.assertRaisesRegex(RuntimeError, 'compiler failed'):
                setup.build_engine(self.gpu, 'gpu', False, self.root)
        self.assertEqual({p.name: p.read_bytes() for p in eng.iterdir()}, original)
        with self.assertRaises(SystemExit):
            setup.require_verified_engine(eng / setup.EXE)

    def test_unverified_source_archive_is_not_compiled(self):
        archive = self.root / 'source.zip'
        prefix = 'llama.cpp-' + setup.LLAMA_CPP_COMMIT + '/'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(prefix + 'ggml/CMakeLists.txt', 'attacker build script')
            z.writestr(prefix + 'gguf-py/__init__.py', '')
        with mock.patch.object(setup, 'LLAMA_CPP_ZIP', str(archive)):
            with self.assertRaises((ValueError, SystemExit)):
                setup.get_llama_cpp()
        self.assertFalse((self.root / 'third_party' / 'llama.cpp' / 'ggml' / 'CMakeLists.txt').exists())


if __name__ == '__main__':
    unittest.main()
