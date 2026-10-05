"""MCP authorization, subprocess isolation, and bounded transport regressions.

All subprocesses are local test fixtures. No package downloads or external servers.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from serve import mcp

FAKE = str(Path(__file__).with_name("mcp_fake_server.py").resolve())


def trusted(**extra):
    return {"command": sys.executable, "args": [FAKE], "execution": "trusted", "allowed_tools": ["echo"], **extra}


@contextmanager
def started(cfg):
    hub = mcp.McpHub({"test": cfg}, {"start_timeout_s": 3, "timeout_s": 3})
    try:
        hub.start(wait=True)
        yield hub
    finally:
        hub.close()


class Authorization(unittest.TestCase):
    def test_omitted_allowlist_exposes_no_tools_and_cannot_be_called_directly(self):
        cfg = trusted()
        del cfg["allowed_tools"]
        with started(cfg) as hub:
            self.assertEqual(hub.openai_tools(), [])
            with self.assertRaisesRegex(mcp.McpError, "not allowed"):
                hub.servers["test"].call("echo", {"text": "denied"}, 1)

    def test_only_exact_names_are_discovered_and_denied_call_never_reaches_process(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "calls.jsonl"
            with started(trusted(env={"FAKE_MCP_LOG": str(log)})) as hub:
                self.assertEqual([x["function"]["name"] for x in hub.openai_tools()], ["test__echo"])
                self.assertTrue(hub.call("test__echo", {"text": "allowed"})["ok"])
                server = hub.servers["test"]
                # Simulate a stale/hallucinated route and untrusted discovery metadata.
                server.tools.append({"name": "die", "annotations": {"readOnlyHint": True}})
                hub._routes["test__die"] = (server, "die")
                self.assertFalse(hub.call("test__die", {})["ok"])
                with self.assertRaisesRegex(mcp.McpError, "not allowed"):
                    server.call("die", {}, 1)
                with self.assertRaisesRegex(mcp.McpError, "not allowed"):
                    server.transport.request("tools/call", {"name": "die", "arguments": {}}, 1)
                with self.assertRaisesRegex(mcp.McpError, "not allowed"):
                    server.transport.notify("tools/call", {"name": "die", "arguments": {}})
                self.assertEqual([json.loads(x)["call"] for x in log.read_text().splitlines()], ["echo"])
                self.assertEqual([x["function"]["name"] for x in hub.openai_tools()], ["test__echo"])

    def test_wildcard_and_case_normalization_do_not_grant_tools(self):
        with started(trusted(allowed_tools=["*", "Echo", "test__echo"])) as hub:
            self.assertEqual(hub.openai_tools(), [])

    def test_stale_route_cannot_bypass_revoked_allowlist(self):
        with started(trusted()) as hub:
            hub.routes()
            hub.servers["test"].cfg["allowed_tools"] = []
            self.assertFalse(hub.call("test__echo", {"text": "denied"})["ok"])
            with self.assertRaisesRegex(mcp.McpError, "not allowed"):
                hub.servers["test"].transport.request("tools/call", {"name": "echo", "arguments": {}}, 1)

    def test_http_transport_blocks_unapproved_direct_calls_before_network(self):
        transport = mcp.HttpTransport("test", {"url": "http://127.0.0.1:9/mcp", "allowed_tools": ["echo"]})
        with self.assertRaisesRegex(mcp.McpError, "not allowed"):
            transport.request("tools/call", {"name": "die", "arguments": {}}, 1)
        with self.assertRaisesRegex(mcp.McpError, "not allowed"):
            transport.notify("tools/call", {"name": "die", "arguments": {}})


class Configuration(unittest.TestCase):
    def test_invalid_security_configuration_fails_closed(self):
        patches = [
            {"allowed_tools": "echo"}, {"allowed_tools": [1]}, {"allow_tools": ["echo"]},
            {"env": []}, {"env": {"TOKEN": 2}}, {"env": {"BAD=KEY": "v"}},
            {"inherit_env": "TOKEN"}, {"inherit_env": ["*"]}, {"inherit_env": ["HOME"]},
            {"inherit_env": ["PATH"]}, {"inherit_env": ["PYTHONPATH"]}, {"env": {"LD_PRELOAD": "x"}},
            {"env": {"BASH_ENV": "x"}}, {"env": {"NODE_OPTIONS": "--require x"}},
            {"execution": "trustd"}, {"execution": False}, {"args": [1]}, {"args": False},
            {"cwd": 1}, {"disabled": "true"}, {"max_response_bytes": True},
            {"max_response_bytes": 0}, {"max_response_bytes": 100_000_000},
            {"execution": "sandbox", "sandbox": {"network": True}},
            {"execution": "sandbox", "sandbox": {"read_only": "x"}},
            {"sandbox": {}},  # Do not silently accept ignored sandbox settings in trusted mode.
        ]
        for patch in patches:
            with self.subTest(patch=patch), self.assertRaises(SystemExit):
                mcp.hub_from_config({"mcp_servers": {"test": trusted(**patch)}})

    def test_invalid_http_urls_headers_or_mixed_execution_settings_fail_closed(self):
        for cfg in [
            {"url": "file:///etc/passwd"}, {"url": "ftp://localhost/mcp"},
            {"url": "http://user:secret@localhost/mcp"}, {"url": "http://localhost/mcp#x"},
            {"url": 123}, {"url": "http://localhost/mcp", "headers": []},
            {"url": "http://localhost/mcp", "headers": {"X-Test": "a\r\nb"}},
            {"url": "http://localhost/mcp", "command": "python"},
            {"url": "http://localhost/mcp", "execution": "trusted"},
        ]:
            with self.subTest(cfg=cfg), self.assertRaises(SystemExit):
                mcp.hub_from_config({"mcp_servers": {"test": cfg}})

    def test_unknown_global_settings_and_nonfinite_limits_fail_closed(self):
        for settings in [[], "x", {"allow_tools": ["echo"]}, {"timeout_s": float("inf")},
                         {"max_rounds": float("nan")}, {"start_timeout_s": True}]:
            with self.subTest(settings=settings), self.assertRaises(SystemExit):
                mcp.hub_from_config({"mcp_servers": {"test": trusted()}, "mcp": settings})

    def test_mounts_cannot_replace_runtime_or_expose_entire_host(self):
        with tempfile.TemporaryDirectory() as d:
            cases = [
                {"source": "/", "target": "/data"},
                {"source": str(Path.home()), "target": "/data"},
                {"source": "/proc", "target": "/data"},
                {"source": d, "target": "/usr"},
                {"source": d, "target": "/app/../usr"},
                {"source": d, "target": "/data", "writable": True},
                {"source": "relative", "target": "/data"},
            ]
            for mount in cases:
                with self.subTest(mount=mount), self.assertRaises(SystemExit):
                    mcp.hub_from_config({"mcp_servers": {"test": trusted(execution="sandbox", sandbox={"read_only": [mount]})}})


class ProcessEnvironment(unittest.TestCase):
    def test_real_child_receives_only_explicit_env_and_fixed_runtime_values(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "env.json"
            code = "import json,os,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(os.environ)))"
            cfg = trusted(args=["-I", "-c", code, str(output)], env={"MCP_TOKEN": "explicit"}, inherit_env=["MCP_APPROVED"])
            with mock.patch.dict(os.environ, {"STRATA_API_KEY": "private", "AWS_SECRET_ACCESS_KEY": "private",
                                             "MCP_APPROVED": "approved", "MCP_TOKEN": "inherited-wrong",
                                             "HOME": d, "PATH": d, "PYTHONPATH": d,
                                             "BASH_ENV": str(Path(d) / "inject.sh")}, clear=True):
                transport = mcp.StdioTransport("env", cfg)
                try:
                    transport.start()
                    transport.proc.wait(timeout=5)
                finally:
                    transport.close()
            observed = json.loads(output.read_text())
            self.assertEqual(observed["MCP_TOKEN"], "explicit")
            self.assertEqual(observed["MCP_APPROVED"], "approved")
            for key in ("STRATA_API_KEY", "AWS_SECRET_ACCESS_KEY", "HOME", "PYTHONPATH", "BASH_ENV"):
                self.assertFalse(key in observed, key)
            self.assertNotEqual(observed.get("PATH"), d)

    def test_launch_arguments_are_not_expanded_by_a_shell(self):
        with tempfile.TemporaryDirectory() as d:
            output, marker = Path(d) / "argv.json", Path(d) / "pwned"
            arg = "$(touch " + str(marker) + "); & `touch " + str(marker) + "`"
            code = "import json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))"
            transport = mcp.StdioTransport("argv", trusted(args=["-I", "-c", code, str(output), arg]))
            try:
                transport.start()
                self.assertEqual(transport.proc.wait(timeout=5), 0)
            finally:
                transport.close()
            self.assertEqual(json.loads(output.read_text()), [arg])
            self.assertFalse(marker.exists())

    @unittest.skipUnless(sys.platform == "linux", "Linux launcher argument-FD probe")
    def test_sandbox_secrets_use_private_fd_not_process_argv_or_launcher_env(self):
        # Stand-in launcher inspects the real process boundary; namespace behavior
        # itself is checked separately by LinuxSandbox against actual bubblewrap.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            output, probe = root / "boundary.json", root / "launcher"
            with (root / "unrelated").open("w+b") as unrelated:
                os.set_inheritable(unrelated.fileno(), True)
                script = """import json,os,pathlib,sys
try:
 os.fstat(UNRELATED)
 leaked_fd=True
except OSError: leaked_fd=False
policy=[]
if '--args' in sys.argv:
 with os.fdopen(int(sys.argv[sys.argv.index('--args')+1]),'rb') as f:
  policy=f.read().decode().rstrip('\\0').split('\\0')
pathlib.Path(OUTPUT).write_text(json.dumps({'argv':sys.argv,'cmdline':pathlib.Path('/proc/self/cmdline').read_bytes().decode(),'launcher_has_secret':'MCP_PRIVATE_TOKEN' in os.environ,'policy':policy,'leaked_fd':leaked_fd}))
"""
                script = script.replace("UNRELATED", str(unrelated.fileno())).replace("OUTPUT", repr(str(output)))
                probe.write_text("#!" + sys.executable + "\n" + script)
                probe.chmod(0o700)
                cfg = {"command": "/usr/bin/true", "allowed_tools": [], "env": {"MCP_PRIVATE_TOKEN": "fd-only-secret"}}
                with mock.patch("shutil.which", return_value=str(probe)):
                    transport = mcp.StdioTransport("private", cfg)
                    try:
                        transport.start()
                        self.assertEqual(transport.proc.wait(timeout=5), 0, list(transport.stderr))
                    finally:
                        transport.close()
            result = json.loads(output.read_text())
            self.assertNotIn("fd-only-secret", result["cmdline"])
            self.assertFalse(result["launcher_has_secret"])
            self.assertIn("fd-only-secret", result["policy"])
            self.assertIn("--unshare-net", result["policy"])
            self.assertEqual(result["argv"][-2:], ["--", "/usr/bin/true"])
            self.assertFalse(result["leaked_fd"])

    def test_shell_metacharacters_remain_literal_arguments(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / "pwned"
            literal = "$(touch " + str(marker) + "); & `touch " + str(marker) + "`"
            with started(trusted()) as hub:
                self.assertEqual(hub.call("test__echo", {"text": literal})["text"], literal)
            self.assertFalse(marker.exists())

    def test_no_bubblewrap_refuses_default_execution_without_running_command(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / "ran"
            cfg = {"command": sys.executable, "args": ["-c", f"open({str(marker)!r}, 'w').close()"], "allowed_tools": []}
            with mock.patch("shutil.which", return_value=None), started(cfg) as hub:
                self.assertEqual(hub.servers["test"].status, "failed")
                self.assertIn("sandbox", hub.servers["test"].error.lower())
            self.assertFalse(marker.exists())


class BoundedTransports(unittest.TestCase):
    def test_invalid_rpc_id_fails_the_pending_request_without_reader_crash(self):
        reply = json.dumps({"jsonrpc": "2.0", "id": {}, "result": {}})
        code = "import sys; sys.stdin.readline(); print(" + repr(reply) + ", flush=True); sys.stdin.read()"
        with started(trusted(args=["-c", code])) as hub:
            self.assertEqual(hub.servers["test"].status, "failed")
            self.assertIn("invalid", hub.servers["test"].error)

    def test_malformed_discovery_result_fails_cleanly(self):
        reply = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": [1]}})
        code = "import sys; sys.stdin.readline(); print(" + repr(reply) + ", flush=True); sys.stdin.read()"
        with started(trusted(args=["-c", code])) as hub:
            self.assertEqual(hub.servers["test"].status, "failed")
            self.assertIn("serverInfo", hub.servers["test"].error)

    def test_oversize_stdio_response_and_log_line_stop_process(self):
        with started(trusted(allowed_tools=["big"], max_response_bytes=1024)) as hub:
            result = hub.call("test__big", {"n": 5000})
            self.assertFalse(result["ok"])
            self.assertIn("limit", result["text"])
        code = "import sys,time; sys.stderr.write('x'*20000); sys.stderr.flush(); time.sleep(10)"
        with started(trusted(args=["-c", code])) as hub:
            self.assertEqual(hub.servers["test"].status, "failed")
            self.assertIn("limit", hub.servers["test"].error)

    def test_http_json_and_sse_limits_reject_unbounded_lines_and_events(self):
        bodies = [
            ("application/json", json.dumps({"id": 1, "result": {"large": "x" * 5000}}).encode()),
            ("text/event-stream", b"data: " + b"x" * 5000 + b"\n\n"),
            ("text/event-stream", (b": notification\n\n" * 1000) + b'data: {"id":1,"result":{}}\n\n'),
            ("text/event-stream", (b"data: \"x\"\n" * 1000) + b"\n"),
        ]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", self.server.content_type)
                self.end_headers()
                self.wfile.write(self.server.body)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for content_type, body in bodies:
                with self.subTest(content_type=content_type, prefix=body[:30]):
                    server.content_type, server.body = content_type, body
                    transport = mcp.HttpTransport("limits", {"url": f"http://127.0.0.1:{server.server_port}/mcp", "max_response_bytes": 1024})
                    with self.assertRaisesRegex(mcp.McpError, "limit"):
                        transport.request("initialize", {}, 2)
        finally:
            server.shutdown()
            server.server_close()

    def test_sse_reader_never_requests_unbounded_input(self):
        class CheckedReader(io.BytesIO):
            def readline(self, size=-1):
                if size < 0 or size > 1025:
                    raise AssertionError("unbounded SSE read")
                return super().readline(size)
            def __iter__(self):
                raise AssertionError("unbounded SSE iteration")
        transport = mcp.HttpTransport("test", {"url": "http://localhost/mcp", "max_response_bytes": 1024})
        self.assertEqual(transport._from_events(CheckedReader(b'data: {"id":1,"result":{}}\n\n'), 1), {"id": 1, "result": {}})


class LinuxSandbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform != "linux" or not shutil.which("bwrap", path="/usr/bin:/bin"):
            raise unittest.SkipTest("Linux bubblewrap is not installed")
        # Independently check host namespace support, not the implementation under test.
        try:
            probe = subprocess.run([shutil.which("bwrap", path="/usr/bin:/bin"), "--unshare-all", "--unshare-user",
                                    "--die-with-parent", "--new-session", "--ro-bind", "/usr", "/usr",
                                    "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
                                    "--proc", "/proc", "--dev", "/dev", "--", "/usr/bin/true"],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        except subprocess.TimeoutExpired:
            raise unittest.SkipTest("host bubblewrap namespace probe timed out")

        if probe.returncode:
            raise unittest.SkipTest("host cannot create bubblewrap namespaces: " + probe.stderr.decode().strip()[:300])

    def test_real_sandbox_hides_host_data_env_network_and_is_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            app, data, output = root / "app", root / "data", root / "out"
            app.mkdir(); data.mkdir(); output.mkdir()
            (data / "read.txt").write_text("visible")
            (root / "secret").write_text("private")
            script = app / "probe.py"
            script.write_text('''import json,os,pathlib,socket
out={"secret_visible":pathlib.Path(os.environ["PROBE_SECRET"]).exists(), "read":pathlib.Path("/data/read.txt").read_text(), "cwd":os.getcwd(), "env":dict(os.environ)}
try:
 pathlib.Path("/data/changed").write_text("bad")
 out["readonly"]=False
except OSError: out["readonly"]=True
try:
 socket.create_connection(("127.0.0.1",int(os.environ["PROBE_PORT"])),timeout=0.5).close()
 out["network"]=True
except OSError: out["network"]=False
out["proc_hosts"] = [p.name for p in pathlib.Path("/proc").iterdir() if p.name.isdigit()]
pathlib.Path("/work/result.json").write_text(json.dumps(out))
''')
            import socket
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0)); listener.listen()
                cfg = {"command": "/usr/bin/python3", "args": ["-I", "/app/probe.py"], "allowed_tools": [],
                       "env": {"PROBE_SECRET": str(root / "secret"), "PROBE_PORT": str(listener.getsockname()[1])},
                       "sandbox": {"read_only": [{"source": str(app), "target": "/app"}, {"source": str(data), "target": "/data"}],
                                   "read_write": [{"source": str(output), "target": "/work"}]}}
                with mock.patch.dict(os.environ, {"STRATA_API_KEY": "private"}):
                    transport = mcp.StdioTransport("sandbox", cfg)
                    try:
                        transport.start()
                        self.assertEqual(transport.proc.wait(timeout=5), 0, list(transport.stderr))
                    finally:
                        transport.close()
            result = json.loads((output / "result.json").read_text())
            self.assertFalse(result["secret_visible"])
            self.assertEqual(result["read"], "visible")
            self.assertTrue(result["readonly"])
            self.assertFalse(result["network"])
            self.assertEqual(result["cwd"], "/work")
            self.assertNotIn("STRATA_API_KEY", result["env"])
            self.assertLess(len(result["proc_hosts"]), 10)
            self.assertFalse((data / "changed").exists())


if __name__ == "__main__":
    unittest.main()
