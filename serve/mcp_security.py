"""Administrator-owned MCP policy and the Linux bubblewrap execution boundary.

Nothing in a model response, a tool description, or MCP annotations grants access.
There is deliberately no shell, package installer, arbitrary bwrap argument list,
network opt-out, or unsandboxed fallback here. See docs/MCP_SECURITY.md.
"""
from __future__ import annotations

import copy
import os
import posixpath
import re
import shutil
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_RESPONSE_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
STDERR_LINE_BYTES = 8192
SYSTEM_PATH = "/usr/bin:/bin"
_COMMON_KEYS = {"allowed_tools", "disabled", "max_response_bytes", "type"}
_STDIO_KEYS = {"command", "args", "env", "inherit_env", "cwd", "execution", "sandbox"}
_HTTP_KEYS = {"url", "headers"}
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
# These alter loading, shell startup or ambient runtime lookup; even explicit
# grants must not inject code into the sandbox launcher before it isolates us.
_RESERVED_ENV = {"HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "PATH", "PWD", "OLDPWD", "SHELL",
                 "COMSPEC", "SYSTEMROOT", "WINDIR", "PATHEXT", "TEMP", "TMP", "TMPDIR",
                 "ENV", "BASH_ENV", "SHELLOPTS", "BASHOPTS", "IFS", "CDPATH",
                 "GCONV_PATH", "LOCPATH", "GLIBC_TUNABLES", "JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS",
                 "_JAVA_OPTIONS", "CLASSPATH", "NODE_OPTIONS", "NODE_PATH", "RUBYOPT", "RUBYLIB"}


def _string(value, key, *, empty=False):
    if not isinstance(value, str) or "\0" in value or (not empty and not value.strip()):
        raise ValueError(f'{key} must be a {"possibly empty " if empty else "nonempty "}string without NUL')
    return value


def _strings(value, key):
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list of strings")
    for item in value:
        _string(item, key)
    return value


def _env_name(name):
    if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
        raise ValueError("environment names must be exact variable names (no wildcards)")
    upper = name.upper()
    if upper in _RESERVED_ENV or upper.startswith(("LD_", "DYLD_", "PYTHON", "PERL", "BASH_FUNC_")):
        raise ValueError(f"environment variable {name!r} controls runtime loading or host paths and is not permitted")


def _mounts(sandbox):
    if not isinstance(sandbox, dict) or set(sandbox) - {"read_only", "read_write"}:
        raise ValueError("sandbox must contain only read_only and read_write mount lists; network is always isolated")
    destinations = []
    home = Path.home().resolve()
    for kind in ("read_only", "read_write"):
        mounts = sandbox.get(kind, [])
        if not isinstance(mounts, list):
            raise ValueError(f"sandbox.{kind} must be a list")
        for mount in mounts:
            if not isinstance(mount, dict) or set(mount) != {"source", "target"}:
                raise ValueError("each sandbox mount must contain exactly source and target")
            source, target = _string(mount["source"], "source"), _string(mount["target"], "target")
            if not os.path.isabs(source):
                raise ValueError("sandbox mount source must be an absolute host path")
            try:
                path = Path(source).resolve(strict=True)
                mode = path.stat().st_mode
            except (OSError, RuntimeError) as e:
                raise ValueError(f"sandbox mount source is unavailable: {e}") from None
            # Do not admit whole homes, host roots or live OS/IPC trees. Narrow
            # data directories inside a home are deliberate administrator grants.
            if (path == home or path in home.parents or str(path) in ("/home", "/root", "/tmp", "/var", "/usr", "/bin", "/sbin", "/lib", "/lib64")
                    or any(path == Path(p) or Path(p) in path.parents for p in ("/proc", "/sys", "/dev", "/run", "/etc"))):
                raise ValueError("sandbox mount source must be a narrow application or data path, not host home/system state")
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError("sandbox mounts must be regular files or directories, never sockets/devices/FIFOs")
            if (target != posixpath.normpath(target) or not target.startswith("/")
                    or target.split("/")[1] not in ("app", "data", "work")):
                raise ValueError("sandbox mount targets must be canonical paths under /app, /data or /work")
            if any(target == prev or target.startswith(prev + "/") or prev.startswith(target + "/") for prev in destinations):
                raise ValueError("sandbox mount targets must not overlap")
            destinations.append(target)
            mount["source"] = str(path)
    return sandbox


def validate_server(cfg):
    """Validate and copy a server policy. Missing allowed_tools grants nothing."""
    if not isinstance(cfg, dict):
        raise ValueError("MCP server must be an object with command or url")
    cfg = copy.deepcopy(cfg)
    http = "url" in cfg
    unknown = set(cfg) - (_COMMON_KEYS | (_HTTP_KEYS if http else _STDIO_KEYS))
    if unknown:
        raise ValueError("unknown or incompatible MCP keys: " + ", ".join(sorted(unknown)))
    if "disabled" in cfg and not isinstance(cfg["disabled"], bool):
        raise ValueError("disabled must be a boolean")
    _strings(cfg.get("allowed_tools", []), "allowed_tools")
    limit = cfg.get("max_response_bytes", DEFAULT_RESPONSE_BYTES)
    if type(limit) is not int or not 1024 <= limit <= MAX_RESPONSE_BYTES:
        raise ValueError(f"max_response_bytes must be an integer between 1024 and {MAX_RESPONSE_BYTES}")
    if http:
        url = _string(cfg["url"], "url")
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in ("http", "https") and parsed.hostname and parsed.port != 0
        except ValueError:
            valid = False
        if not valid or parsed.username is not None or parsed.password is not None or parsed.fragment or any(c.isspace() for c in url):
            raise ValueError("url must be an http(s) endpoint without userinfo, whitespace or fragment")
        if cfg.get("type", "http") not in ("http", "streamable-http"):
            raise ValueError("type must be http or streamable-http; the old SSE transport is unsupported")
        headers = cfg.get("headers", {})
        if not isinstance(headers, dict):
            raise ValueError("headers must be an object of strings")
        for key, value in headers.items():
            _string(key, "header name")
            _string(value, "header value", empty=True)
            if not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key) or any(ord(c) < 32 or ord(c) == 127 for c in value):
                raise ValueError("headers must not contain invalid names or control characters")
    else:
        _string(cfg.get("command"), "command")
        if cfg.get("type", "stdio") != "stdio":
            raise ValueError("type must be stdio for commands")
        args = cfg.get("args", [])
        if not isinstance(args, list):
            raise ValueError("args must be a list of strings")
        for arg in args:
            _string(arg, "args", empty=True)
        if "cwd" in cfg:
            _string(cfg["cwd"], "cwd")
            if not os.path.isabs(cfg["cwd"]):
                raise ValueError("cwd must be absolute (a sandbox path in sandbox mode)")
        env = cfg.get("env", {})
        if not isinstance(env, dict):
            raise ValueError("env must be an object of strings")
        for key, value in env.items():
            _env_name(key)
            _string(value, "env value", empty=True)
        for name in _strings(cfg.get("inherit_env", []), "inherit_env"):
            _env_name(name)
        execution = cfg.get("execution", "sandbox")
        if execution not in ("sandbox", "trusted"):
            raise ValueError("execution must be sandbox or trusted")
        if execution == "trusted" and "sandbox" in cfg:
            raise ValueError("sandbox settings cannot be used with execution=trusted")
        if execution == "sandbox":
            _mounts(cfg.get("sandbox", {}))
    return cfg


def tool_allowed(cfg, name):
    return isinstance(name, str) and name in cfg.get("allowed_tools", [])


def authorize(cfg, method, params):
    if method == "tools/call" and not tool_allowed(cfg, params.get("name") if isinstance(params, dict) else None):
        raise ValueError("MCP tool is not allowed by the administrator's exact allowed_tools policy")


def child_environment(cfg):
    """No ambient credentials, proxies, homes, loader variables or executable search path."""
    env = {"PATH": SYSTEM_PATH, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    if os.name == "nt":
        root = os.environ.get("SystemRoot", r"C:\Windows")
        env = {"SystemRoot": root, "PATH": os.path.join(root, "System32")}
    for name in cfg.get("inherit_env", []):
        _env_name(name)
        if name in os.environ:
            env[name] = os.environ[name]
    env.update(cfg.get("env", {}))
    return env


def launch_spec(cfg):
    """Return argv, launcher environment, cwd. Failing sandbox setup never falls back."""
    cfg = validate_server(cfg)
    command, args = cfg["command"], cfg.get("args", [])
    env = child_environment(cfg)
    if cfg.get("execution", "sandbox") == "trusted":
        exe = shutil.which(command, path=env["PATH"]) or command
        if os.name == "nt" and Path(exe).suffix.lower() != ".exe":
            raise ValueError("trusted Windows commands must be .exe programs; .cmd/.bat shell wrappers are refused")
        # A deterministic cwd prevents importing a server module from Strata's
        # working directory. Trusted execution still has all host user rights.
        return [exe, *args], env, cfg.get("cwd", os.path.abspath(os.sep))
    if sys.platform != "linux":
        raise ValueError('MCP sandbox requires Linux bubblewrap; use an isolated HTTP server or deliberately set execution="trusted"')
    bwrap = shutil.which("bwrap", path=SYSTEM_PATH)
    if not bwrap:
        raise ValueError('MCP sandbox requires bubblewrap at /usr/bin or /bin; no unsandboxed fallback')
    if not command.startswith("/") or command != posixpath.normpath(command):
        raise ValueError("sandbox command must be an absolute canonical path inside the sandbox")
    argv = [bwrap, "--unshare-all", "--unshare-user", "--unshare-ipc", "--unshare-pid", "--unshare-net",
            "--unshare-uts", "--die-with-parent", "--new-session", "--cap-drop", "ALL", "--clearenv"]
    # Only OS runtimes are ambient filesystem grants. Never bind /, /etc, /run,
    # host /tmp, home, host procfs, credentials, devices or IPC sockets.
    for path in ("/usr", "/bin", "/sbin", "/lib", "/lib64"):
        if os.path.exists(path):
            argv += ["--ro-bind", path, path]
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/home/mcp",
             "--dir", "/app", "--dir", "/data", "--dir", "/work"]
    for kind, flag in (("read_only", "--ro-bind"), ("read_write", "--bind")):
        for mount in cfg.get("sandbox", {}).get(kind, []):
            argv += [flag, mount["source"], mount["target"]]
    env.update({"HOME": "/home/mcp", "TMPDIR": "/tmp"})
    for name, value in env.items():
        argv += ["--setenv", name, value]
    argv += ["--chdir", cfg.get("cwd", "/work"), "--", command, *args]
    # Explicit tool secrets are installed only after namespace setup, not passed
    # to bwrap's own dynamic loader. launch_command protects these options in an FD.
    return argv, {"PATH": SYSTEM_PATH, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}, "/"


@contextmanager
def launch_command(cfg):
    """Keep tool credentials out of bwrap's public argv and launcher environment.

    An anonymous mode-0600 file avoids pipe-buffer deadlocks on large policies.
    Only its descriptor is inherited; bubblewrap consumes/closes --args before
    executing the sandboxed command. The parent closes its copy on every path.
    """
    argv, env, cwd = launch_spec(cfg)
    if cfg.get("execution", "sandbox") == "trusted":
        yield argv, env, cwd, ()
        return
    # bwrap parses only options from --args; its command must stay in the
    # outer argv. Do not find the separator by value: env/args may contain "--".
    command_at = len(argv) - len(cfg.get("args", [])) - 2
    payload = b"\0".join(os.fsencode(arg) for arg in argv[1:command_at]) + b"\0"
    if len(payload) > 65536:
        raise ValueError("MCP sandbox launch configuration exceeds the 65,536-byte limit")
    with tempfile.TemporaryFile(mode="w+b") as policy:
        policy.write(payload)
        policy.flush()
        policy.seek(0)
        yield [argv[0], "--args", str(policy.fileno()), *argv[command_at:]], env, cwd, (policy.fileno(),)
