# Security of tools called by Strata

This page covers `serve/mcp.py`: MCP tools the Strata chat model can call. It does not configure the
opposite direction, where an external assistant manages Strata through `tools/strata_mcp.py`.

MCP configuration is administrator-owned executable policy. Protect the run config and `--mcp-config`
file from untrusted writes. A model, webpage, tool description, tool annotation, or chat request cannot
add permissions. Restart Strata after changing a configuration file.

## Exact tool permissions

Every server needs an explicit list of tool names to make any tools available:

```json
{
  "mcp_servers": {
    "search": {
      "url": "http://127.0.0.1:3000/mcp",
      "allowed_tools": ["search", "get_document"],
      "headers": {"Authorization": "Bearer replace-with-server-token"},
      "max_response_bytes": 1048576
    }
  }
}
```

Use the server's original tool names, not Strata's `search__search` names. Names are exact and case
sensitive. `"*"` is a literal name, not a wildcard. Omitting `allowed_tools`, or setting it to `[]`,
permits no tools. Strata filters discovery, checks every invocation again, and checks the transport
before sending `tools/call`. Stale routes, direct calls, and `readOnlyHint` or similar annotations do
not grant permissions. A tool must also still be present after a server restarts and lists its tools.

An allowed tool can still be dangerous. A `read_file` name says nothing about what its implementation
does. Allowing `run_shell`, arbitrary SQL, or a write tool delegates that capability to the model,
including when its decision is influenced by hostile content. Review the implementation and narrow
the server's data access and credentials. Tool authorization cannot make an untrusted executable safe.

## Linux subprocess isolation

Stdio servers use `execution: "sandbox"` by default. Strata requires a system-installed `bwrap`
(bubblewrap) in `/usr/bin` or `/bin` and working Linux user, mount, PID, IPC, UTS and network namespaces.
It does not install bubblewrap, MCP packages, or runtimes. A missing/unsupported sandbox or a failed
namespace setup leaves that server failed with no tools. **There is no fallback to host execution.**
Windows and macOS need an independently isolated HTTP server, or the explicit trusted mode below.

The sandbox has:

- Read-only OS runtime trees: `/usr`, `/bin`, `/sbin`, `/lib`, `/lib64` where present. The system
  installation and files under those trees must be trusted and suitable for sharing.
- A fresh root filesystem, private `/tmp`, empty `/home/mcp`, private procfs and minimal devices.
  It does not share host home, `/etc`, `/run`, host `/tmp`, host processes, IPC namespaces, or network.
- A new terminal session, dropped capabilities, bubblewrap's no-new-privileges protection, and
  `--die-with-parent`. Only the stdio pipes and a private launch-policy descriptor reach bubblewrap; other inherited descriptors close.
  Bubblewrap consumes and closes the launch-policy descriptor before executing the tool.
- Explicit application/data mounts. `read_only` mounts permit reads; `read_write` mounts permit host
  writes. Mount targets must be canonical paths under `/app`, `/data` or `/work` and must not overlap.
  Sources must be absolute existing regular files or directories. Whole host homes, host roots and
  live system/IPC trees are rejected. Narrow directories within a home are explicit data grants.
- No network option. The child cannot reach the internet or host localhost services. Use an
  independently managed, isolated HTTP MCP service for tools that need network access.

`command` must be an absolute path **inside the sandbox**. `cwd` is also a sandbox path and defaults
to `/work`. Host working directories and programs outside the read-only runtime are not automatically
mounted. Mount an application's installed code and dependencies explicitly. Provision packages before
starting Strata; do not use on-demand installers such as `npx -y` or `uvx` as a security boundary.

For example, with an already installed Python server in `/opt/my-mcp` and notes in
`/home/me/Documents/notes`:

```json
{
  "mcp_servers": {
    "notes": {
      "command": "/usr/bin/python3",
      "args": ["-I", "/app/server.py", "--root", "/data"],
      "execution": "sandbox",
      "allowed_tools": ["list_notes", "read_note"],
      "sandbox": {
        "read_only": [
          {"source": "/opt/my-mcp", "target": "/app"},
          {"source": "/home/me/Documents/notes", "target": "/data"}
        ],
        "read_write": []
      }
    }
  }
}
```

Adapt arguments and names to that installed server. If persistent writes are needed, deliberately add
an application-specific directory, for example `{"source":"/srv/mcp-output","target":"/work"}` to
`read_write`. Do not mount directories with credentials, Unix sockets, service control sockets,
sensitive hard links or submounts. Directory grants include their contents: an administrator must
keep those grants narrow and prevent other host users from replacing them during startup. A socket
inside a granted directory can give access to a host service despite network namespace isolation.

### Runnable local example

On Linux with working bubblewrap and `/usr/bin/python3`, run this from the Strata repository root.
It starts only the repository's test fixture, mounts that single file, and calls two allowed tools:

```bash
python3 - <<'PY'
from pathlib import Path
from serve.mcp import McpHub

hub = McpHub({"demo": {
    "command": "/usr/bin/python3",
    "args": ["-I", "/app/mcp_fake_server.py"],
    "allowed_tools": ["echo", "add"],
    "sandbox": {"read_only": [{
        "source": str(Path("serve/mcp_fake_server.py").resolve()),
        "target": "/app/mcp_fake_server.py"
    }]}
}})
try:
    hub.start(wait=True)
    assert hub.servers["demo"].status == "ready", hub.servers["demo"].error
    print(hub.call("demo__echo", {"text": "sandbox ready"}))
    print(hub.call("demo__add", {"a": 2, "b": 40}))
    assert not hub.call("demo__die", {})["ok"]
finally:
    hub.close()
PY
```

### What this boundary does not provide

This is a Linux namespace/filesystem sandbox, not a virtual machine or a complete hostile-code
containment system. It shares the host kernel and has no seccomp syscall filter, cgroup CPU/memory
quota, disk quota, or per-tool argument/path policy. A malicious server can consume resources,
exercise kernel vulnerabilities, modify explicitly writable mounts, read all explicitly shared data,
and misuse granted credentials. Tool cancellation is cooperative; trusted servers may ignore it.
Use an independently managed container/VM with resource controls for stronger isolation. The
sandbox runtime, approved mounts, bubblewrap and kernel updates remain the administrator's responsibility.

The launcher follows the upstream [bubblewrap manual](https://github.com/containers/bubblewrap/blob/main/bwrap.xml)
and [security guidance](https://github.com/containers/bubblewrap/blob/main/README.md#sandbox-security).
Bubblewrap supplies mechanisms; the mount and namespace policy above supplies this application's boundary.

## Explicit trusted mode

`"execution": "trusted"` deliberately runs the command with the host account's filesystem, network,
and process privileges. This is **not sandboxed**, even though the environment is reduced and tools
are allowlisted. Use it only for an audited server you would run directly as that account, and keep
Strata bound to localhost or authenticated. Windows `.cmd`/`.bat` wrappers are refused; use an
absolute `.exe` runtime such as `node.exe` or `python.exe` with the installed server as an argument.

```json
{
  "mcp_servers": {
    "audited_local": {
      "command": "C:\\Python312\\python.exe",
      "args": ["-I", "C:\\McpTools\\server.py"],
      "execution": "trusted",
      "allowed_tools": ["lookup"],
      "env": {"SERVICE_REGION": "local"},
      "inherit_env": ["MCP_LOOKUP_TOKEN"]
    }
  }
}
```

Trusted mode defaults to the filesystem root as `cwd`, not Strata's current directory. Set an
explicit absolute `cwd` when necessary. It cannot be combined with ignored `sandbox` settings.

## Environment, HTTP and resource limits

Subprocesses do not inherit Strata's environment. The base environment is a fixed `/usr/bin:/bin`
`PATH` and `C.UTF-8` locale on Unix, or `SystemRoot` and its `System32` path on Windows. Sandboxes
add the private `HOME=/home/mcp` and `TMPDIR=/tmp`. No host home, proxy, cloud token, API key, shell
startup, Python module path, or user executable search path is inherited automatically.

`env` explicitly supplies string values; `inherit_env` names exact variables to copy if present.
Explicit values take precedence. Wildcards are invalid. Runtime-loading and ambient-path variables
such as `PATH`, `HOME`, `LD_*`, `DYLD_*`, `PYTHON*`, `NODE_OPTIONS`, `BASH_ENV` and `ENV` are refused,
even if explicitly requested. Values are passed as data, never expanded by a shell. Only give a tool
the credentials it needs. Bubblewrap receives options and secret environment values through a private
anonymous argument file (at most 65,536 bytes), not its public process arguments or loader environment.
The tool receives approved secrets in its own environment. This is not a secret vault against privileged
or same-account host processes, and secrets supplied as the tool's own command-line arguments remain
visible as arguments when that tool executes.

HTTP servers are administrator-selected `http://` or `https://` endpoints, with optional explicit
headers. URL credentials, fragments, malformed headers and non-HTTP schemes are refused. Use HTTPS
for credentials outside trusted localhost transport. Strata does not follow redirects or use ambient
proxy variables for MCP requests. It cannot sandbox a remote HTTP service; isolation, permissions,
network policy and authentication on that service are the administrator's responsibility.

Each server's `max_response_bytes` defaults to 1,048,576 and accepts integers from 1,024 through
16,777,216. It limits a stdio JSON line, an HTTP JSON response, or the **whole SSE response including
notifications, comments, and incomplete events**, before decoding or JSON parsing. Stderr lines have
a separate 8,192-byte limit with only 40 recent bounded log lines retained. Oversized stdio output
ends the process; oversized HTTP output fails the call. `max_result_chars` remains a separate limit
on text shown to the model, not a transport security limit. These limits do not bound arbitrary
memory allocations by a subprocess or replace service resource quotas.

Unknown server, sandbox and global `mcp` settings fail validation. Types are checked without
coercion. Do not copy unsupported client settings and assume they are enforced.

## Migrating an existing configuration

1. Add explicit `allowed_tools` to each server. An unchanged legacy server exposes no tools.
2. For stdio, install/review the runtime and server first. On Linux configure the required mounts
   and use absolute sandbox paths. Unsupported hosts must use a separately isolated HTTP server or
   deliberately opt into `execution: "trusted"` with the consequences above.
3. Replace inherited environment assumptions with `env` and exact `inherit_env` grants. Replace
   shell wrappers with an executable runtime and a literal argument list.
4. Remove unsupported keys. Confirm needed responses fit `max_response_bytes`, then restart Strata.
5. Inspect the Monitor tab or run `python -m serve.mcp config.json`. Failed sandbox startup means
   that server has no tools; it never causes unsandboxed execution.

Run `python -m unittest serve.test_mcp_security serve.test_mcp serve.test_mcp_redirects -v` to check
local authorization, real subprocess environment probes, bounded transports and redirect protection.
The real Linux isolation test skips only when an independent bubblewrap namespace probe cannot run;
its skip reason records the host limitation. A skip is not evidence that isolation worked on that host.
