# Fork security review — 2026-10-05

Scope: FV137/Strata at upstream `6f32ec070f23ced9f50e704d854d775da52591ab` (v0.1.39).
This is a source review and local regression exercise of the Python server, container entry point,
installer download paths, and MCP boundary. It is not an audit of all C++/CUDA/HIP code, release binaries,
dependencies or model weights, or the separate experimental Intel server. No GPU inference or container image
build was run.

The Orca installer addition is a separate change (PR #1). This hardening branch starts from the fork's
main branch and does not require the Orca patch.

## First patch

| Finding | Exposure before this patch | Change |
| --- | --- | --- |
| Non-loopback startup accepted no key | `--host 0.0.0.0` only warned; Docker used that host with an empty key and a broadly published example port | Require a key before binding, loading the model or starting MCP. Direct `serve()` callers also pass the check. Docker validates before setup. |
| Docker key changes could be ignored on reuse | `API_KEY` was passed only during first setup; a saved volume skipped it | Apply `STRATA_API_KEY` / `API_KEY` on every start. Reject conflicting or explicitly empty values. Keep the secret out of the entry point's setup command line. |
| Unbounded and incomplete POST reads | Most routes trusted Content-Length, read without a deadline, and accepted EOF before the advertised length if the bytes were valid JSON | Central body reader: 32 MiB API limit, 64 KiB controls, absolute read deadline, framing validation, reject incomplete bodies before changing state. |
| MCP HTTP redirect handling | The default urllib opener could follow a redirect with configured credentials and the MCP session header | Reject redirects for requests, notifications and session cleanup. Configure the final MCP URL directly. Bound the error-body read too. |

These are conditional exposures, not evidence that a Strata installation has been compromised. The existing
Host/Origin controls and constant-time API key comparison remain in place. The ordinary localhost startup
still works without a key. `/health` remains available without the key for health checks.

Startup key precedence is explicit CLI, environment, then config. A non-string, surrounding whitespace, or
explicitly empty CLI/environment key is an error; there is no fallback from an explicit empty value to a saved
key. A config without a key remains valid for loopback. Hostnames other than literal `localhost` require a key,
even if they currently resolve to loopback.

The body deadline is 30 seconds total, including clients that keep sending small fragments. `/load` and
`/unload` retain their 2-second deadline. It does not time-limit inference. Controls are `/settings`, `/config`,
`/load`, `/unload`, `/v1/load`, `/v1/unload`, and `/v1/vram`. API bodies larger than 32 MiB now get 413; clients
must reduce their payload. Chunked uploads are explicitly rejected; send a Content-Length. This does not cap
aggregate connections, JSON parsing cost or the size of remote image responses.

Docker examples now use `-p 127.0.0.1:8080:8080` and require a key for the container's listener. The key remains
necessary because the container network is a separate exposure. Environment keys must be supplied again when
recreating a container; they are not newly written into the model config by the entry point. A saved config key
still works when neither environment alias is supplied and is preserved through `REINSTALL=1`.

## Remaining work, in priority order

| Priority and boundary | Evidence / practical consequence | Next change |
| --- | --- | --- |
| High when vision accepts requests from another trust level: image sources | `serve/server.py: Vision.load` accepts arbitrary HTTP(S) sources and server-local paths, follows redirects and reads without a byte cap. A request can make the server contact an internal service or read a local picture. Image decoding follows the read; this is not a demonstrated general text-file exfiltration primitive. | Make uploaded image bytes the default boundary; decide whether URL fetching is needed. If retained, enforce destination policy across DNS resolution and redirects, plus response-size and decode limits. Limit any local-file support to explicit directories. |
| High if broad MCP tools are enabled: execution authority | `serve/mcp.py: StdioTransport.start` inherits the server environment and user rights. `McpHub` executes configured tools at the model's request. The server checks API access and page origin but has no per-tool human approval boundary or OS sandbox. | Run tools under a dedicated account/container with narrow mounts, explicit environment and outbound access; add a per-tool policy before granting file writes or shell access. Local hosting does not prevent prompt injection through tool results or repository content. |
| High-impact supply-chain boundary: installer binaries | `setup.py: get_prebuilt`, `get_prebuilt_hip` and `get_llama_cpp` download/extract artifacts. The prebuilt path checks archive metadata/compatibility rather than an independently pinned binary digest; release fallback can use a newer release. | Pin a reviewed release and verify its artifacts against trusted digests or attestations before extraction/execution. The Orca shard hashes in PR #1 cover those weights, not the installer or engine. |
| Resource exhaustion | `Server` is threaded; body limits do not cap simultaneous connections or add an absolute header deadline. MCP response/result truncation occurs after some unbounded transport reads, and image decoding/native input needs its own limits. | Add connection/header limits and bounded MCP/image transport reads; keep external access behind a suitably configured authenticated proxy. |
| Secrets and persisted data | Configs can contain keys; configured MCP subprocesses inherit environment credentials. Browser chat storage and optional request monitoring retain conversation data. | Keep production data and credentials out of initial evaluation. Define retention, file permissions and child-process environment rules before integrating private work. |
| Native model/parser trust | The review did not validate memory safety in GGUF parsing, pack conversion, image decoding or the GPU engine. | Evaluate model files and binaries in a dedicated account or machine; do not treat a successful hash check as proof that the original content is safe. |

Do not expose this as a service for mutually untrusted users: a shared key provides access, not tenant isolation
or separate tool permissions. A tunnel to an unkeyed loopback listener can still publish it; a server cannot
infer that a local proxy is forwarding external traffic. Set a key for that deployment too.

## Verification

The unmodified server's 22 existing security tests passed before editing. New regression cases were run against
the original code first: they reproduced the unkeyed bind, late key validation, Docker restart/key behavior,
body framing/length/deadline problems and redirect following. All network probes used local test servers and
dummy credentials; no third-party service was tested.

The focused set passes 43 tests:

```bash
python -m unittest serve.test_hardening serve.test_mcp_redirects serve.test_security tools.test_docker_security -q
```

It includes normal authenticated and local requests, malformed and truncated bodies, oversized declarations
without uploading oversized content, a client that drips bytes, early startup rejection, Docker first/reused
volume behavior, key rotation, saved-key preservation during reinstall, and MCP 301/302/303/307/308 redirects
including session cleanup. The reinstall test uses the real setup config writer inside the setup substitute.

The Docker test executes the entry point in a temporary directory with a small setup substitute. It checks
shell flow, environment and arguments; it does not claim that a GPU image was built or started. Existing
telemetry code emits ResourceWarnings for unclosed `/proc` reads in this environment; that is outside this patch.

The broader server suite ran 282 tests: 275 passed and 7 were skipped for unavailable model/tokenizer fixtures
or Windows-only coverage. Shell syntax and whitespace checks passed too:

```bash
python -m unittest discover -s serve -p 'test_*.py' -q
sh -n docker-entrypoint.sh
git diff --check
```

The test environment uses Python 3.12, the repository's Jinja2/MarkupSafe and regex pins, and the optional
jsonschema package for the schema-validation tests. No installer dependency pins are changed by this patch.

A separate controlled image-loader probe read a temporary PNG by local path, file URI and loopback HTTP URL.
That confirms the remaining source boundary in the table; it did not read private files or contact other services.

## Change review

### Standards

Read-only review of the complete diff found no actionable AGENTS.md / SECURITY.md violations and no meaningful
code-smell findings. The shared access-policy helper and central body reader consolidate the security checks.
This review did not independently rerun the test suite or validate the native engine.

### Spec

Review caught a Docker reinstall gap: the preflight accepted a saved API key, but setup's config writer discarded
it. The fix preserves an already-saved key before copying the regenerated config back to the volume; the added
regression checks both configs and keeps the key out of process arguments and output. The broader server test
result above is separate from the review. Follow-up review confirmed saved-key reinstall, environment-only
reinstall and key rotation during reinstall. No spec blockers remain.

Review totals: Standards — 0 findings. Spec — 1 finding, resolved; 0 remaining.

## References

- [Docker port publishing](https://docs.docker.com/engine/network/port-publishing/): default publication and
  loopback mappings. Docker documents limitations for old Engine versions and deliberately enabled direct routing.
- [Python urllib.request](https://docs.python.org/3.12/library/urllib.request.html): custom redirect handling.
- [Security policy](../SECURITY.md), [container instructions](INSTALL.md#docker-linux), and
  [server settings](DETAILS.md) describe the user-visible boundaries.
