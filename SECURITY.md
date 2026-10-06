# Security

## Reporting a vulnerability

Please report anything that should not be public until it is fixed **privately**, through GitHub's private
vulnerability reporting: the repository's **Security** tab, **Report a vulnerability**
([direct link](https://github.com/Niko1221/Strata/security/advisories/new)). Say what you found, how to reproduce it
(the request, the config keys involved, the Strata version), and what an attacker gains. We answer there, fix it in
a release, and credit you in the advisory unless you ask us not to.

Ordinary hardening ideas and findings that are safe to discuss in public are welcome as an
[issue](https://github.com/Niko1221/Strata/issues) (#544 collects audit findings) or as a pull request, which we
review like our own code.

Supported: the latest release. Fixes go into the next release, not into older ones.

## What the server exposes

Strata runs a model on your PC and serves it over HTTP (`serve/server.py`). Out of the box it is reachable from
this PC only. The details and every setting are in [docs/DETAILS.md](docs/DETAILS.md) ("From other devices",
"Host names", "Web pages without an API key", "Tools from MCP servers").

- **Where it listens.** `127.0.0.1` by default. `--host 0.0.0.0` (or `"host"` in `strata-<model>.json`) opens it to
  your network. This fork refuses a non-loopback listener without an API key, before loading the model or
  starting MCP servers. Docker checks this before setup too and applies environment keys on every start.
- **API key.** `"api_key"` in the run config (or `STRATA_API_KEY`) is required on `/v1/*` and on every endpoint
  that shows the model's state, requests or answers (`/status`, `/metrics`, `/settings`, `/mcp`, `/props`, `/slots`,
  `/api/requests`, `/config`) and on every `POST`. It is compared in constant time. Set one before you open the
  server to your network or put a tunnel in front of it.
- **Host check (DNS rebinding, 0.1.38).** Without an API key the server answers only requests whose `Host` is a name
  it knows (`localhost`, an IP address, the address it listens on and, when it listens beyond this PC, this PC's
  name). Others get 403. `"allowed_hosts"` adds names; with an API key the key decides.
- **Origin check (0.1.38).** Without an API key, a browser `POST` to `/v1/*` from another web site's page (it carries
  an `Origin` header) gets 403 unless that origin is Strata's own page, `localhost`, an allowed host, or listed in
  `"trusted_origins"` / `"cors_origins"`. Changing settings (`/settings`, and `/config`: the few run config keys the
  page's Model settings may change - never the network, key, MCP or program keys), `/load` and `/unload` and the
  MCP tools are accepted only as JSON from Strata's own page (or `"trusted_origins"`), so a page elsewhere cannot
  change settings or run tools.
- **CORS** is off unless `"cors_origins"` lists origins, and then only for `/v1/*`.
- **Request bodies** are limited to 32 MiB (64 KiB for settings/config/load/unload/VRAM controls), with a 30-second
  total read deadline (2 seconds for `/load` and `/unload`). Oversized requests get 413, a body deadline gets 408,
  and malformed lengths, duplicate lengths, transfer encodings or truncated bodies get 400. This is a body-read
  limit, not a limit on inference time or the total number of connections.
- **MCP tools** are opt-in: only the servers you put in the run config, only for requests from Strata's own page
  that ask for them. They run with your user's rights, and the model decides when to call them. HTTP MCP
  redirects are rejected so configured credentials and session IDs stay at the configured endpoint; configure
  the final endpoint URL directly.
- **The request monitor** (`/api-monitor`, which keeps the last prompts and answers in memory) is off unless
  `"api_monitor": true` is set.

This fork has a limited source review and CPU-only regression checks documented in
[the hardening review](docs/SECURITY_REVIEW.md). This is not a complete audit of the native engine, model files,
dependencies or release binaries.
