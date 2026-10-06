"""Startup access policy shared by the server and the container entry point."""
import ipaddress
import os


def api_key_of(cli_key, cfg):
    """Explicit CLI, then environment, then config. An explicit empty key never disables authentication."""
    explicit = cli_key is not None or "STRATA_API_KEY" in os.environ
    key = cli_key if cli_key is not None else os.environ.get("STRATA_API_KEY", cfg.get("api_key", ""))
    if not isinstance(key, str) or key != key.strip() or (explicit and not key):
        raise ValueError("API key must be a non-empty string without surrounding whitespace; "
                         "omit --api-key / STRATA_API_KEY for local use without a key")
    return key


def require_key_for_bind(host, key):
    """Only literal loopback addresses and localhost may listen without an API key."""
    if not isinstance(key, str) or key != key.strip():
        raise ValueError("API key must be a string without surrounding whitespace")
    if key:
        return
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise ValueError("an API key is required when listening beyond loopback; set --api-key, STRATA_API_KEY, "
                     "or api_key in the run config, or use --host 127.0.0.1")
