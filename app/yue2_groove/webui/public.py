"""Optional public access (phone / app) through an ngrok tunnel.

Turned on by two lines in ``app/.env``:

    NGROK_AUTHTOKEN=<token from dashboard.ngrok.com>
    NGROK_DOMAIN=<your free static domain, e.g. abc-def-123.ngrok-free.app>

and it refuses to open without a password (``YUE2_GROOVE_AUTH=user:password``),
because the tunnel puts the GPU server on the public internet.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

log = logging.getLogger("yue2_groove")


def _ensure_pyngrok():
    try:
        import pyngrok  # noqa: F401

        return True
    except ImportError:
        pass
    log.info("public: installing pyngrok …")
    for cmd in (
        [sys.executable, "-m", "pip", "install", "-q", "pyngrok"],
        ["uv", "pip", "install", "--python", sys.executable, "-q", "pyngrok"],
    ):
        try:
            subprocess.run(cmd, check=True, timeout=300)  # noqa: S603
            import pyngrok  # noqa: F401

            return True
        except Exception:  # noqa: BLE001, S112
            continue
    log.warning("public: pyngrok could not be installed — no public link")
    return False


def start_tunnel(port: int, auth) -> str | None:
    """Open the tunnel to *port*; returns the public URL or None (never raises)."""
    token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if not token:
        return None
    open_ok = os.environ.get("YUE2_PUBLIC_NO_PASSWORD", "").strip().lower() in ("1", "true", "yes", "ja")
    if not auth and open_ok:
        log.warning("public: no password — anyone with the link can use this server")
    if not auth and not open_ok:
        log.warning(
            "public: NGROK_AUTHTOKEN is set but YUE2_GROOVE_AUTH (user:password) is not — "
            "refusing to put the server on the internet without a password"
        )
        return None
    if not _ensure_pyngrok():
        return None
    try:
        from pyngrok import conf, ngrok

        conf.get_default().auth_token = token
        options = {"bind_tls": True}
        domain = os.environ.get("NGROK_DOMAIN", "").strip()
        if domain:
            options["domain"] = domain.replace("https://", "").strip("/")
        tunnel = ngrok.connect(str(port), "http", **options)
        url = tunnel.public_url
        # no "http://127.0.0.1" in this line: Pinokio opens the first local URL it sees
        log.info("public link (phone / app): %s", url)
        return url
    except Exception as exc:  # noqa: BLE001
        log.warning("public: tunnel failed: %s", exc)
        return None
