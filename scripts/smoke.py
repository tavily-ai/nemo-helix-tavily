# SPDX-License-Identifier: Apache-2.0

"""Check that the installed plugin loads and runs on a NeMo Helix platform.

Run it with the Python of the platform's environment after installing the plugin
wheel there, for example inside an ``nhx-api`` image, the way NVIDIA's image build
would install it::

    pip install --no-deps nemo_tavily_plugin-<version>-py3-none-any.whl
    python scripts/smoke.py

``--no-deps`` because Helix images report their own packages as version 0.0.0, so
pip would otherwise try to replace them from PyPI.

It checks, in order:

1. Helix discovers the plugin's service, CLI and skill entry points.
2. The proxy is off by default: the service mounts no routes. Authz derivation is
   clean with the proxy off and on.
3. A platform started without ``NEMO_TAVILY_ENABLED`` serves no Tavily routes, and
   one started with ``NEMO_TAVILY_ENABLED=true`` serves the status route.

It never contacts Tavily and needs no API key.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

STATUS_PATH = "/apis/tavily/v2/workspaces/default/status"
READY_TIMEOUT_SECONDS = 180


def check_discovery() -> None:
    from nemo_helix_plugin.discovery import discover_cli, discover_services, discover_skills

    assert "tavily" in discover_services(), "nemo.services entry point 'tavily' not found"
    assert "tavily" in discover_cli(), "nemo.cli entry point 'tavily' not found"
    skills = discover_skills()
    assert "tavily" in skills, "nemo.skills entry point 'tavily' not found"
    assert (skills["tavily"]() / "nemo-tavily-web-search" / "SKILL.md").is_file(), "SKILL.md missing from the wheel"
    print("ok: service, CLI and skill entry points are discovered")


def check_routes_and_authz() -> None:
    from fastapi.routing import APIRoute
    from nemo_helix_plugin.authz_discovery import _derive_service_contribution
    from nemo_helix_plugin.config import Configuration

    from nemo_tavily_plugin.config import TavilyConfig
    from nemo_tavily_plugin.service import TavilyService

    try:
        for enabled, expected in ((False, set()), (True, {"/mcp", "/status"})):
            Configuration.set_override(TavilyConfig(enabled=enabled))
            service = TavilyService()
            routes = {
                route.path
                for spec in service.get_routers()
                for route in spec.router.routes
                if isinstance(route, APIRoute)
            }
            assert routes == expected, f"enabled={enabled}: expected routes {expected}, got {routes}"
            _, problems, _ = _derive_service_contribution(service)
            assert problems == [], f"enabled={enabled}: authz problems {problems}"
    finally:
        Configuration.clear_override(TavilyConfig)
    print("ok: no routes when disabled, /mcp and /status when enabled, authz clean")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _status_code_on_platform(*, enabled: bool) -> int:
    """Start a minimal platform, return the HTTP status of the Tavily status route, stop it."""
    nemo = Path(sys.executable).with_name("nemo")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "NHX_DATA_DIR": f"{tmp}/data", "XDG_STATE_HOME": f"{tmp}/state"}
        env["NHX_BASE_URL"] = f"http://127.0.0.1:{port}"
        env.pop("NEMO_TAVILY_ENABLED", None)
        if enabled:
            env["NEMO_TAVILY_ENABLED"] = "true"
        command = [str(nemo), "services", "run", "--services", "entities,secrets,tavily", "--port", str(port)]
        log_path = Path(tmp) / "platform.log"
        with log_path.open("wb") as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + READY_TIMEOUT_SECONDS
                while True:
                    if proc.poll() is not None:
                        raise AssertionError(f"platform exited with {proc.returncode}:\n{log_path.read_text()[-4000:]}")
                    try:
                        if httpx.get(f"http://127.0.0.1:{port}/health/ready", timeout=2).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() > deadline:
                        raise AssertionError(
                            f"platform not ready in {READY_TIMEOUT_SECONDS}s:\n{log_path.read_text()[-4000:]}"
                        )
                    time.sleep(1)
                return httpx.get(f"http://127.0.0.1:{port}{STATUS_PATH}", timeout=10).status_code
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()


def check_live_platform() -> None:
    off = _status_code_on_platform(enabled=False)
    assert off == 404, f"proxy disabled: expected 404 from {STATUS_PATH}, got {off}"
    on = _status_code_on_platform(enabled=True)
    assert on == 200, f"proxy enabled: expected 200 from {STATUS_PATH}, got {on}"
    print("ok: a live platform serves the proxy only when NEMO_TAVILY_ENABLED=true")


def main() -> None:
    check_discovery()
    check_routes_and_authz()
    check_live_platform()


if __name__ == "__main__":
    main()
