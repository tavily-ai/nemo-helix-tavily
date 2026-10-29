# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route tests for the Tavily MCP proxy service."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nemo_helix_plugin.authz import CallerKind, get_path_rules, get_path_scope
from nemo_helix_plugin.authz_discovery import _derive_service_contribution
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.config import Configuration

from nemo_tavily_plugin import service
from nemo_tavily_plugin.config import TavilyConfig
from nemo_tavily_plugin.service import (
    TavilyService,
    get_agent_base_url,
    get_tavily_config,
    get_upstream_client,
    tavily_api_key,
)

MCP = "/v2/workspaces/team-a/mcp"
STATUS = "/v2/workspaces/team-a/status"
UPSTREAM = "https://mcp.example/mcp"
IN_CLUSTER = "http://nhx-api.nemo.svc:8080"


@pytest.fixture(autouse=True)
def enabled() -> Iterator[None]:
    """Turn the proxy on; tests of the disabled plugin override this again."""
    Configuration.set_override(TavilyConfig(enabled=True))
    yield
    Configuration.clear_override(TavilyConfig)


def _make_app(handler: Callable[[httpx.Request], httpx.Response]) -> FastAPI:
    app = FastAPI()
    for spec in TavilyService().get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app.dependency_overrides[tavily_api_key] = lambda: "tvly-workspace-key"
    app.dependency_overrides[get_upstream_client] = lambda: upstream
    app.dependency_overrides[get_tavily_config] = lambda: TavilyConfig(enabled=True, mcp_url=UPSTREAM)
    app.dependency_overrides[get_agent_base_url] = lambda: IN_CLUSTER
    return app


def _never_called(request: httpx.Request) -> httpx.Response:
    raise AssertionError("upstream must not be called")


def _tools_list_response() -> httpx.Response:
    tools = [{"name": name} for name in ("tavily_search", "tavily_extract", "tavily_crawl", "tavily_research")]
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"tools": tools}})


class TestRoutes:
    def test_tools_list_is_filtered_and_request_is_rewritten(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return _tools_list_response()

        client = TestClient(_make_app(handler))
        response = client.post(
            MCP,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Authorization": "Bearer helix-token", "Mcp-Session-Id": "s-1"},
            params={"tavilyApiKey": "tvly-someone-else"},
        )

        assert response.status_code == 200
        assert [tool["name"] for tool in response.json()["result"]["tools"]] == ["tavily_search", "tavily_extract"]
        upstream_request = seen[0]
        assert str(upstream_request.url) == UPSTREAM
        assert upstream_request.headers["authorization"] == "Bearer tvly-workspace-key"
        assert upstream_request.headers["x-client-name"] == "nemo-helix"
        assert upstream_request.headers["mcp-session-id"] == "s-1"

    def test_allowed_tool_call_is_forwarded(self) -> None:
        result = {"jsonrpc": "2.0", "id": 7, "result": {"content": [{"type": "text", "text": "ok"}]}}

        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["params"]["name"] == "tavily_search"
            return httpx.Response(200, json=result)

        client = TestClient(_make_app(handler))
        response = client.post(
            MCP,
            json={"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "tavily_search"}},
        )

        assert response.json() == result

    def test_disallowed_tool_call_never_reaches_tavily(self) -> None:
        client = TestClient(_make_app(_never_called))
        response = client.post(
            MCP,
            json={"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "tavily_crawl"}},
        )

        assert response.status_code == 200
        assert response.json()["error"]["code"] == -32602

    def test_get_and_delete_are_405_without_reaching_tavily(self) -> None:
        client = TestClient(_make_app(_never_called))

        assert client.get(MCP).status_code == 405
        assert client.delete(MCP).status_code == 405

    def test_upstream_failure_is_502(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        response = TestClient(_make_app(handler)).post(MCP, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})

        assert response.status_code == 502


class TestEnabled:
    @pytest.fixture
    def loaded(self) -> Iterator[None]:
        """Load TavilyConfig the way the platform does, not from the test override."""
        Configuration.clear_override(TavilyConfig)
        Configuration.clear_cache()
        yield
        Configuration.clear_cache()

    @pytest.mark.usefixtures("loaded")
    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEMO_TAVILY_ENABLED", raising=False)

        assert TavilyConfig.get().enabled is False

    @pytest.mark.usefixtures("loaded")
    def test_enabled_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_TAVILY_ENABLED", "true")

        assert TavilyConfig.get().enabled is True

    def test_disabled_plugin_mounts_nothing(self) -> None:
        Configuration.set_override(TavilyConfig(enabled=False))

        app = _make_app(_never_called)

        assert TavilyService().get_routers() == []
        assert TestClient(app).post(MCP, json={"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 404
        assert TestClient(app).get(STATUS).status_code == 404


class TestStatus:
    def test_reports_the_agent_facing_mcp_url(self) -> None:
        response = TestClient(_make_app(_never_called)).get(STATUS)

        assert response.status_code == 200
        assert response.json() == {
            "workspace": "team-a",
            "mcp_url": f"{IN_CLUSTER}/apis/tavily/v2/workspaces/team-a/mcp",
        }

    def test_agent_base_url_prefers_the_in_cluster_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEMO_INTERNAL_BASE_URL", raising=False)
        monkeypatch.setenv("NHX_INTERNAL_BASE_URL", f"{IN_CLUSTER}/")
        monkeypatch.setattr(service, "get_platform_config", lambda: SimpleNamespace(base_url="http://127.0.0.1:8080"))

        assert get_agent_base_url() == IN_CLUSTER

    def test_agent_base_url_falls_back_to_the_platform(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEMO_INTERNAL_BASE_URL", raising=False)
        monkeypatch.delenv("NHX_INTERNAL_BASE_URL", raising=False)
        monkeypatch.setattr(service, "get_platform_config", lambda: SimpleNamespace(base_url="http://127.0.0.1:8080"))

        assert get_agent_base_url() == "http://127.0.0.1:8080"


class TestAuthz:
    def _routes(self) -> dict[str, APIRoute]:
        routes = [route for spec in TavilyService().get_routers() for route in spec.router.routes]
        assert all(isinstance(route, APIRoute) for route in routes)
        return {route.path: route for route in routes if isinstance(route, APIRoute)}

    def test_mcp_route_admits_agents_and_requires_invoke_permission(self) -> None:
        route = self._routes()["/mcp"]

        (rule,) = get_path_rules(route.endpoint)
        assert route.methods == {"POST"}
        assert set(rule.callers) == {CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL}
        assert [permission.id for permission in rule.permissions] == ["tavily.mcp.invoke"]
        assert get_path_scope(route.endpoint)

    def test_status_route_requires_read_permission(self) -> None:
        route = self._routes()["/status"]

        (rule,) = get_path_rules(route.endpoint)
        assert route.methods == {"GET"}
        assert [permission.id for permission in rule.permissions] == ["tavily.mcp.read"]
        assert get_path_scope(route.endpoint)

    def test_derivation_is_clean_whether_enabled_or_not(self) -> None:
        contribution, problems, _ = _derive_service_contribution(TavilyService())
        assert problems == []
        assert set(contribution.endpoints) == {
            "/apis/tavily/v2/workspaces/{workspace}/mcp",
            "/apis/tavily/v2/workspaces/{workspace}/status",
        }

        Configuration.set_override(TavilyConfig(enabled=False))
        contribution, problems, _ = _derive_service_contribution(TavilyService())
        assert problems == []
        assert contribution.endpoints == {}


def _secrets_client(handler: Callable[[httpx.Request], httpx.Response]) -> AsyncNemoClient:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AsyncNemoClient(base_url="http://nhx.test", http_client=http_client)


@pytest.mark.asyncio
class TestApiKeyResolution:
    async def test_reads_the_workspace_secret(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"name": "tavily-api-key", "workspace": "team-a", "value": "tvly-abc"})

        key = await tavily_api_key("team-a", client=_secrets_client(handler))

        assert key == "tvly-abc"
        assert seen[0].url.path == "/apis/secrets/v2/workspaces/team-a/secrets/tavily-api-key/access"

    async def test_missing_secret_explains_setup(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "not found"})

        with pytest.raises(HTTPException) as raised:
            await tavily_api_key("team-a", client=_secrets_client(handler))

        assert raised.value.status_code == 412
        assert "nemo tavily setup --workspace team-a" in raised.value.detail

    async def test_forbidden_secret_is_403(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"detail": "forbidden"})

        with pytest.raises(HTTPException) as raised:
            await tavily_api_key("team-a", client=_secrets_client(handler))

        assert raised.value.status_code == 403
