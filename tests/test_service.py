# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route tests for the Tavily MCP proxy service."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nemo_helix_plugin.authz import CallerKind, get_path_rules, get_path_scope
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_tavily_plugin.config import TavilyConfig
from nemo_tavily_plugin.service import (
    TavilyService,
    get_tavily_config,
    get_upstream_client,
    tavily_api_key,
)

MCP = "/v2/workspaces/team-a/mcp"
UPSTREAM = "https://mcp.example/mcp"


def _make_app(handler: Callable[[httpx.Request], httpx.Response]) -> FastAPI:
    app = FastAPI()
    for spec in TavilyService().get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app.dependency_overrides[tavily_api_key] = lambda: "tvly-workspace-key"
    app.dependency_overrides[get_upstream_client] = lambda: upstream
    app.dependency_overrides[get_tavily_config] = lambda: TavilyConfig(mcp_url=UPSTREAM)
    return app


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
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("upstream must not be called")

        client = TestClient(_make_app(handler))
        response = client.post(
            MCP,
            json={"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "tavily_crawl"}},
        )

        assert response.status_code == 200
        assert response.json()["error"]["code"] == -32602

    def test_get_and_delete_are_405_without_reaching_tavily(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("upstream must not be called")

        client = TestClient(_make_app(handler))

        assert client.get(MCP).status_code == 405
        assert client.delete(MCP).status_code == 405

    def test_upstream_failure_is_502(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        response = TestClient(_make_app(handler)).post(MCP, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})

        assert response.status_code == 502


class TestAuthz:
    def test_route_admits_agents_and_requires_invoke_permission(self) -> None:
        (route,) = [route for spec in TavilyService().get_routers() for route in spec.router.routes]
        assert isinstance(route, APIRoute)

        (rule,) = get_path_rules(route.endpoint)
        assert route.methods == {"POST"}
        assert set(rule.callers) == {CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL}
        assert [permission.id for permission in rule.permissions] == ["tavily.mcp.invoke"]
        assert get_path_scope(route.endpoint)


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
