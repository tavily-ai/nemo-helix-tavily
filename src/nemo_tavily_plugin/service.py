# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tavily plugin service, registered under ``nemo.services``.

Serves a workspace-scoped MCP endpoint that agents attach as an MCP server::

    /apis/tavily/v2/workspaces/{workspace}/mcp

Each request is forwarded to Tavily's hosted MCP server with the workspace's
Tavily API key, read from Helix Secrets. The key never reaches the agent.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nemo_helix_plugin.service import NemoService, RouterSpec
from nemo_tavily_plugin.authz import TavilyMcpPerms, scope
from nemo_tavily_plugin.config import TavilyConfig
from nemo_tavily_plugin.proxy import (
    blocked_tool_calls,
    blocked_tools_response,
    forward,
    is_tools_list,
    parse_messages,
    upstream_headers,
)

logger = logging.getLogger(__name__)

# Agents reach the proxy as themselves when auth is off, or as ``service:agents``
# through a deployment's auth-proxy sidecar when it is on, so admit both kinds.
_MCP_CALLERS = [CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL]

_upstream_client: httpx.AsyncClient | None = None


def get_tavily_config() -> TavilyConfig:
    return TavilyConfig.get()


def get_upstream_client(config: TavilyConfig = Depends(get_tavily_config)) -> httpx.AsyncClient:
    """Return the shared HTTP client for Tavily, creating it on first use."""
    global _upstream_client
    if _upstream_client is None:
        _upstream_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=config.timeout_seconds))
    return _upstream_client


async def tavily_api_key(
    workspace: str,
    client: AsyncNemoClient = Depends(get_nemo_client),
    config: TavilyConfig = Depends(get_tavily_config),
) -> str:
    """Read the workspace's Tavily API key as the calling principal."""
    secrets_client = AsyncSecretsClient.from_client(client)
    try:
        response = await secrets_client.access_secret(workspace=workspace, name=config.secret_name)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=412,
            detail=(
                f"No Tavily API key in workspace {workspace!r}. "
                f"Run `nemo tavily setup --workspace {workspace}` to add one."
            ),
        ) from exc
    except PermissionDeniedError as exc:
        raise HTTPException(
            status_code=403,
            detail=f"Not allowed to read secret {config.secret_name!r} in workspace {workspace!r}.",
        ) from exc
    return response.data().value


class TavilyService(NemoService):
    """Workspace-scoped MCP proxy to Tavily's hosted MCP server."""

    name: ClassVar[str] = "tavily"
    dependencies: ClassVar[list[str]] = ["secrets"]

    def get_routers(self) -> list[RouterSpec]:
        return [
            RouterSpec(
                _build_mcp_router(),
                tag="Tavily",
                description="MCP endpoint for Tavily web search and extraction.",
                prefix="/v2/workspaces/{workspace}",
            )
        ]

    async def on_shutdown(self) -> None:
        global _upstream_client
        if _upstream_client is not None:
            await _upstream_client.aclose()
            _upstream_client = None


async def _proxy(
    request: Request,
    *,
    api_key: str,
    upstream: httpx.AsyncClient,
    config: TavilyConfig,
) -> Response:
    body = await request.body()
    messages = parse_messages(body)
    blocked = blocked_tool_calls(messages, config.allowed_tools)
    if blocked:
        return blocked_tools_response(messages, blocked, config.allowed_tools)

    headers = upstream_headers(request.headers, api_key=api_key, client_name=config.client_name)
    try:
        return await forward(
            upstream,
            method=request.method,
            url=config.mcp_url,
            headers=headers,
            body=body,
            allowed=config.allowed_tools,
            filter_tools_list=is_tools_list(messages),
        )
    except httpx.HTTPError as exc:
        logger.warning("Tavily MCP request failed: %s", exc)
        raise HTTPException(status_code=502, detail="Could not reach the Tavily MCP server.") from exc


def _build_mcp_router() -> APIRouter:
    router = APIRouter()

    @router.post("/mcp", response_model=None)
    @scope.write
    @path_rule(callers=_MCP_CALLERS, permissions=[TavilyMcpPerms.INVOKE])
    async def post_mcp(
        request: Request,
        api_key: str = Depends(tavily_api_key),
        upstream: httpx.AsyncClient = Depends(get_upstream_client),
        config: TavilyConfig = Depends(get_tavily_config),
    ) -> Response:
        """Send an MCP message (initialize, tools/list, tools/call, ...) to Tavily."""
        return await _proxy(request, api_key=api_key, upstream=upstream, config=config)

    @router.get("/mcp", response_model=None)
    @scope.read
    @path_rule(callers=_MCP_CALLERS, permissions=[TavilyMcpPerms.INVOKE])
    async def get_mcp(
        request: Request,
        api_key: str = Depends(tavily_api_key),
        upstream: httpx.AsyncClient = Depends(get_upstream_client),
        config: TavilyConfig = Depends(get_tavily_config),
    ) -> Response:
        """Open the MCP server-to-client event stream, when Tavily offers one."""
        return await _proxy(request, api_key=api_key, upstream=upstream, config=config)

    @router.delete("/mcp", response_model=None)
    @scope.write
    @path_rule(callers=_MCP_CALLERS, permissions=[TavilyMcpPerms.INVOKE])
    async def delete_mcp(
        request: Request,
        api_key: str = Depends(tavily_api_key),
        upstream: httpx.AsyncClient = Depends(get_upstream_client),
        config: TavilyConfig = Depends(get_tavily_config),
    ) -> Response:
        """End an MCP session."""
        return await _proxy(request, api_key=api_key, upstream=upstream, config=config)

    return router
