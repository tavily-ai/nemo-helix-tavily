# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tavily plugin service, registered under ``nemo.services``.

Serves a workspace-scoped MCP endpoint that agents attach as an MCP server::

    POST /apis/tavily/v2/workspaces/{workspace}/mcp
    GET  /apis/tavily/v2/workspaces/{workspace}/status

Nothing is mounted unless ``TavilyConfig.enabled`` is set, so an installed but
unconfigured plugin serves no routes and never contacts Tavily.

Each message is forwarded to Tavily's hosted MCP server with the workspace's
Tavily API key, read from Helix Secrets. The key never reaches the agent. Only
POST is served: Tavily offers no server-to-client stream (GET) or session
teardown (DELETE), and FastAPI's 405 for those is what MCP clients expect.
"""

from __future__ import annotations

import logging
import os
from typing import ClassVar

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.config import get_platform_config
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nemo_helix_plugin.service import NemoService, RouterSpec
from pydantic import BaseModel

from nemo_tavily_plugin.authz import TavilyMcpPerms, scope
from nemo_tavily_plugin.config import ENABLE_HINT, SECRET_NAME, TavilyConfig
from nemo_tavily_plugin.proxy import forward, is_tools_list, parse_message, proxy_url, refusal, upstream_headers

logger = logging.getLogger(__name__)

_upstream_client: httpx.AsyncClient | None = None


def get_tavily_config() -> TavilyConfig:
    return TavilyConfig.get()


def get_agent_base_url() -> str:
    """Return the platform URL deployed agents can reach.

    On Kubernetes that is the in-cluster API Service, which the Helm chart sets as
    ``NHX_INTERNAL_BASE_URL``; the platform's own base URL is loopback in the API
    pod. Elsewhere agents share the platform's network, so its base URL works.
    """
    internal = os.environ.get("NEMO_INTERNAL_BASE_URL") or os.environ.get("NHX_INTERNAL_BASE_URL")
    return (internal or get_platform_config().base_url).rstrip("/")


class TavilyStatus(BaseModel):
    workspace: str
    mcp_url: str


def get_upstream_client(config: TavilyConfig = Depends(get_tavily_config)) -> httpx.AsyncClient:
    """Return the shared HTTP client for Tavily, creating it on first use."""
    global _upstream_client
    if _upstream_client is None:
        _upstream_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=config.timeout_seconds))
    return _upstream_client


async def tavily_api_key(workspace: str, client: AsyncNemoClient = Depends(get_nemo_client)) -> str:
    """Read the workspace's Tavily API key as the calling principal."""
    try:
        response = await AsyncSecretsClient.from_client(client).access_secret(workspace=workspace, name=SECRET_NAME)
    except NotFoundError as exc:
        detail = f"No Tavily API key in workspace {workspace!r}. Run `nemo tavily setup --workspace {workspace}`."
        raise HTTPException(status_code=412, detail=detail) from exc
    except PermissionDeniedError as exc:
        detail = f"Not allowed to read secret {SECRET_NAME!r} in workspace {workspace!r}."
        raise HTTPException(status_code=403, detail=detail) from exc
    return response.data().value


class TavilyService(NemoService):
    """Workspace-scoped MCP proxy to Tavily's hosted MCP server."""

    name: ClassVar[str] = "tavily"
    dependencies: ClassVar[list[str]] = ["secrets"]

    def get_routers(self) -> list[RouterSpec]:
        if not get_tavily_config().enabled:
            return []
        return [
            RouterSpec(
                _build_mcp_router(),
                tag="Tavily",
                description="MCP endpoint for Tavily web search and extraction.",
                prefix="/v2/workspaces/{workspace}",
            )
        ]

    async def on_startup(self) -> None:
        if not get_tavily_config().enabled:
            logger.info("Tavily MCP proxy is installed but disabled. %s", ENABLE_HINT)

    async def on_shutdown(self) -> None:
        global _upstream_client
        if _upstream_client is not None:
            await _upstream_client.aclose()
            _upstream_client = None


def _build_mcp_router() -> APIRouter:
    router = APIRouter()

    @router.post("/mcp", response_model=None)
    @scope.write
    # Agents call as themselves when auth is off, or as ``service:agents`` through
    # a deployment's auth-proxy sidecar when it is on.
    @path_rule(callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL], permissions=[TavilyMcpPerms.INVOKE])
    async def post_mcp(
        request: Request,
        api_key: str = Depends(tavily_api_key),
        upstream: httpx.AsyncClient = Depends(get_upstream_client),
        config: TavilyConfig = Depends(get_tavily_config),
    ) -> Response:
        """Send an MCP message (initialize, tools/list, tools/call, ...) to Tavily."""
        body = await request.body()
        message = parse_message(body)
        if (refused := refusal(message, config.allowed_tools)) is not None:
            return refused
        try:
            return await forward(
                upstream,
                url=config.mcp_url,
                headers=upstream_headers(request.headers, api_key=api_key),
                body=body,
                allowed=config.allowed_tools,
                filter_tools_list=is_tools_list(message),
            )
        except httpx.HTTPError as exc:
            logger.warning("Tavily MCP request failed: %s", exc)
            raise HTTPException(status_code=502, detail="Could not reach the Tavily MCP server.") from exc

    @router.get("/status")
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL], permissions=[TavilyMcpPerms.READ])
    async def get_status(workspace: str, base_url: str = Depends(get_agent_base_url)) -> TavilyStatus:
        """Report that the proxy is enabled, and the MCP URL deployed agents should use for *workspace*."""
        return TavilyStatus(workspace=workspace, mcp_url=proxy_url(base_url, workspace))

    return router
