# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CLI's calls to the workspace proxy: its status, and one Tavily tool at a time.

``proxy_mcp_url`` asks the platform whether the proxy is enabled and which URL
deployed agents reach it at. ``call_tool`` backs ``nemo tavily search|extract``.

The CLI goes through the platform's proxy rather than to Tavily directly, so it
never needs the API key and gets the same tool allowlist and attribution as
agents. Tavily's hosted server is stateful: each call opens an MCP session
(``initialize``, then ``notifications/initialized``) before ``tools/call``.
Sessions are not closed afterwards because Tavily answers ``DELETE`` with 405.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.client.types import PreparedRequest
from nemo_tavily_plugin.proxy import parse_message

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "nemo-cli", "version": "1"}

_PATH_TEMPLATE = "/apis/tavily/v2/workspaces/{workspace}/mcp"
_STATUS_PATH_TEMPLATE = "/apis/tavily/v2/workspaces/{workspace}/status"


class ToolCallError(Exception):
    """Tavily, or the proxy, rejected the tool call."""


def proxy_mcp_url(client: NemoClient, *, workspace: str) -> str | None:
    """Return the MCP URL deployed agents should use for *workspace*, or ``None`` when the proxy is off.

    The platform mounts no Tavily routes unless the proxy is enabled, so a 404 here
    means it is disabled, or the plugin is not installed on the platform.
    """
    request: PreparedRequest[None] = PreparedRequest(
        path_template=_STATUS_PATH_TEMPLATE,
        path_params={"workspace": workspace},
        method="GET",
        content=None,
        content_type=None,
        response_type=None,
    )
    try:
        response = client.send(request).http_response
    except NotFoundError:
        return None
    return response.json()["mcp_url"]


def call_tool(client: NemoClient, *, workspace: str, name: str, arguments: dict[str, Any]) -> Any:
    """Call Tavily tool *name* through *workspace*'s proxy and return its result.

    Returns the tool's structured content when it has one, else its text, decoded
    as JSON when possible.
    """
    initialize = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": CLIENT_INFO}
    response = _post(client, workspace, {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": initialize})
    _reply(response, 0)
    session_id = response.headers.get("mcp-session-id")
    if not session_id:
        raise ToolCallError("Tavily did not start an MCP session.")

    _post(client, workspace, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session_id=session_id)
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    result = _reply(_post(client, workspace, call, session_id=session_id), 1)

    text = "\n".join(item["text"] for item in result.get("content") or [] if item.get("type") == "text")
    if result.get("isError"):
        raise ToolCallError(text or f"{name} failed.")
    if isinstance(result.get("structuredContent"), dict):
        return result["structuredContent"]
    try:
        return json.loads(text)
    except ValueError:
        return text


def _post(
    client: NemoClient, workspace: str, message: dict[str, Any], *, session_id: str | None = None
) -> httpx.Response:
    headers = {"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": PROTOCOL_VERSION}
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    request: PreparedRequest[None] = PreparedRequest(
        path_template=_PATH_TEMPLATE,
        path_params={"workspace": workspace},
        method="POST",
        content=json.dumps(message).encode(),
        content_type="application/json",
        response_type=None,
        extra_headers=headers,
    )
    return client.send(request).http_response


def _reply(response: httpx.Response, request_id: int) -> dict[str, Any]:
    """Return the ``result`` of the JSON-RPC reply to *request_id*, from plain JSON or SSE."""
    if "text/event-stream" in response.headers.get("content-type", ""):
        lines = response.text.splitlines()
        messages = [parse_message(line.removeprefix("data:").encode()) for line in lines if line.startswith("data:")]
    else:
        messages = [parse_message(response.content)]
    for message in messages:
        if not isinstance(message, dict) or message.get("id") != request_id:
            continue
        if "error" in message:
            raise ToolCallError(str((message["error"] or {}).get("message") or message["error"]))
        return message.get("result") or {}
    raise ToolCallError(f"No reply from Tavily to MCP request {request_id}.")
