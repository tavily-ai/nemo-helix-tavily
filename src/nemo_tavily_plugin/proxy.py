# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP pass-through to Tavily's hosted MCP server.

The proxy speaks just enough MCP to do its job. It forwards every request and
response unchanged, except that it:

- replaces the caller's credentials with the workspace's Tavily API key and an
  ``X-Client-Name`` attribution header;
- filters ``tools/list`` results down to the allowed tools;
- refuses ``tools/call`` for any other tool without contacting Tavily.

Everything else, including tool schemas and descriptions, comes from the hosted
server, so changes there reach Helix agents without a plugin release.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, Response, StreamingResponse

# Only these request headers reach Tavily. An allowlist keeps the caller's Helix
# credentials, principal headers, and cookies on this side of the proxy.
FORWARDED_REQUEST_HEADERS = frozenset(
    {"accept", "content-type", "last-event-id", "mcp-protocol-version", "mcp-session-id"}
)
FORWARDED_RESPONSE_HEADERS = frozenset({"cache-control", "content-type", "mcp-protocol-version", "mcp-session-id"})

# JSON-RPC "Invalid params", which MCP uses for unknown tools.
UNKNOWN_TOOL_ERROR_CODE = -32602
INTERNAL_ERROR_CODE = -32603

try:
    _PLUGIN_VERSION = version("nemo-tavily-plugin")
except PackageNotFoundError:
    _PLUGIN_VERSION = "0.0.0"

USER_AGENT = f"nemo-tavily-plugin/{_PLUGIN_VERSION}"


def upstream_headers(incoming: Mapping[str, str], *, api_key: str, client_name: str) -> dict[str, str]:
    """Build the headers sent to Tavily from the caller's request headers."""
    headers = {name: value for name, value in incoming.items() if name.lower() in FORWARDED_REQUEST_HEADERS}
    headers["Authorization"] = f"Bearer {api_key}"
    headers["X-Client-Name"] = client_name
    headers["User-Agent"] = USER_AGENT
    return headers


def downstream_headers(upstream: Mapping[str, str]) -> dict[str, str]:
    """Pick the upstream response headers the caller needs."""
    return {name: value for name, value in upstream.items() if name.lower() in FORWARDED_RESPONSE_HEADERS}


def parse_messages(body: bytes) -> list[dict[str, Any]]:
    """Return the JSON-RPC messages in *body*, or ``[]`` when it is not JSON-RPC."""
    if not body:
        return []
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    items = payload if isinstance(payload, list) else [payload]
    return [item for item in items if isinstance(item, dict)]


def is_tools_list(messages: list[dict[str, Any]]) -> bool:
    return any(message.get("method") == "tools/list" for message in messages)


def blocked_tool_calls(messages: list[dict[str, Any]], allowed: Collection[str]) -> list[dict[str, Any]]:
    """Return the ``tools/call`` requests in *messages* for tools outside *allowed*."""
    blocked = []
    for message in messages:
        if message.get("method") != "tools/call":
            continue
        params = message.get("params")
        name = params.get("name") if isinstance(params, dict) else None
        if name not in allowed:
            blocked.append(message)
    return blocked


def jsonrpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def blocked_tools_response(
    messages: list[dict[str, Any]],
    blocked: list[dict[str, Any]],
    allowed: Collection[str],
) -> Response:
    """Refuse a request that calls a tool outside *allowed*.

    A batch is refused as a whole, so no tool in it runs. Every request in the
    batch gets an error; notifications (no ``id``) get none, per JSON-RPC.
    """
    blocked_ids = {id(message) for message in blocked}
    allowed_list = ", ".join(sorted(allowed)) or "none"
    errors = []
    for message in messages:
        if "id" not in message:
            continue
        if id(message) in blocked_ids:
            params = message.get("params")
            name = params.get("name") if isinstance(params, dict) else None
            text = f"Tool {name!r} is not available through NeMo Helix. Available tools: {allowed_list}."
        else:
            text = "Request refused: the batch calls a tool that is not available through NeMo Helix."
        errors.append(jsonrpc_error(message["id"], UNKNOWN_TOOL_ERROR_CODE, text))
    if not errors:
        return Response(status_code=202)
    return JSONResponse(content=errors if len(messages) > 1 else errors[0])


def filter_tools_payload(payload: Any, allowed: Collection[str]) -> Any:
    """Drop tools outside *allowed* from a ``tools/list`` result (one message or a batch)."""
    if isinstance(payload, list):
        return [filter_tools_payload(item, allowed) for item in payload]
    if not isinstance(payload, dict):
        return payload
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("tools"), list):
        tools = [tool for tool in result["tools"] if isinstance(tool, dict) and tool.get("name") in allowed]
        payload = {**payload, "result": {**result, "tools": tools}}
    return payload


def filter_tools_list_body(body: bytes, content_type: str, allowed: Collection[str]) -> bytes:
    """Filter a ``tools/list`` response body, either plain JSON or an SSE stream."""
    if "text/event-stream" in content_type:
        return _filter_sse(body, allowed)
    if "application/json" in content_type:
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body
        return json.dumps(filter_tools_payload(payload, allowed)).encode()
    return body


def _filter_sse(body: bytes, allowed: Collection[str]) -> bytes:
    lines = body.decode().split("\n")
    out = []
    for line in lines:
        ending = "\r" if line.endswith("\r") else ""
        stripped = line.removesuffix("\r")
        if stripped.startswith("data:"):
            data = stripped.removeprefix("data:").removeprefix(" ")
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                out.append(line)
                continue
            out.append(f"data: {json.dumps(filter_tools_payload(payload, allowed))}{ending}")
        else:
            out.append(line)
    return "\n".join(out).encode()


async def forward(
    client: httpx.AsyncClient,
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    allowed: Collection[str],
    filter_tools_list: bool,
) -> Response:
    """Send one MCP request to Tavily and relay the response.

    A ``tools/list`` response is buffered so it can be filtered. Everything else
    streams through as it arrives, which keeps SSE responses incremental.
    """
    request = client.build_request(method, url, headers=dict(headers), content=body or None)
    if filter_tools_list:
        upstream = await client.send(request)
        content_type = upstream.headers.get("content-type", "")
        return Response(
            content=filter_tools_list_body(upstream.content, content_type, allowed),
            status_code=upstream.status_code,
            headers=downstream_headers(upstream.headers),
        )

    upstream = await client.send(request, stream=True)
    return StreamingResponse(
        upstream.aiter_bytes(),
        status_code=upstream.status_code,
        headers=downstream_headers(upstream.headers),
        background=BackgroundTask(upstream.aclose),
    )
