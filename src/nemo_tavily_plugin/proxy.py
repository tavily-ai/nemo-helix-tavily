# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP pass-through to Tavily's hosted MCP server.

The proxy speaks just enough MCP to do its job. It forwards every message and
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
from typing import Any

import httpx
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, Response, StreamingResponse

CLIENT_NAME = "nemo-helix"

# Only these request headers reach Tavily. An allowlist keeps the caller's Helix
# credentials, principal headers, and cookies on this side of the proxy.
FORWARDED_REQUEST_HEADERS = frozenset({"accept", "content-type", "mcp-protocol-version", "mcp-session-id"})
FORWARDED_RESPONSE_HEADERS = frozenset({"cache-control", "content-type", "mcp-protocol-version", "mcp-session-id"})

# JSON-RPC error codes. MCP reports unknown tools as "Invalid params".
INVALID_REQUEST = -32600
UNKNOWN_TOOL = -32602


def upstream_headers(incoming: Mapping[str, str], *, api_key: str) -> dict[str, str]:
    """Build the headers sent to Tavily from the caller's request headers."""
    headers = {name: value for name, value in incoming.items() if name.lower() in FORWARDED_REQUEST_HEADERS}
    headers["Authorization"] = f"Bearer {api_key}"
    headers["X-Client-Name"] = CLIENT_NAME
    return headers


def downstream_headers(upstream: Mapping[str, str]) -> dict[str, str]:
    """Pick the upstream response headers the caller needs."""
    return {name: value for name, value in upstream.items() if name.lower() in FORWARDED_RESPONSE_HEADERS}


def parse_message(body: bytes) -> Any:
    """Return the decoded JSON-RPC body, or ``None`` when it is empty or not JSON."""
    try:
        return json.loads(body) if body else None
    except ValueError:
        return None


def is_tools_list(message: Any) -> bool:
    return isinstance(message, dict) and message.get("method") == "tools/list"


def refusal(message: Any, allowed: Collection[str]) -> JSONResponse | None:
    """Return the error for a message that must not reach Tavily, or ``None`` to forward it.

    Batches are refused outright (MCP dropped JSON-RPC batching), so a disallowed
    tool cannot slip through inside one.
    """
    if isinstance(message, list):
        return JSONResponse(_error(None, INVALID_REQUEST, "Batched JSON-RPC requests are not supported."), 400)
    if not isinstance(message, dict) or message.get("method") != "tools/call":
        return None
    params = message.get("params")
    name = params.get("name") if isinstance(params, dict) else None
    if name in allowed:
        return None
    available = ", ".join(sorted(allowed)) or "none"
    text = f"Tool {name!r} is not available through NeMo Helix. Available tools: {available}."
    return JSONResponse(_error(message.get("id"), UNKNOWN_TOOL, text))


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def filter_tools(payload: Any, allowed: Collection[str]) -> Any:
    """Drop tools outside *allowed* from a ``tools/list`` result."""
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
        return payload
    tools = [tool for tool in result["tools"] if isinstance(tool, dict) and tool.get("name") in allowed]
    return {**payload, "result": {**result, "tools": tools}}


def filter_tools_list_body(body: bytes, content_type: str, allowed: Collection[str]) -> bytes:
    """Filter a ``tools/list`` response body, either plain JSON or an SSE stream."""
    if "application/json" in content_type:
        payload = parse_message(body)
        return body if payload is None else json.dumps(filter_tools(payload, allowed)).encode()
    if "text/event-stream" not in content_type:
        return body
    lines = []
    for line in body.decode().split("\n"):
        data, ending = line.removesuffix("\r"), "\r" if line.endswith("\r") else ""
        payload = parse_message(data.removeprefix("data:").encode()) if data.startswith("data:") else None
        lines.append(line if payload is None else f"data: {json.dumps(filter_tools(payload, allowed))}{ending}")
    return "\n".join(lines).encode()


async def forward(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    allowed: Collection[str],
    filter_tools_list: bool,
) -> Response:
    """POST one MCP message to Tavily and relay the response.

    A ``tools/list`` response is buffered so it can be filtered. Everything else
    streams through as it arrives, which keeps SSE responses incremental.
    """
    request = client.build_request("POST", url, headers=dict(headers), content=body)
    if filter_tools_list:
        upstream = await client.send(request)
        return Response(
            content=filter_tools_list_body(upstream.content, upstream.headers.get("content-type", ""), allowed),
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
