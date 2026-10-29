# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MCP pass-through: header handling, tool filtering, refusals."""

from __future__ import annotations

import json

import httpx
import pytest
from starlette.responses import Response, StreamingResponse

from nemo_tavily_plugin.proxy import (
    INVALID_REQUEST,
    UNKNOWN_TOOL,
    downstream_headers,
    filter_tools_list_body,
    forward,
    is_tools_list,
    parse_message,
    refusal,
    upstream_headers,
)

ALLOWED = ["tavily_search", "tavily_extract"]
TOOLS = [
    {"name": name, "inputSchema": {}} for name in ("tavily_search", "tavily_extract", "tavily_crawl", "tavily_map")
]


def _tools_list_result() -> dict:
    return {"jsonrpc": "2.0", "id": 1, "result": {"tools": TOOLS, "nextCursor": None}}


def _call(name: str, request_id: int = 7) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": {"name": name, "arguments": {}}}


def _json(response: Response) -> dict:
    return json.loads(bytes(response.body))


class TestHeaders:
    def test_replaces_caller_credentials_with_tavily_key_and_attribution(self) -> None:
        incoming = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Mcp-Session-Id": "abc",
            "Mcp-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer helix-user-token",
            "X-NHX-Principal-Id": "user:alice",
            "Cookie": "session=1",
        }

        headers = upstream_headers(incoming, api_key="tvly-key")

        assert headers == {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Mcp-Session-Id": "abc",
            "Mcp-Protocol-Version": "2025-06-18",
            "Authorization": "Bearer tvly-key",
            "X-Client-Name": "nemo-helix",
        }

    def test_downstream_keeps_session_and_content_type_only(self) -> None:
        upstream = {"Content-Type": "text/event-stream", "Mcp-Session-Id": "abc", "Set-Cookie": "x=1", "Server": "uv"}

        assert downstream_headers(upstream) == {"Content-Type": "text/event-stream", "Mcp-Session-Id": "abc"}


class TestRefusal:
    def test_allowed_tools_and_other_methods_are_forwarded(self) -> None:
        assert refusal(_call("tavily_search"), ALLOWED) is None
        assert refusal({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, ALLOWED) is None
        assert refusal(parse_message(b"not json"), ALLOWED) is None

    def test_other_tool_gets_unknown_tool_error(self) -> None:
        response = refusal(_call("tavily_crawl"), ALLOWED)

        assert response is not None
        body = _json(response)
        assert body["id"] == 7
        assert body["error"]["code"] == UNKNOWN_TOOL
        assert "'tavily_crawl'" in body["error"]["message"]
        assert "tavily_extract, tavily_search" in body["error"]["message"]

    def test_call_without_tool_name_is_refused(self) -> None:
        assert refusal({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {}}, ALLOWED) is not None

    def test_batches_are_refused(self) -> None:
        response = refusal([_call("tavily_search", 1), _call("tavily_crawl", 2)], ALLOWED)

        assert response is not None
        assert response.status_code == 400
        assert _json(response)["error"]["code"] == INVALID_REQUEST


class TestToolsListFiltering:
    def test_detects_tools_list(self) -> None:
        assert is_tools_list({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert not is_tools_list(_call("tavily_search"))
        assert not is_tools_list(None)

    def test_filters_json_response(self) -> None:
        body = json.dumps(_tools_list_result()).encode()

        filtered = json.loads(filter_tools_list_body(body, "application/json", ALLOWED))

        assert [tool["name"] for tool in filtered["result"]["tools"]] == ALLOWED
        assert filtered["result"]["nextCursor"] is None

    def test_filters_sse_response_and_keeps_framing(self) -> None:
        body = f"event: message\r\ndata: {json.dumps(_tools_list_result())}\r\n\r\n".encode()

        filtered = filter_tools_list_body(body, "text/event-stream", ALLOWED).decode()

        assert filtered.startswith("event: message\r\ndata: ")
        assert filtered.endswith("\r\n\r\n")
        payload = json.loads(filtered.split("\r\n")[1].removeprefix("data: "))
        assert [tool["name"] for tool in payload["result"]["tools"]] == ALLOWED

    def test_leaves_other_responses_alone(self) -> None:
        body = b'{"jsonrpc": "2.0", "id": 1, "result": {"content": []}}'

        assert json.loads(filter_tools_list_body(body, "application/json", ALLOWED)) == json.loads(body)
        assert filter_tools_list_body(b"plain", "text/plain", ALLOWED) == b"plain"


@pytest.mark.asyncio
class TestForward:
    async def test_tools_list_is_filtered_and_session_header_returned(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=_tools_list_result(), headers={"mcp-session-id": "s-1"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await forward(
                client,
                url="https://mcp.example/mcp",
                headers={"Authorization": "Bearer tvly-key"},
                body=b'{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}',
                allowed=ALLOWED,
                filter_tools_list=True,
            )

        assert (seen[0].method, seen[0].headers["authorization"]) == ("POST", "Bearer tvly-key")
        assert response.headers["mcp-session-id"] == "s-1"
        assert [tool["name"] for tool in _json(response)["result"]["tools"]] == ALLOWED

    async def test_other_messages_stream_through_unchanged(self) -> None:
        payload = b'data: {"jsonrpc": "2.0", "id": 7, "result": {"content": [{"type": "text", "text": "hi"}]}}\n\n'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await forward(
                client,
                url="https://mcp.example/mcp",
                headers={},
                body=json.dumps(_call("tavily_search")).encode(),
                allowed=ALLOWED,
                filter_tools_list=False,
            )
            assert isinstance(response, StreamingResponse)
            chunks = [chunk async for chunk in response.body_iterator]

        assert response.headers["content-type"] == "text/event-stream"
        assert b"".join(chunk.encode() if isinstance(chunk, str) else bytes(chunk) for chunk in chunks) == payload
