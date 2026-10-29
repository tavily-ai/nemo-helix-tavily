# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``call_tool`` against a stand-in for the workspace proxy and Tavily's stateful MCP server."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from nemo_helix_plugin.client.client import NemoClient

from nemo_tavily_plugin.tool_call import ToolCallError, call_tool

BASE_URL = "http://nhx.test"
SESSION_ID = "session-1"


def _sse(message: dict[str, Any], **headers: str) -> httpx.Response:
    body = f"event: message\ndata: {json.dumps(message)}\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream", **headers})


class _Server:
    """Answers like Tavily's hosted MCP server behind the proxy, recording each request."""

    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.url.path == "/apis/tavily/v2/workspaces/team-a/mcp"
        message = json.loads(request.content)
        if message["method"] == "initialize":
            return _sse({"jsonrpc": "2.0", "id": message["id"], "result": {}}, **{"mcp-session-id": SESSION_ID})
        assert request.headers["mcp-session-id"] == SESSION_ID
        if message["method"] == "notifications/initialized":
            return httpx.Response(202)
        return _sse({"jsonrpc": "2.0", "id": message["id"], **self.reply})


def _call(server: _Server, **arguments: Any) -> Any:
    client = NemoClient(base_url=BASE_URL, http_client=httpx.Client(transport=httpx.MockTransport(server)))
    return call_tool(client, workspace="team-a", name="tavily_search", arguments=arguments)


def test_opens_a_session_then_calls_the_tool() -> None:
    server = _Server({"result": {"content": [{"type": "text", "text": "{}"}], "structuredContent": {"results": []}}})

    result = _call(server, query="nemo")

    assert result == {"results": []}
    methods = [json.loads(request.content)["method"] for request in server.requests]
    assert methods == ["initialize", "notifications/initialized", "tools/call"]
    assert json.loads(server.requests[-1].content)["params"] == {
        "name": "tavily_search",
        "arguments": {"query": "nemo"},
    }


def test_falls_back_to_json_text() -> None:
    server = _Server({"result": {"content": [{"type": "text", "text": '{"answer": "42"}'}]}})

    assert _call(server, query="q") == {"answer": "42"}


def test_returns_plain_text_as_is() -> None:
    server = _Server({"result": {"content": [{"type": "text", "text": "not json"}]}})

    assert _call(server, query="q") == "not json"


def test_raises_tool_errors() -> None:
    server = _Server({"result": {"isError": True, "content": [{"type": "text", "text": "quota exceeded"}]}})

    with pytest.raises(ToolCallError, match="quota exceeded"):
        _call(server, query="q")


def test_raises_json_rpc_errors() -> None:
    server = _Server({"error": {"code": -32602, "message": "Tool 'tavily_crawl' is not available"}})

    with pytest.raises(ToolCallError, match="not available"):
        _call(server, query="q")
