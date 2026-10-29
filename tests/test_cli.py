# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo tavily`` CLI tests against an in-memory stand-in for the agents and secrets APIs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
import pytest
import typer
from click.testing import Result
from nemo_helix_plugin.client.client import NemoClient
from nemo_tavily_plugin.cli import TavilyCLI
from typer.testing import CliRunner

BASE_URL = "http://nhx.test"
WORKSPACE = "team-a"
PROXY_URL = f"{BASE_URL}/apis/tavily/v2/workspaces/{WORKSPACE}/mcp"
DIRECT_URL = "https://mcp.tavily.com/mcp/"

ClientT = TypeVar("ClientT", bound=NemoClient)

_ROUTE = re.compile(
    r"^/apis/(agents|secrets)/v2/workspaces/(?P<ws>[^/]+)/(?P<kind>agents|deployments|secrets)(?:/(?P<name>[^/]+))?$"
)


def _page(items: list[dict[str, Any]]) -> dict[str, Any]:
    pagination = {
        "page": 1,
        "page_size": 100,
        "current_page_size": len(items),
        "total_pages": 1,
        "total_results": len(items),
    }
    return {"data": items, "pagination": pagination}


@dataclass
class _Platform:
    """Just enough of the agents, secrets and Tavily MCP APIs for the CLI, recording each request."""

    agents: dict[str, dict[str, Any]] = field(default_factory=dict)
    deployments: dict[str, dict[str, Any]] = field(default_factory=dict)
    secrets: dict[str, str] = field(default_factory=dict)
    tool_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == f"/apis/tavily/v2/workspaces/{WORKSPACE}/mcp":
            return self._mcp(request)
        match = _ROUTE.match(request.url.path)
        assert match, f"unexpected path {request.url.path}"
        kind, name = match["kind"], match["name"]
        store = {"agents": self.agents, "deployments": self.deployments}.get(kind)

        if kind == "secrets":
            return self._secrets(request, name)
        assert store is not None
        if request.method == "GET" and name is None:
            return httpx.Response(200, json=_page(list(store.values())))
        if request.method == "GET":
            return (
                httpx.Response(200, json=store[name]) if name in store else httpx.Response(404, json={"detail": "nf"})
            )
        if request.method == "DELETE":
            store.pop(name, None)
            return httpx.Response(204)
        if request.method == "POST":
            body = json.loads(request.content)
            if kind == "deployments":
                body = {**body, "name": f"{body['agent']}-new", "status": "running"}
            store[body["name"]] = {**body, "workspace": WORKSPACE}
            return httpx.Response(201, json=store[body["name"]])
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    def _secrets(self, request: httpx.Request, name: str | None) -> httpx.Response:
        if request.method == "GET":
            if name not in self.secrets:
                return httpx.Response(404, json={"detail": "nf"})
            return httpx.Response(200, json={"name": name, "workspace": WORKSPACE})
        body = json.loads(request.content)
        self.secrets[name or body["name"]] = body["value"]
        return httpx.Response(
            201 if request.method == "POST" else 200, json={"name": name or body["name"], "workspace": WORKSPACE}
        )

    def _mcp(self, request: httpx.Request) -> httpx.Response:
        message = json.loads(request.content)
        if message["method"] == "notifications/initialized":
            return httpx.Response(202)
        result: dict[str, Any] = {}
        if message["method"] == "tools/call":
            params = message["params"]
            self.tool_calls.append((params["name"], params["arguments"]))
            result = {"content": [], "structuredContent": self.tool_results[params["name"]]}
        body = f"data: {json.dumps({'jsonrpc': '2.0', 'id': message['id'], 'result': result})}\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream", "mcp-session-id": "s1"})

    def calls(self) -> list[tuple[str, str]]:
        return [(r.method, r.url.path.rsplit("/v2/workspaces/team-a/", 1)[1]) for r in self.requests]


@dataclass
class _State:
    platform: _Platform
    output_format: str = "json"

    def typed_client(self, client_cls: type[ClientT], timeout: float = 60.0) -> ClientT:
        http_client = httpx.Client(transport=httpx.MockTransport(self.platform))
        return client_cls.from_client(NemoClient(base_url=BASE_URL, http_client=http_client))

    def get_workspace(self) -> str | None:
        return WORKSPACE

    def get_base_url(self, default: str | None = None) -> str | None:
        return BASE_URL

    def get_output_format(self, override: str | None = None, *, apply_non_tty_default: bool = True) -> str:
        return override or self.output_format

    def get_no_truncate(self, override: bool | None = None) -> bool:
        return True

    def get_timestamp_format(self, override: str | None = None) -> str:
        return "iso8601"


def _agent(name: str = "researcher", **config: Any) -> dict[str, Any]:
    return {
        "name": name,
        "workspace": WORKSPACE,
        "description": "Research assistant",
        "config_format": "nemo-agents-spec-v1",
        "config": {"config_format": "nemo-agents-spec-v1", "name": name, **config},
    }


def _deployment(agent: str = "researcher", mode: str = "subprocess", status: str = "running") -> dict[str, Any]:
    return {"name": f"{agent}-old", "workspace": WORKSPACE, "agent": agent, "status": status, "deployment_mode": mode}


@pytest.fixture
def platform() -> _Platform:
    return _Platform()


@pytest.fixture
def app() -> typer.Typer:
    return TavilyCLI().get_cli()


def _invoke(app: typer.Typer, platform: _Platform, *args: str, output_format: str = "json", **kwargs: Any) -> Result:
    return CliRunner().invoke(app, list(args), obj=_State(platform, output_format), **kwargs)


class TestSetup:
    def test_creates_secret(self, app: typer.Typer, platform: _Platform) -> None:
        result = _invoke(app, platform, "setup", "--api-key", "tvly-abc")

        assert result.exit_code == 0, result.output
        assert platform.secrets == {"tavily-api-key": "tvly-abc"}
        assert platform.calls()[-1] == ("POST", "secrets")

    def test_updates_existing_secret(self, app: typer.Typer, platform: _Platform) -> None:
        platform.secrets["tavily-api-key"] = "tvly-old"

        result = _invoke(app, platform, "setup", "--api-key", "tvly-new")

        assert result.exit_code == 0, result.output
        assert platform.secrets["tavily-api-key"] == "tvly-new"
        assert platform.calls()[-1] == ("PATCH", "secrets/tavily-api-key")

    def test_prompts_and_ignores_environment(self, app: typer.Typer, platform: _Platform) -> None:
        result = _invoke(app, platform, "setup", env={"TAVILY_API_KEY": "tvly-env"}, input="tvly-typed\n")

        assert result.exit_code == 0, result.output
        assert platform.secrets["tavily-api-key"] == "tvly-typed"

    def test_rejects_non_tavily_key(self, app: typer.Typer, platform: _Platform) -> None:
        result = _invoke(app, platform, "setup", "--api-key", "sk-nope")

        assert result.exit_code == 1
        assert platform.secrets == {}


class TestEnable:
    def test_rebuilds_and_redeploys_with_proxy_server(self, app: typer.Typer, platform: _Platform) -> None:
        platform.secrets["tavily-api-key"] = "tvly-abc"
        platform.agents["researcher"] = _agent()
        platform.deployments["researcher-old"] = _deployment()

        result = _invoke(app, platform, "enable", "researcher", "--yes")

        assert result.exit_code == 0, result.output
        config = platform.agents["researcher"]["config"]
        assert config["mcp"]["servers"]["tavily"] == {"transport": "streamable-http", "url": PROXY_URL}
        assert platform.agents["researcher"]["description"] == "Research assistant"
        assert ("DELETE", "deployments/researcher-old") in platform.calls()
        assert ("DELETE", "agents/researcher") in platform.calls()
        assert platform.deployments["researcher-new"]["deployment_mode"] == "subprocess"
        assert "is running" in result.output

    def test_agent_without_deployment_is_only_re_registered(self, app: typer.Typer, platform: _Platform) -> None:
        platform.secrets["tavily-api-key"] = "tvly-abc"
        platform.agents["researcher"] = _agent()

        result = _invoke(app, platform, "enable", "researcher", "--yes")

        assert result.exit_code == 0, result.output
        assert "tavily" in platform.agents["researcher"]["config"]["mcp"]["servers"]
        assert not any(method == "POST" and path == "deployments" for method, path in platform.calls())

    def test_already_enabled_is_a_no_op(self, app: typer.Typer, platform: _Platform) -> None:
        platform.secrets["tavily-api-key"] = "tvly-abc"
        platform.agents["researcher"] = _agent(
            mcp={"servers": {"tavily": {"transport": "streamable-http", "url": PROXY_URL}}}
        )

        result = _invoke(app, platform, "enable", "researcher", "--yes")

        assert result.exit_code == 0, result.output
        assert "already enabled" in result.output
        assert all(method == "GET" for method, _ in platform.calls())

    def test_replaces_direct_server_with_proxy(self, app: typer.Typer, platform: _Platform) -> None:
        platform.secrets["tavily-api-key"] = "tvly-abc"
        platform.agents["researcher"] = _agent(mcp={"servers": {"tavily": {"transport": "http", "url": DIRECT_URL}}})

        result = _invoke(app, platform, "enable", "researcher", "--yes")

        assert result.exit_code == 0, result.output
        assert f"Replacing 'researcher''s existing 'tavily' MCP server ({DIRECT_URL})" in result.output
        assert platform.agents["researcher"]["config"]["mcp"]["servers"]["tavily"]["url"] == PROXY_URL

    def test_requires_api_key(self, app: typer.Typer, platform: _Platform) -> None:
        platform.agents["researcher"] = _agent()

        result = _invoke(app, platform, "enable", "researcher", "--yes")

        assert result.exit_code == 1
        assert "nemo tavily setup" in result.output
        assert all(method == "GET" for method, _ in platform.calls())

    def test_rejects_nat_agents(self, app: typer.Typer, platform: _Platform) -> None:
        platform.secrets["tavily-api-key"] = "tvly-abc"
        platform.agents["nat"] = {**_agent("nat"), "config_format": "nat-workflow-v1", "config": {"workflow": {}}}

        result = _invoke(app, platform, "enable", "nat", "--yes")

        assert result.exit_code == 1
        assert "nemo-agent-toolkit-tavily" in result.output

    def test_container_deployment_needs_explicit_url(self, app: typer.Typer, platform: _Platform) -> None:
        platform.secrets["tavily-api-key"] = "tvly-abc"
        platform.agents["researcher"] = _agent()
        platform.deployments["researcher-old"] = _deployment(mode="docker")

        refused = _invoke(app, platform, "enable", "researcher", "--yes")
        allowed = _invoke(
            app, platform, "enable", "researcher", "--yes", "--mcp-url", "http://host.docker.internal:8080/x"
        )

        assert refused.exit_code == 1
        assert "--mcp-url" in refused.output
        assert allowed.exit_code == 0, allowed.output
        assert (
            platform.agents["researcher"]["config"]["mcp"]["servers"]["tavily"]["url"]
            == "http://host.docker.internal:8080/x"
        )
        assert platform.deployments["researcher-new"]["deployment_mode"] == "docker"

    def test_declining_the_prompt_changes_nothing(self, app: typer.Typer, platform: _Platform) -> None:
        platform.secrets["tavily-api-key"] = "tvly-abc"
        platform.agents["researcher"] = _agent()

        result = _invoke(app, platform, "enable", "researcher", input="n\n")

        assert result.exit_code != 0
        assert all(method == "GET" for method, _ in platform.calls())


class TestDisable:
    def test_removes_server_and_redeploys(self, app: typer.Typer, platform: _Platform) -> None:
        servers = {
            "tavily": {"transport": "streamable-http", "url": PROXY_URL},
            "github": {"transport": "stdio", "url": "gh"},
        }
        platform.agents["researcher"] = _agent(mcp={"servers": servers})
        platform.deployments["researcher-old"] = _deployment()

        result = _invoke(app, platform, "disable", "researcher", "--yes")

        assert result.exit_code == 0, result.output
        assert platform.agents["researcher"]["config"]["mcp"]["servers"] == {"github": servers["github"]}
        assert "researcher-new" in platform.deployments

    def test_not_enabled_is_a_no_op(self, app: typer.Typer, platform: _Platform) -> None:
        platform.agents["researcher"] = _agent()

        result = _invoke(app, platform, "disable", "researcher", "--yes")

        assert result.exit_code == 0, result.output
        assert "not enabled" in result.output

    def test_direct_server_is_left_alone(self, app: typer.Typer, platform: _Platform) -> None:
        direct = {"tavily": {"transport": "http", "url": DIRECT_URL}}
        platform.agents["researcher"] = _agent(mcp={"servers": direct})

        result = _invoke(app, platform, "disable", "researcher", "--yes")

        assert result.exit_code == 0, result.output
        assert "does not go through this plugin" in result.output
        assert platform.agents["researcher"]["config"]["mcp"]["servers"] == direct
        assert all(method == "GET" for method, _ in platform.calls())


class TestStatus:
    def test_reports_key_and_per_agent_state(self, app: typer.Typer, platform: _Platform) -> None:
        platform.secrets["tavily-api-key"] = "tvly-abc"
        platform.agents["researcher"] = _agent(mcp={"servers": {"tavily": {"url": PROXY_URL}}})
        platform.agents["triage"] = _agent("triage")
        platform.agents["direct"] = _agent("direct", mcp={"servers": {"tavily": {"url": DIRECT_URL}}})

        result = _invoke(app, platform, "status")

        assert result.exit_code == 0, result.output
        printed = json.loads(result.stdout)
        assert printed["api_key_configured"] is True
        assert {row["name"]: row["tavily_mcp"] for row in printed["agents"]} == {
            "researcher": "enabled",
            "triage": "disabled",
            "direct": "disabled",
        }


class TestSearch:
    def test_prints_results_table(self, app: typer.Typer, platform: _Platform) -> None:
        platform.tool_results["tavily_search"] = {
            "answer": None,
            "results": [{"title": "NeMo", "url": "https://nvidia.com/nemo", "score": 0.9, "content": "..."}],
        }

        result = _invoke(app, platform, "search", "nemo helix", "-n", "3", output_format="table")

        assert result.exit_code == 0, result.output
        assert "https://nvidia.com/nemo" in result.output
        assert platform.tool_calls == [("tavily_search", {"query": "nemo helix", "max_results": 3})]

    def test_json_prints_the_full_result(self, app: typer.Typer, platform: _Platform) -> None:
        platform.tool_results["tavily_search"] = {"query": "q", "results": []}

        result = _invoke(app, platform, "search", "q", "--depth", "advanced", "--topic", "news")

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == {"query": "q", "results": []}
        assert platform.tool_calls[0][1] == {"query": "q", "search_depth": "advanced", "topic": "news"}

    def test_unexpected_result_is_printed_as_json(self, app: typer.Typer, platform: _Platform) -> None:
        platform.tool_results["tavily_search"] = {"detail": "something new"}

        result = _invoke(app, platform, "search", "q", output_format="table")

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == {"detail": "something new"}


class TestExtract:
    def test_prints_page_content_and_failures(self, app: typer.Typer, platform: _Platform) -> None:
        platform.tool_results["tavily_extract"] = {
            "results": [{"url": "https://a.test", "title": "A", "raw_content": "Hello from A"}],
            "failed_results": [{"url": "https://b.test", "error": "Failed to fetch url"}],
        }

        result = _invoke(app, platform, "extract", "https://a.test", "https://b.test", output_format="table")

        assert result.exit_code == 0, result.output
        assert "Hello from A" in result.stdout
        assert "Could not extract https://b.test: Failed to fetch url" in result.stderr
        assert platform.tool_calls == [("tavily_extract", {"urls": ["https://a.test", "https://b.test"]})]

    def test_fails_when_no_page_was_extracted(self, app: typer.Typer, platform: _Platform) -> None:
        platform.tool_results["tavily_extract"] = {
            "results": [],
            "failed_results": [{"url": "https://b.test", "error": "Failed to fetch url"}],
        }

        result = _invoke(app, platform, "extract", "https://b.test", output_format="table")

        assert result.exit_code == 1
