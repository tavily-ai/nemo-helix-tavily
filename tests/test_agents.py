# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the agent config transforms behind enable/disable."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from nemo_helix_plugin.agents.types import AgentDeployment, DeploymentStatus

from nemo_tavily_plugin.agents import (
    mcp_url_for_mode,
    tavily_enabled,
    wait_for_deployment,
    with_tavily,
    without_tavily,
)
from nemo_tavily_plugin.proxy import proxy_url

URL = "http://localhost:8080/apis/tavily/v2/workspaces/team-a/mcp"


def test_proxy_url_is_workspace_scoped() -> None:
    assert proxy_url("http://localhost:8080/", "team-a") == URL


@pytest.mark.parametrize(
    ("url", "mode", "expected"),
    [
        (URL, None, URL),
        (URL, "subprocess", URL),
        ("http://nhx-api.nemo.svc:8080/x", "k8s", "http://nhx-api.nemo.svc:8080/x"),
        (URL, "docker", "http://host.docker.internal:8080/apis/tavily/v2/workspaces/team-a/mcp"),
        ("http://[::1]/x", "docker", "http://host.docker.internal/x"),
        ("http://helix.example:8080/x", "docker", "http://helix.example:8080/x"),
        (URL, "openshell", None),
    ],
)
def test_mcp_url_for_mode(url: str, mode: str | None, expected: str | None) -> None:
    assert mcp_url_for_mode(url, mode) == expected


def test_with_tavily_adds_server_and_keeps_others() -> None:
    config = {"name": "a", "mcp": {"servers": {"github": {"transport": "streamable-http", "url": "https://gh"}}}}

    updated = with_tavily(config, URL)

    assert updated["mcp"]["servers"]["tavily"] == {"transport": "streamable-http", "url": URL}
    assert updated["mcp"]["servers"]["github"]["url"] == "https://gh"
    assert "tavily" not in config["mcp"]["servers"], "input must not be mutated"
    assert tavily_enabled(updated, "team-a")


def test_enabled_means_this_workspaces_proxy_on_any_host() -> None:
    def config(url: str) -> dict:
        return {"mcp": {"servers": {"tavily": {"transport": "streamable-http", "url": url}}}}

    assert tavily_enabled(config("http://host.docker.internal:8080/apis/tavily/v2/workspaces/team-a/mcp/"), "team-a")
    assert not tavily_enabled(config(URL), "team-b")
    assert not tavily_enabled(config("https://mcp.tavily.com/mcp/"), "team-a")
    assert not tavily_enabled({"mcp": {"servers": {}}}, "team-a")


def test_with_tavily_handles_missing_mcp_section() -> None:
    assert with_tavily({"name": "a"}, URL)["mcp"] == {
        "servers": {"tavily": {"transport": "streamable-http", "url": URL}}
    }


def test_without_tavily_removes_only_tavily() -> None:
    config = with_tavily({"mcp": {"servers": {"github": {"url": "https://gh"}}}}, URL)

    updated = without_tavily(config)

    assert updated["mcp"]["servers"] == {"github": {"url": "https://gh"}}
    assert not tavily_enabled(updated, "team-a")
    assert tavily_enabled(config, "team-a"), "input must not be mutated"


def test_without_tavily_tolerates_configs_without_mcp() -> None:
    assert without_tavily({"name": "a"}) == {"name": "a"}


def test_wait_returns_when_running() -> None:
    client = MagicMock()
    statuses: Iterator[DeploymentStatus] = iter(["starting", "starting", "running"])
    client.get_deployment.side_effect = lambda **_: MagicMock(
        data=lambda: AgentDeployment(name="d", status=next(statuses))
    )
    sleeps: list[float] = []

    deployment = wait_for_deployment(client, workspace="team-a", name="d", timeout=60, sleep=sleeps.append)

    assert deployment.status == "running"
    assert len(sleeps) == 2


def test_wait_gives_up_at_timeout() -> None:
    client = MagicMock()
    client.get_deployment.return_value.data.return_value = AgentDeployment(name="d", status="starting")
    now = iter([0.0, 5.0, 11.0])

    deployment = wait_for_deployment(
        client, workspace="team-a", name="d", timeout=10, sleep=lambda _: None, clock=lambda: next(now)
    )

    assert deployment.status == "starting"
