# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turning the Tavily MCP server on and off for a registered agent.

Tavily is one MCP server entry, ``mcp.servers.tavily``, pointing at this plugin's
workspace proxy. Agents cannot be updated in place, so a toggle rebuilds the
agent the same way ``nemo agents redeploy`` does: undeploy, delete, register with
the new config, then deploy again with the previous runtime settings. The
agent's Ethos fileset survives the delete, and the stored config wins over any
``agent.yaml`` in it at deploy time.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.agents.types import Agent, AgentDeployment, CreateDeploymentRequest, RegisterAgentRequest
from nemo_helix_plugin.client.errors import NotFoundError

SERVER_NAME = "tavily"
SPEC_CONFIG_FORMAT = "nemo-agents-spec-v1"
LIVE_DEPLOYMENT_STATUSES = frozenset({"pending", "starting", "running"})


def proxy_path(workspace: str) -> str:
    return f"/apis/tavily/v2/workspaces/{workspace}/mcp"


def proxy_url(base_url: str, workspace: str) -> str:
    """Return the workspace's Tavily MCP proxy URL on the platform at *base_url*."""
    return f"{base_url.rstrip('/')}{proxy_path(workspace)}"


def tavily_server_url(config: dict[str, Any]) -> str | None:
    """Return the URL of the agent's ``tavily`` MCP server, if it has one."""
    servers = (config.get("mcp") or {}).get("servers") or {}
    return (servers.get(SERVER_NAME) or {}).get("url")


def tavily_enabled(config: dict[str, Any], workspace: str) -> bool:
    """Whether *config* reaches Tavily through this plugin's proxy for *workspace*.

    Only the URL path is compared, so agents enabled with ``--mcp-url`` (another host
    for the same platform) count. A ``tavily`` server pointing anywhere else, such as
    Tavily's hosted MCP directly, does not.
    """
    return urlsplit(tavily_server_url(config) or "").path.rstrip("/") == proxy_path(workspace)


def with_tavily(config: dict[str, Any], url: str) -> dict[str, Any]:
    """Return a copy of *config* with the Tavily MCP server attached at *url*."""
    updated = copy.deepcopy(config)
    mcp = updated["mcp"] = updated.get("mcp") or {}
    mcp["servers"] = {**(mcp.get("servers") or {}), SERVER_NAME: {"transport": "streamable-http", "url": url}}
    return updated


def without_tavily(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *config* with the Tavily MCP server removed."""
    updated = copy.deepcopy(config)
    servers = (updated.get("mcp") or {}).get("servers")
    if isinstance(servers, dict):
        servers.pop(SERVER_NAME, None)
    return updated


def live_deployments(client: AgentsClient, *, workspace: str, agent: str) -> list[AgentDeployment]:
    return [
        dep
        for dep in client.list_deployments(workspace=workspace).items()
        if dep.agent == agent and dep.status in LIVE_DEPLOYMENT_STATUSES
    ]


def rebuild_agent(
    client: AgentsClient,
    *,
    workspace: str,
    agent: Agent,
    config: dict[str, Any],
    echo: Callable[[str], None],
) -> AgentDeployment | None:
    """Replace *agent*'s config with *config*, redeploying it if it was running.

    Returns the new deployment, or ``None`` when the agent had no live deployment.
    """
    live = live_deployments(client, workspace=workspace, agent=agent.name)
    if len({dep.deployment_mode for dep in live}) > 1:
        raise ValueError(
            f"Agent {agent.name!r} has live deployments in different modes; undeploy all but one and try again."
        )

    for dep in live:
        client.delete_deployment(workspace=workspace, name=dep.name).data()
        echo(f"Undeployed {dep.name}.")
    try:
        client.delete_agent(workspace=workspace, name=agent.name).data()
    except NotFoundError:
        pass
    client.register_agent(
        workspace=workspace,
        body=RegisterAgentRequest(
            name=agent.name, description=agent.description, config=config, config_format=agent.config_format
        ),
    ).data()
    echo(f"Re-registered agent {agent.name}.")

    if not live:
        return None
    previous = live[0]
    deployment = client.create_deployment(
        workspace=workspace,
        body=CreateDeploymentRequest(
            agent=agent.name,
            deployment_mode=previous.deployment_mode,
            image=previous.image,
            use_image_entrypoint=previous.use_image_entrypoint,
            environment=previous.environment,
        ),
    ).data()
    echo(f"Deploying {deployment.name} ({deployment.deployment_mode}).")
    return deployment


def wait_for_deployment(
    client: AgentsClient,
    *,
    workspace: str,
    name: str,
    timeout: float,
    poll_interval: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> AgentDeployment:
    """Poll a deployment until it is running or failed, or *timeout* seconds pass."""
    deadline = clock() + timeout
    while True:
        deployment = client.get_deployment(workspace=workspace, name=name).data()
        if deployment.status in ("running", "failed") or clock() >= deadline:
            return deployment
        sleep(poll_interval)
