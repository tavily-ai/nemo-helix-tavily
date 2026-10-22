# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo tavily`` — set up Tavily and toggle the Tavily MCP server on agents."""

from __future__ import annotations

from typing import Any, Optional

import typer
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.cli import NemoCLI, create_typer_app
from nemo_helix_plugin.cli_error_handling import handle_errors
from nemo_helix_plugin.cli_options import ListOutputFormatOption, WorkspaceOption
from nemo_helix_plugin.cli_output import Column, format_output
from nemo_helix_plugin.cli_state import cli_state, resolve_base_url, resolve_cli_workspace, resolve_output_format
from nemo_helix_plugin.cli_warnings import collect_warnings
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest, HelixSecretUpdateRequest
from nemo_tavily_plugin.agents import (
    LIVE_DEPLOYMENT_STATUSES,
    SERVER_NAME,
    UnsupportedAgentError,
    agent_deployments,
    proxy_url,
    rebuild_agent,
    require_spec_agent,
    tavily_enabled,
    wait_for_deployment,
    with_tavily,
    without_tavily,
)
from nemo_tavily_plugin.config import DEFAULT_SECRET_NAME

AgentArgument = typer.Argument(..., help="Name of the agent.")
YesOption = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt.")
WaitOption = typer.Option(True, "--wait/--no-wait", help="Wait for the redeployed agent to be running.")
TimeoutOption = typer.Option(300, "--timeout", "-t", help="Seconds to wait for the redeployed agent.")

_STATUS_COLUMNS = [Column("name"), Column("config_format"), Column("tavily_mcp")]


def _secret_exists(secrets: SecretsClient, workspace: str) -> bool:
    try:
        secrets.get_secret(workspace=workspace, name=DEFAULT_SECRET_NAME).data()
    except NotFoundError:
        return False
    return True


def _tavily_server_url(config: dict[str, Any]) -> str | None:
    servers = (config.get("mcp") or {}).get("servers") or {}
    return (servers.get(SERVER_NAME) or {}).get("url")


def _fail(message: str) -> typer.Exit:
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(code=1)


class TavilyCLI(NemoCLI):
    """Exposes ``nemo tavily ...``."""

    name = "tavily"
    description = "Tavily MCP (web search, web extract) for NeMo Helix agents."

    def get_cli(self) -> typer.Typer:
        app = create_typer_app(help=self.description)

        @app.command()
        @collect_warnings
        @handle_errors
        def setup(
            ctx: typer.Context,
            api_key: Optional[str] = typer.Option(
                None,
                "--api-key",
                help="Tavily API key (tvly-...). Prompted for when omitted.",
            ),
            workspace: WorkspaceOption = None,
        ) -> None:
            """Store the workspace's Tavily API key in Helix Secrets."""
            workspace = resolve_cli_workspace(ctx, workspace)
            key = (api_key or typer.prompt("Tavily API key", hide_input=True)).strip()
            if not key.startswith("tvly-"):
                raise _fail("Tavily API keys start with 'tvly-'.")

            secrets = cli_state(ctx).typed_client(SecretsClient)
            if _secret_exists(secrets, workspace):
                secrets.update_secret(
                    workspace=workspace,
                    name=DEFAULT_SECRET_NAME,
                    body=HelixSecretUpdateRequest(value=key),
                ).data()
                typer.echo(f"Updated secret {DEFAULT_SECRET_NAME!r} in workspace {workspace!r}.")
            else:
                secrets.create_secret(
                    workspace=workspace,
                    body=HelixSecretCreateRequest(
                        name=DEFAULT_SECRET_NAME,
                        value=key,
                        description="Tavily API key used by the Tavily MCP proxy.",
                    ),
                ).data()
                typer.echo(f"Created secret {DEFAULT_SECRET_NAME!r} in workspace {workspace!r}.")
            typer.echo(f"Next: nemo tavily enable <agent> --workspace {workspace}")

        @app.command()
        @collect_warnings
        @handle_errors
        def enable(
            ctx: typer.Context,
            agent: str = AgentArgument,
            mcp_url: Optional[str] = typer.Option(
                None,
                "--mcp-url",
                help=(
                    "MCP URL the agent should use. Defaults to this platform's Tavily proxy, "
                    "which suits subprocess deployments; container deployments need a URL they can reach."
                ),
            ),
            yes: bool = YesOption,
            wait: bool = WaitOption,
            timeout: int = TimeoutOption,
            workspace: WorkspaceOption = None,
        ) -> None:
            """Enable Tavily MCP (web search, web extract) on an agent. Rebuilds and redeploys the agent."""
            workspace = resolve_cli_workspace(ctx, workspace)
            state = cli_state(ctx)
            url = mcp_url or proxy_url(resolve_base_url(), workspace)
            agents = state.typed_client(AgentsClient)
            current = agents.get_agent(workspace=workspace, name=agent).data()
            try:
                require_spec_agent(current)
            except UnsupportedAgentError as exc:
                raise _fail(str(exc)) from exc

            existing_url = _tavily_server_url(current.config)
            if existing_url == url:
                typer.echo(f"Tavily MCP is already enabled on {agent!r}.")
                return
            if not _secret_exists(state.typed_client(SecretsClient), workspace):
                raise _fail(f"No Tavily API key in workspace {workspace!r}. Run `nemo tavily setup` first.")
            if mcp_url is None:
                container_modes = {
                    dep.deployment_mode
                    for dep in agent_deployments(agents, workspace=workspace, agent=agent)
                    if dep.status in LIVE_DEPLOYMENT_STATUSES and dep.deployment_mode != "subprocess"
                }
                if container_modes:
                    raise _fail(
                        f"{agent!r} runs in {', '.join(sorted(container_modes))} mode, which cannot reach {url}. "
                        "Pass --mcp-url with an address the container can reach."
                    )

            if existing_url and not tavily_enabled(current.config, workspace):
                typer.echo(f"Replacing {agent!r}'s existing {SERVER_NAME!r} MCP server ({existing_url}).")
            if not yes:
                typer.confirm(
                    f"Enable Tavily MCP on {agent!r} (web search, web extract)? This rebuilds and redeploys the agent.",
                    abort=True,
                )
            deployment = rebuild_agent(
                agents,
                workspace=workspace,
                agent=current,
                config=with_tavily(current.config, url),
                echo=typer.echo,
            )
            _finish(
                agents,
                workspace=workspace,
                deployment_name=deployment.name if deployment else None,
                wait=wait,
                timeout=timeout,
            )
            typer.echo(f"Tavily MCP enabled on {agent!r} (web search, web extract via {url}).")

        @app.command()
        @collect_warnings
        @handle_errors
        def disable(
            ctx: typer.Context,
            agent: str = AgentArgument,
            yes: bool = YesOption,
            wait: bool = WaitOption,
            timeout: int = TimeoutOption,
            workspace: WorkspaceOption = None,
        ) -> None:
            """Disable Tavily MCP (web search, web extract) on an agent. Rebuilds and redeploys the agent."""
            workspace = resolve_cli_workspace(ctx, workspace)
            agents = cli_state(ctx).typed_client(AgentsClient)
            current = agents.get_agent(workspace=workspace, name=agent).data()
            if not tavily_enabled(current.config, workspace):
                typer.echo(f"Tavily MCP is not enabled on {agent!r}.")
                if existing_url := _tavily_server_url(current.config):
                    typer.echo(
                        f"Its {SERVER_NAME!r} MCP server ({existing_url}) does not go through this plugin; "
                        "left unchanged."
                    )
                return
            if not yes:
                typer.confirm(
                    f"Disable Tavily MCP on {agent!r} (web search, web extract)? This rebuilds and redeploys the agent.",
                    abort=True,
                )
            deployment = rebuild_agent(
                agents,
                workspace=workspace,
                agent=current,
                config=without_tavily(current.config),
                echo=typer.echo,
            )
            _finish(
                agents,
                workspace=workspace,
                deployment_name=deployment.name if deployment else None,
                wait=wait,
                timeout=timeout,
            )
            typer.echo(f"Tavily MCP disabled on {agent!r}.")

        @app.command()
        @collect_warnings
        @handle_errors
        def status(
            ctx: typer.Context,
            workspace: WorkspaceOption = None,
            output_format: ListOutputFormatOption = None,
        ) -> None:
            """Show whether the workspace has a Tavily key and which agents have Tavily MCP enabled."""
            workspace = resolve_cli_workspace(ctx, workspace)
            state = cli_state(ctx)
            resolved_output_format = resolve_output_format(ctx, output_format)
            key_configured = _secret_exists(state.typed_client(SecretsClient), workspace)
            rows = [
                {
                    "name": agent.name,
                    "config_format": agent.config_format,
                    "tavily_mcp": "enabled" if tavily_enabled(agent.config, workspace) else "disabled",
                }
                for agent in state.typed_client(AgentsClient).list_agents(workspace=workspace).items()
            ]
            if resolved_output_format in ("json", "yaml"):
                format_output(
                    {"workspace": workspace, "api_key_configured": key_configured, "agents": rows},
                    output_format=resolved_output_format,
                )
                return
            typer.echo(f"Tavily API key: {'configured' if key_configured else 'missing (run `nemo tavily setup`)'}")
            format_output(rows, is_list=True, output_format=resolved_output_format, output_columns=_STATUS_COLUMNS)

        return app


def _finish(agents: AgentsClient, *, workspace: str, deployment_name: str | None, wait: bool, timeout: int) -> None:
    if deployment_name is None:
        typer.echo("The agent had no live deployment, so it was not redeployed.")
        return
    if not wait:
        return
    deployment = wait_for_deployment(agents, workspace=workspace, name=deployment_name, timeout=timeout)
    if deployment.status == "failed":
        raise _fail(f"Deployment {deployment_name} failed: {deployment.error or 'no error reported'}")
    if deployment.status != "running":
        raise _fail(f"Deployment {deployment_name} is still {deployment.status} after {timeout}s.")
    typer.echo(f"Deployment {deployment_name} is running.")
