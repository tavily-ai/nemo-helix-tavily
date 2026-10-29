# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo tavily`` — set up Tavily and toggle the Tavily MCP server on agents."""

from __future__ import annotations

from typing import Any, Optional

import typer
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.agents.types import Agent
from nemo_helix_plugin.cli import NemoCLI, create_typer_app
from nemo_helix_plugin.cli_error_handling import handle_errors
from nemo_helix_plugin.cli_options import ListOutputFormatOption, WorkspaceOption
from nemo_helix_plugin.cli_output import Column, format_output
from nemo_helix_plugin.cli_state import cli_state, resolve_base_url, resolve_cli_workspace, resolve_output_format
from nemo_helix_plugin.cli_warnings import collect_warnings
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest, HelixSecretUpdateRequest
from nemo_tavily_plugin.agents import (
    SERVER_NAME,
    SPEC_CONFIG_FORMAT,
    live_deployments,
    proxy_url,
    rebuild_agent,
    tavily_enabled,
    tavily_server_url,
    wait_for_deployment,
    with_tavily,
    without_tavily,
)
from nemo_tavily_plugin.config import SECRET_NAME
from nemo_tavily_plugin.tool_call import ToolCallError, call_tool

AgentArgument = typer.Argument(..., help="Name of the agent.")
YesOption = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt.")
WaitOption = typer.Option(True, "--wait/--no-wait", help="Wait for the redeployed agent to be running.")
TimeoutOption = typer.Option(300, "--timeout", "-t", help="Seconds to wait for the redeployed agent.")

_STATUS_COLUMNS = [Column("name"), Column("config_format"), Column("tavily_mcp")]
_SEARCH_COLUMNS = [Column("title"), Column("url"), Column("score")]
# Tavily searches and extractions can take longer than the CLI's default timeout.
_TOOL_TIMEOUT_SECONDS = 300.0


def _secret_exists(secrets: SecretsClient, workspace: str) -> bool:
    try:
        secrets.get_secret(workspace=workspace, name=SECRET_NAME).data()
    except NotFoundError:
        return False
    return True


def _fail(message: str) -> typer.Exit:
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(code=1)


def _rebuild(
    agents: AgentsClient,
    *,
    workspace: str,
    agent: Agent,
    config: dict[str, Any],
    action: str,
    yes: bool,
    wait: bool,
    timeout: int,
) -> None:
    """Confirm, rebuild the agent with *config*, and wait for its new deployment."""
    if not yes:
        typer.confirm(
            f"{action} Tavily MCP on {agent.name!r} (web search, web extract)? This rebuilds and redeploys the agent.",
            abort=True,
        )
    deployment = rebuild_agent(agents, workspace=workspace, agent=agent, config=config, echo=typer.echo)
    if deployment is None:
        typer.echo("The agent had no live deployment, so it was not redeployed.")
        return
    if not wait:
        return
    deployment = wait_for_deployment(agents, workspace=workspace, name=deployment.name, timeout=timeout)
    if deployment.status == "failed":
        raise _fail(f"Deployment {deployment.name} failed: {deployment.error or 'no error reported'}")
    if deployment.status != "running":
        raise _fail(f"Deployment {deployment.name} is still {deployment.status} after {timeout}s.")
    typer.echo(f"Deployment {deployment.name} is running.")


def _call(ctx: typer.Context, workspace: str, name: str, arguments: dict[str, Any]) -> Any:
    """Call a Tavily tool through the workspace proxy, leaving out unset arguments."""
    client = cli_state(ctx).typed_client(NemoClient, timeout=_TOOL_TIMEOUT_SECONDS)
    arguments = {key: value for key, value in arguments.items() if value is not None}
    try:
        return call_tool(client, workspace=workspace, name=name, arguments=arguments)
    except ToolCallError as exc:
        raise _fail(str(exc)) from exc


def _printed_as_is(result: Any, output_format: str) -> bool:
    """Print *result* unchanged for json/yaml, or when it lacks the ``results`` the table views expect."""
    if isinstance(result, str):
        typer.echo(result)
        return True
    if output_format in ("json", "yaml"):
        format_output(result, output_format=output_format)
        return True
    if not isinstance(result, dict) or "results" not in result:
        format_output(result, output_format="json")
        return True
    return False


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
                None, "--api-key", help="Tavily API key (tvly-...). Prompted for when omitted."
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
                    workspace=workspace, name=SECRET_NAME, body=HelixSecretUpdateRequest(value=key)
                ).data()
                typer.echo(f"Updated secret {SECRET_NAME!r} in workspace {workspace!r}.")
            else:
                body = HelixSecretCreateRequest(name=SECRET_NAME, value=key, description="Tavily API key")
                secrets.create_secret(workspace=workspace, body=body).data()
                typer.echo(f"Created secret {SECRET_NAME!r} in workspace {workspace!r}.")
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
            if current.config_format != SPEC_CONFIG_FORMAT:
                raise _fail(
                    f"{agent!r} is a {current.config_format} agent; only {SPEC_CONFIG_FORMAT} agents are supported. "
                    "NAT workflows can use the nemo-agent-toolkit-tavily package instead."
                )

            existing_url = tavily_server_url(current.config)
            if existing_url == url:
                typer.echo(f"Tavily MCP is already enabled on {agent!r}.")
                return
            if not _secret_exists(state.typed_client(SecretsClient), workspace):
                raise _fail(f"No Tavily API key in workspace {workspace!r}. Run `nemo tavily setup` first.")
            if mcp_url is None:
                container_modes = {
                    dep.deployment_mode
                    for dep in live_deployments(agents, workspace=workspace, agent=agent)
                    if dep.deployment_mode != "subprocess"
                }
                if container_modes:
                    raise _fail(
                        f"{agent!r} runs in {', '.join(sorted(container_modes))} mode, which cannot reach {url}. "
                        "Pass --mcp-url with an address the container can reach."
                    )
            if existing_url and not tavily_enabled(current.config, workspace):
                typer.echo(f"Replacing {agent!r}'s existing {SERVER_NAME!r} MCP server ({existing_url}).")

            _rebuild(
                agents,
                workspace=workspace,
                agent=current,
                config=with_tavily(current.config, url),
                action="Enable",
                yes=yes,
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
                if existing_url := tavily_server_url(current.config):
                    typer.echo(
                        f"Its {SERVER_NAME!r} MCP server ({existing_url}) does not go through this plugin; "
                        "left unchanged."
                    )
                return

            _rebuild(
                agents,
                workspace=workspace,
                agent=current,
                config=without_tavily(current.config),
                action="Disable",
                yes=yes,
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
            output_format = resolve_output_format(ctx, output_format)
            key_configured = _secret_exists(state.typed_client(SecretsClient), workspace)
            rows = [
                {
                    "name": agent.name,
                    "config_format": agent.config_format,
                    "tavily_mcp": "enabled" if tavily_enabled(agent.config, workspace) else "disabled",
                }
                for agent in state.typed_client(AgentsClient).list_agents(workspace=workspace).items()
            ]
            if output_format in ("json", "yaml"):
                summary = {"workspace": workspace, "api_key_configured": key_configured, "agents": rows}
                format_output(summary, output_format=output_format)
                return
            typer.echo(f"Tavily API key: {'configured' if key_configured else 'missing (run `nemo tavily setup`)'}")
            format_output(rows, is_list=True, output_format=output_format, output_columns=_STATUS_COLUMNS)

        @app.command()
        @collect_warnings
        @handle_errors
        def search(
            ctx: typer.Context,
            query: str = typer.Argument(..., help="Search query."),
            max_results: Optional[int] = typer.Option(
                None, "--max-results", "-n", help="Maximum number of results to return."
            ),
            depth: Optional[str] = typer.Option(
                None, "--depth", help="Search depth: basic, advanced, fast or ultra-fast."
            ),
            topic: Optional[str] = typer.Option(None, "--topic", help="Search topic: general or news."),
            workspace: WorkspaceOption = None,
            output_format: ListOutputFormatOption = None,
        ) -> None:
            """Search the web with Tavily, using the workspace's Tavily key."""
            workspace = resolve_cli_workspace(ctx, workspace)
            output_format = resolve_output_format(ctx, output_format)
            arguments = {"query": query, "max_results": max_results, "search_depth": depth, "topic": topic}
            result = _call(ctx, workspace, "tavily_search", arguments)
            if _printed_as_is(result, output_format):
                return
            if result.get("answer"):
                typer.echo(f"{result['answer']}\n")
            format_output(
                result.get("results") or [],
                is_list=True,
                output_format=output_format,
                output_columns=_SEARCH_COLUMNS,
            )

        @app.command()
        @collect_warnings
        @handle_errors
        def extract(
            ctx: typer.Context,
            urls: list[str] = typer.Argument(..., help="URLs to extract content from."),
            query: Optional[str] = typer.Option(
                None, "--query", "-q", help="Return only the passages of each page relevant to this query."
            ),
            depth: Optional[str] = typer.Option(
                None, "--depth", help="Extract depth: basic, or advanced for protected sites and tables."
            ),
            workspace: WorkspaceOption = None,
            output_format: ListOutputFormatOption = None,
        ) -> None:
            """Extract the content of web pages as markdown with Tavily, using the workspace's Tavily key."""
            workspace = resolve_cli_workspace(ctx, workspace)
            output_format = resolve_output_format(ctx, output_format)
            arguments = {"urls": urls, "query": query, "extract_depth": depth}
            result = _call(ctx, workspace, "tavily_extract", arguments)
            if _printed_as_is(result, output_format):
                return
            for page in result.get("results") or []:
                typer.echo(
                    f"# {page.get('title') or page.get('url')}\n{page.get('url')}\n\n{page.get('raw_content')}\n"
                )
            for failure in result.get("failed_results") or []:
                typer.echo(f"Could not extract {failure.get('url')}: {failure.get('error')}", err=True)
            if not result.get("results"):
                raise typer.Exit(code=1)

        return app
