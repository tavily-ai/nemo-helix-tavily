---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-tavily-web-search
description: 'Gives a NeMo Helix agent live web search and page extraction through Tavily, or removes it. Use when the user wants an agent to search the web, look up current or recent information, cite sources, read a URL, ground answers in live data, or turn web search on or off for an agent. Trigger keywords: web search, internet search, Tavily, search tool, extract, browse, current events, up-to-date answers, grounding, citations, nemo tavily.'
version: 0.1
allowed-tools: Bash
metadata:
  author: Tavily <support@tavily.com>
---

# Tavily web search for NeMo Helix agents

The `nemo tavily` commands give an agent two tools, `tavily_search` and `tavily_extract`, served by Tavily's hosted MCP server through a Helix proxy. The proxy adds the workspace's Tavily API key, so agents never hold the key.

## Before you start

1. Confirm the platform is up: `nemo workspaces list`.
2. Confirm the plugin is installed: `nemo tavily --help`. If the command is missing, the `nemo-tavily-plugin` package is not installed in the platform's environment; tell the user rather than working around it.
3. Pick the workspace. Every command takes `--workspace`; otherwise it uses the active context's workspace.
4. Confirm the proxy is enabled: `nemo tavily status --workspace <ws>` should print `Tavily proxy: enabled`. It is off by default. Turning it on is a platform operator's change: set `NEMO_TAVILY_ENABLED=true` in the platform's environment (Helm: `platformConfig.tavily.enabled: true`) and restart the platform. Ask the user before restarting anything; do not do it on your own.

## Turn web search on

1. Store the Tavily API key once per workspace. Ask the user for the key; never invent or echo it.

   ```bash
   nemo tavily setup --workspace <ws>          # prompts for the key
   ```

   Prefer the prompt. `--api-key tvly-...` also works but leaves the key in shell history.

2. Enable it on the agent. This rebuilds the agent and, if it was deployed, redeploys it with the same runtime settings, so confirm with the user first.

   ```bash
   nemo tavily enable <agent> --workspace <ws> --yes
   ```

3. Verify. `nemo tavily status --workspace <ws>` should show the proxy as enabled, the key as configured and the agent as `enabled`. Then ask the agent something that needs current information:

   ```bash
   nemo agents invoke -a <agent> -w <ws> -i "What is the latest stable Python release?"
   ```

   The answer should cite web sources. Tool calls appear in the agent's traces.

## Search or extract from the CLI

The same tools can be called directly, without an agent, using the workspace's key. Use these to check the key works, or to look something up:

```bash
nemo tavily search "<query>" --workspace <ws> [-n 5] [--depth advanced] [--topic news]
nemo tavily extract <url> [<url> ...] --workspace <ws> [--query "<focus>"] [--depth advanced]
```

`search` prints a table of results; `extract` prints each page as markdown. Add `-f json` for Tavily's full response.

## Turn web search off

```bash
nemo tavily disable <agent> --workspace <ws> --yes
```

## Prompting the agent

Enabling web search adds the tools but does not change the agent's instructions. If the agent ignores the tools, suggest adding a line to its system prompt such as: "Use web search for anything recent or time-sensitive, and cite source URLs." The model does not know today's date, so for "this year" or "latest" questions, putting the date in the prompt or the question helps.

## Limits

- Only `nemo-agents-spec-v1` agents are supported.
- `enable` picks an MCP URL the agent can reach for subprocess, Docker and Kubernetes deployments. Other deployment modes need `--mcp-url`.
- Agents see only `tavily_search` and `tavily_extract`. Calls to other Tavily tools are refused.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `The Tavily MCP proxy is not enabled on this platform ...` | The proxy is off (the default), or the plugin is not installed on the platform | Ask the operator to set `NEMO_TAVILY_ENABLED=true` (Helm: `platformConfig.tavily.enabled: true`) and restart |
| `No Tavily API key in workspace` (HTTP 412 from `search`/`extract`) | The workspace secret is missing | `nemo tavily setup --workspace <ws>` |
| Agent errors with HTTP 412 from `/apis/tavily/...` | The key was deleted after enabling | Run `nemo tavily setup` again |
| `... only nemo-agents-spec-v1 agents are supported` | Agent uses another config format | Enable Tavily on a `nemo-agents-spec-v1` agent instead |
| `... runs in <mode> mode. Pass --mcp-url ...` | Deployment mode with no automatic URL | Re-run `enable` with `--mcp-url` |
