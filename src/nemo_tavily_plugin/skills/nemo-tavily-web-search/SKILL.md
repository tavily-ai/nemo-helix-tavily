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

3. Verify. `nemo tavily status --workspace <ws>` should show the key as configured and the agent as `enabled`. Then ask the agent something that needs current information:

   ```bash
   nemo agents invoke -a <agent> -w <ws> -i "What is the latest stable Python release?"
   ```

   The answer should cite web sources. Tool calls appear in the agent's traces in Studio.

## Turn web search off

```bash
nemo tavily disable <agent> --workspace <ws> --yes
```

## Prompting the agent

Enabling web search adds the tools but does not change the agent's instructions. If the agent ignores the tools, suggest adding a line to its system prompt such as: "Use web search for anything recent or time-sensitive, and cite source URLs." The model does not know today's date, so for "this year" or "latest" questions, putting the date in the prompt or the question helps.

## Limits

- Only `nemo-agents-spec-v1` agents are supported. For NAT workflows, use the `nemo-agent-toolkit-tavily` package.
- The default MCP URL is the platform's own address, which works for subprocess deployments. For docker or Kubernetes deployments, pass `--mcp-url` with an address the agent container can reach.
- Agents see only `tavily_search` and `tavily_extract`. Calls to other Tavily tools are refused.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No Tavily API key in workspace` | The workspace secret is missing | `nemo tavily setup --workspace <ws>` |
| Agent errors with HTTP 412 from `/apis/tavily/...` | The key was deleted after enabling | Run `nemo tavily setup` again |
| `... only nemo-agents-spec-v1 agents are supported` | NAT workflow agent | Use `nemo-agent-toolkit-tavily` |
| `... runs in docker mode, which cannot reach ...` | Container deployment | Re-run `enable` with `--mcp-url` |
