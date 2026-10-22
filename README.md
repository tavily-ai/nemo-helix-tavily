<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tavily plugin

Gives NeMo Helix agents live web search and page extraction through [Tavily](https://tavily.com). Agents get two tools, `tavily_search` and `tavily_extract`, from Tavily's hosted MCP server.

## How it works

```
agent ──MCP──▶ /apis/tavily/v2/workspaces/<ws>/mcp ──MCP──▶ mcp.tavily.com/mcp
               (this plugin:
                 • adds the workspace's Tavily key from Helix Secrets
                 • tags traffic with X-Client-Name: nemo-helix
                 • lists and allows only tavily_search and tavily_extract)
```

The proxy forwards MCP traffic unchanged apart from those three things. Tool schemas, descriptions and behavior come from the hosted server, so Tavily's improvements reach Helix agents without a plugin release. Agents never see the Tavily API key.

| Surface | Entry point | What it provides |
|---|---|---|
| Service | `nemo.services` → `tavily` | MCP proxy at `POST /apis/tavily/v2/workspaces/{workspace}/mcp` |
| CLI | `nemo.cli` → `tavily` | `nemo tavily setup`, `enable`, `disable`, `status` |
| Skill | `nemo.skills` → `tavily` | `nemo-tavily-web-search`, for coding assistants and the Studio Assistant |

## Install

The plugin is a uv workspace member, so `make bootstrap` (or `uv sync --all-packages`) installs it. Restart the platform afterwards; plugin services are discovered at startup:

```bash
nemo services restart
```

## Use

```bash
nemo tavily setup --workspace <ws>                 # store the key as workspace secret `tavily-api-key` (prompts)
nemo tavily enable <agent> --workspace <ws> --yes  # add Tavily MCP, rebuild and redeploy the agent
nemo tavily status --workspace <ws>                # key configured? which agents have Tavily MCP?
nemo tavily disable <agent> --workspace <ws> --yes
```

`enable` adds this to the agent's config stored in Helix (not your local `agent.yaml`), then rebuilds the agent the same way `nemo agents redeploy` does:

```yaml
mcp:
  servers:
    tavily:
      transport: streamable-http
      url: <platform>/apis/tavily/v2/workspaces/<ws>/mcp
```

## Configuration

| Setting | Env var | Default |
|---|---|---|
| Upstream MCP URL | `NEMO_TAVILY_MCP_URL` | `https://mcp.tavily.com/mcp` |
| Tools agents can use | `NEMO_TAVILY_ALLOWED_TOOLS` | `["tavily_search","tavily_extract"]` |
| Upstream read timeout (s) | `NEMO_TAVILY_TIMEOUT_SECONDS` | `300` |

Set these in the platform's environment, or under the `tavily` key in the platform config file.

## Limits

- Only `nemo-agents-spec-v1` agents. NAT workflows can use the `nemo-agent-toolkit-tavily` package.
- The default MCP URL is the platform's own address, which subprocess deployments can reach. For docker or Kubernetes deployments, pass `--mcp-url` with an address the container can reach. With platform auth on, container agents must call the proxy through their auth-proxy sidecar; this is not handled yet.

## Tests

```bash
uv run --frozen pytest plugins/nemo-tavily/tests -v
```
