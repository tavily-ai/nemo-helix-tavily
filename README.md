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
| Service | `nemo.services` → `tavily` | MCP proxy at `POST /apis/tavily/v2/workspaces/{workspace}/mcp`, and `GET .../status` for the URL agents use |
| CLI | `nemo.cli` → `tavily` | `nemo tavily setup`, `enable`, `disable`, `status`, `search`, `extract` |
| Skill | `nemo.skills` → `tavily` | `nemo-tavily-web-search`, for coding assistants and the Studio Assistant |

## Install

The plugin is a uv workspace member, so `make bootstrap` (or `uv sync --all-packages`) installs it.

Installing it serves nothing: the proxy is off until an operator turns it on, so an installed but unused plugin mounts no routes and never contacts Tavily. Turn it on in the platform's environment and restart; routes are mounted at startup:

```bash
NEMO_TAVILY_ENABLED=true nemo services restart
```

With Helm, set it in the platform config instead:

```yaml
platformConfig:
  tavily:
    enabled: true
```

## Use

```bash
nemo tavily setup --workspace <ws>                 # store the key as workspace secret `tavily-api-key` (prompts)
nemo tavily enable <agent> --workspace <ws> --yes  # add Tavily MCP, rebuild and redeploy the agent
nemo tavily status --workspace <ws>                # proxy enabled? key configured? which agents have Tavily MCP?
nemo tavily disable <agent> --workspace <ws> --yes

nemo tavily search "latest nemotron release" --workspace <ws> -n 5   # call the tools directly
nemo tavily extract https://example.com --workspace <ws> -f json
```

`search` and `extract` go through the same proxy as agents, so they use the workspace's key, the tool allowlist and the `X-Client-Name` attribution. They need no key on your machine.

Every command checks that the proxy is enabled. If it is not, `setup` still stores the key and says how to turn the proxy on; the other commands stop with that message.

`enable` adds this to the agent's config stored in Helix (not your local `agent.yaml`), then rebuilds the agent the same way `nemo agents redeploy` does:

```yaml
mcp:
  servers:
    tavily:
      transport: streamable-http
      url: <platform>/apis/tavily/v2/workspaces/<ws>/mcp
```

The platform reports `<platform>` from `GET /apis/tavily/v2/workspaces/<ws>/status`, as an address deployed agents can reach: the in-cluster API Service on Kubernetes (`NHX_INTERNAL_BASE_URL`, set by the Helm chart), otherwise the platform's own base URL. `enable` adjusts it for the agent's deployment mode: Docker containers reach a loopback host through `host.docker.internal`. Pass `--mcp-url` to override.

## Configuration

| Setting | Env var | Default |
|---|---|---|
| Serve the proxy (read at startup) | `NEMO_TAVILY_ENABLED` | `false` |
| Upstream MCP URL | `NEMO_TAVILY_MCP_URL` | `https://mcp.tavily.com/mcp` |
| Tools agents can use | `NEMO_TAVILY_ALLOWED_TOOLS` | `["tavily_search","tavily_extract"]` |
| Upstream read timeout (s) | `NEMO_TAVILY_TIMEOUT_SECONDS` | `300` |

Set these in the platform's environment, or under the `tavily` key in the platform config file (Helm: `platformConfig.tavily`).

## Limits

- Only `nemo-agents-spec-v1` agents. NAT workflows can use the `nemo-agent-toolkit-tavily` package.
- Subprocess, Docker and Kubernetes deployments get a reachable MCP URL automatically. Other modes, such as OpenShell, need `--mcp-url`.
- With platform auth on, container agents must call the proxy through their auth-proxy sidecar; this is not handled yet.

## Tests

```bash
uv run --frozen pytest plugins/nemo-tavily/tests -v
```
