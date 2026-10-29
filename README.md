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

## Enable Tavily

Installing the plugin serves nothing. The proxy stays off until a platform operator turns it on, so an installed but unused plugin mounts no routes and never contacts Tavily. Enabling web search for an agent takes three steps:

| Step | Who | How often |
|---|---|---|
| 1. Turn on the proxy | Platform operator | Once per platform |
| 2. Store a Tavily API key | Workspace user | Once per workspace |
| 3. Enable Tavily on an agent | Workspace user | Once per agent |

Requires NeMo Helix 0.9 or later. Earlier releases have a different agents API, and the plugin does not load on them.

### 1. Turn on the proxy (platform operator)

The proxy reads `enabled` at startup, so every change needs a platform restart.

**Local platform.** In a source checkout the plugin is a uv workspace member, so `make bootstrap` (or `uv sync --all-packages`) installs it. Otherwise, install the `nemo-tavily-plugin` wheel into the environment that runs `nemo services`. Then restart with the proxy on:

```bash
NEMO_TAVILY_ENABLED=true nemo services restart
```

The restarted platform takes the variable from your shell, so set it every time you start the platform (for example, export it in your shell profile). Alternatively, set `tavily: {enabled: true}` in the platform config file.

**Kubernetes (Helm).** The plugin must be installed in the API image, because the chart cannot add Python packages at runtime. If your `nhx-api` image does not include it, build one that does, and point both `api.image` and `core.image` at it:

```dockerfile
FROM <your nhx-api image>
COPY nemo_tavily_plugin-<version>-py3-none-any.whl /tmp/
RUN /app/.venv/bin/pip install --no-cache-dir /tmp/nemo_tavily_plugin-<version>-py3-none-any.whl
```

Then turn the proxy on in your Helm values and upgrade the release. Pass all of your existing values files as well; a value left out of `helm upgrade` goes back to its default.

```yaml
platformConfig:
  tavily:
    enabled: true
```

```bash
helm upgrade <release> <chart> -n <namespace> -f values.yaml -f values-tavily.yaml --wait
```

**Check.** `nemo tavily status --workspace <ws>` prints `Tavily proxy: enabled, agents use <url>`. On Kubernetes the URL names the in-cluster API Service, for example `http://nemo-helix-api:8080/...`, not `localhost`. If it prints `Tavily proxy: disabled`, the platform did not pick up the setting: check that it restarted.

### 2. Store a Tavily API key (workspace user)

The `nemo tavily` commands come from the same plugin, so install it wherever your `nemo` CLI runs too. Then store the key for the workspace. The command prompts for it, so the key stays out of shell history:

```bash
nemo tavily setup --workspace <ws>
```

The key is saved as the workspace secret `tavily-api-key`. If the proxy is off, `setup` still stores it and prints how to turn the proxy on.

**Check.** Search from the CLI. This goes through the same proxy agents use, with the workspace's key, so it needs no key on your machine:

```bash
nemo tavily search "latest nemotron release" --workspace <ws> -n 3
```

### 3. Enable Tavily on an agent (workspace user)

```bash
nemo tavily enable <agent> --workspace <ws>
```

`enable` rebuilds and redeploys the agent, so it asks for confirmation; `--yes` skips the prompt. The agent is redeployed in the same mode and with the same image it had. Only `nemo-agents-spec-v1` agents are supported.

**Check.** `nemo tavily status --workspace <ws>` lists the agent as `enabled`. Then ask the agent something recent. Its answer should cite web sources, and its traces show `tavily_search` calls.

To turn web search off for an agent:

```bash
nemo tavily disable <agent> --workspace <ws>
```

### Calling the tools directly

`search` and `extract` call Tavily without an agent, for checking a key or looking something up:

```bash
nemo tavily search "latest nemotron release" --workspace <ws> -n 5
nemo tavily extract https://example.com --workspace <ws> -f json > page.json
```

They use the workspace's key, the tool allowlist and the `X-Client-Name` attribution, like agents do. `search` prints a table of results and `extract` prints each page as markdown; `-f json` gives Tavily's full response.

### What `enable` changes

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
