# SPDX-FileCopyrightText: Copyright (c) 2026 Alpha AI Technologies Inc dba Tavily
# SPDX-License-Identifier: MIT

"""The Tavily plugin's authz scope and permission vocabulary."""

from __future__ import annotations

from nemo_helix_plugin.authz import AuthzScope, PermissionSet, perm

scope = AuthzScope("tavily")


class TavilyMcpPerms(PermissionSet, namespace="tavily.mcp"):
    INVOKE = perm("Call Tavily web search tools through the workspace MCP proxy")
    READ = perm("See the URL agents use to reach the workspace MCP proxy")
