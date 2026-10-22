# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Tavily plugin's authz scope and permission vocabulary."""

from __future__ import annotations

from nemo_helix_plugin.authz import AuthzScope, PermissionSet, perm

scope = AuthzScope("tavily")


class TavilyMcpPerms(PermissionSet, namespace="tavily.mcp"):
    INVOKE = perm("Call Tavily web search tools through the workspace MCP proxy")
