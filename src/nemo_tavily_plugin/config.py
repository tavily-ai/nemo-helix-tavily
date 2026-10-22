# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Tavily plugin.

Every field has a default, so the plugin works without operator configuration.
Override with environment variables or the ``tavily`` key of the platform config::

    NEMO_TAVILY_MCP_URL=https://mcp.tavily.com/mcp
    NEMO_TAVILY_ALLOWED_TOOLS='["tavily_search"]'
"""

from __future__ import annotations

from typing import ClassVar

from nemo_helix_plugin.config import NemoConfig
from pydantic import Field

# Workspace secret that holds the Tavily API key. Fixed so `nemo tavily setup` and
# the proxy always agree on it.
SECRET_NAME = "tavily-api-key"


class TavilyConfig(NemoConfig):
    """Configuration for the Tavily MCP proxy."""

    plugin_name: ClassVar[str] = "tavily"
    plugin_description: ClassVar[str] = "Configuration for the NeMo Helix Tavily plugin."

    mcp_url: str = Field(
        default="https://mcp.tavily.com/mcp",
        description="Upstream Tavily MCP endpoint the proxy forwards to.",
    )
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["tavily_search", "tavily_extract"],
        description="Tavily MCP tools agents can list and call. Every other tool is hidden and refused.",
    )
    timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Read timeout for upstream MCP requests, including idle time on SSE responses.",
    )
