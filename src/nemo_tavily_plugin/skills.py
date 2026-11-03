# SPDX-FileCopyrightText: Copyright (c) 2026 Alpha AI Technologies Inc dba Tavily
# SPDX-License-Identifier: MIT

from pathlib import Path


def get_skills_path() -> Path:
    """Return the directory containing Tavily plugin skills."""
    return Path(__file__).parent / "skills"
