"""Configuration for the controlled local-tool experiment."""
from __future__ import annotations

import os

from hgre_core.paths import (
    LOCAL_TOOL_CASES,
    LOCAL_TOOL_DATA,
    LOCAL_TOOL_RESULTS,
    LOCAL_TOOL_RUNTIME,
)

CASES = LOCAL_TOOL_CASES
DATA = LOCAL_TOOL_DATA
RESULTS = LOCAL_TOOL_RESULTS
RUNTIME = LOCAL_TOOL_RUNTIME
DEFAULT_REPS = 3


def api_key():
    return os.environ.get("DEEPSEEK_API_KEY", "")


def base_url():
    return os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def model_name():
    return os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
