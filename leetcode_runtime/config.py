"""Configuration for the LeetCode runtime experiment."""
from __future__ import annotations

import os

from hgre_core.paths import LEETCODE_DATA, LEETCODE_RESULTS

DATASET = LEETCODE_DATA
RESULTS = LEETCODE_RESULTS

DEFAULT_REPS = 5
K_MIN = 200
K_MAX = 2000
TRIAL_TIMEOUT_S = 180.0
POLL_MS = 100


def api_key():
    return os.environ.get("DEEPSEEK_API_KEY", "")


def base_url():
    return os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def model_name():
    return os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
