"""LLM prior for M2 cold start (and LLM-estimate baseline)."""
from __future__ import annotations

import hashlib
import json

from hgre_core.paths import LEETCODE_RESULTS
from leetcode_runtime import config

CACHE = LEETCODE_RESULTS / "llm_prior_cache.json"
PROMPT = (
    "You are estimating the wall-clock runtime of a Python LeetCode solution.\\n"
    "The official tests will be executed K times; estimate the TOTAL time in milliseconds.\\n"
    "Output ONLY compact JSON: {\"estimated_ms\": <number>}.\n"
)


def _load():
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(cache):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def estimate(problem, code, k, fallback_ms=1000.0):
    key = problem.get("task_id", "") + "|" + hashlib.sha256(code.encode("utf-8")).hexdigest()[:16] + "|" + str(k)
    cache = _load()
    if key in cache:
        return float(cache[key])
    value = None
    if config.api_key():
        try:
            from langchain_deepseek import ChatDeepSeek
            llm = ChatDeepSeek(model=config.model_name(), api_key=config.api_key(),
                               api_base=config.base_url(), temperature=0, max_tokens=80,
                               timeout=30, max_retries=2)
            text = llm.invoke(PROMPT + "K=%d\\nProblem: %s\\nCode:\\n%s" % (
                k, str(problem.get("problem_description") or "")[:1200], code[:4000])).content
            start, end = text.find("{"), text.rfind("}")
            payload = json.loads(text[start:end + 1])
            value = float(payload.get("estimated_ms"))
        except Exception:
            value = None
    if value is None or value <= 0:
        value = fallback_ms
    cache[key] = value
    _save(cache)
    return value
