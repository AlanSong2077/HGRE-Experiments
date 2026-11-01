"""LLM runtime prior for local-tool cold start and baseline."""
from __future__ import annotations
import hashlib, json
from local_tool_runtime import config

CACHE = config.RESULTS / "llm_prior_cache.json"
PROMPT = ("Estimate the wall-clock runtime in milliseconds for the following shell command.\n"
          "Output ONLY compact JSON: {\"estimated_ms\": <number>}.\n")


def _load():
    try: return json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception: return {}


def _save(cache):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache,ensure_ascii=False,indent=1),encoding="utf-8")


def estimate(case, command, fallback_ms=2000.0):
    """Return (ms, source). source=llm|constant_fallback|cache."""
    key = str(case.get("case_id")) + "|" + hashlib.sha256(str(command).encode()).hexdigest()[:16]
    cache = _load()
    if key in cache:
        entry = cache[key]
        if isinstance(entry, dict):
            return float(entry.get("ms") or fallback_ms), str(entry.get("source") or "cache"), True
        return float(entry), "cache", True
    value = None
    if config.api_key():
        try:
            from langchain_deepseek import ChatDeepSeek
            llm = ChatDeepSeek(model=config.model_name(), api_key=config.api_key(),
                               api_base=config.base_url(), temperature=0, max_tokens=80,
                               timeout=30, max_retries=2)
            text = llm.invoke(PROMPT + "Command type: %s\nSource: %s\nCommand:\n%s" % (
                case.get("command_type"), json.dumps(case.get("source") or {},ensure_ascii=False),
                str(command)[:4000])).content
            a,b = str(text).find("{"), str(text).rfind("}")
            value = float(json.loads(str(text)[a:b+1]).get("estimated_ms"))
        except Exception:
            value = None
    if value is None or value <= 0:
        value = fallback_ms
        source = "constant_fallback"
    else:
        source = "llm"
    cache[key] = {"ms": value, "source": source}
    _save(cache)
    return value, source, False
