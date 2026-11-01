"""Isolated single-trial worker: run one agent trial and emit JSON."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from leetcode_runtime import agent

MARKER = "__TRIAL_RESULT__"


def main():
    payload = json.load(sys.stdin)
    problem = payload["problem"]
    history = payload.get("history") or []
    k = int(payload["k"])
    problem = dict(problem)
    problem["repeat_k"] = k
    problem["trial_timeout_s"] = float(payload.get("trial_timeout_s") or 180.0)
    state = {"history": history}
    try:
        rec = agent.run_task(problem, state, use_llm=True)
        out = {"ok": True, "rec": rec}
    except BaseException as exc:
        out = {"ok": False, "error": repr(exc),
               "trace": traceback.format_exc()[-1200:]}
    print(MARKER + json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
