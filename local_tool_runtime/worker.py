"""Run one local-tool trial in an isolated process."""
from __future__ import annotations
import json, sys, traceback
from local_tool_runtime import agent
MARKER = "__TRIAL_RESULT__"
def main():
    payload = json.load(sys.stdin)
    case = payload["case"]
    try:
        rec = agent.run_case(case, {})
        out = {"ok": True, "rec": rec}
    except BaseException as exc:
        etype = type(exc).__name__
        out = {"ok": False, "error": etype + ": " + repr(exc),
               "error_type": etype, "trace": traceback.format_exc()[-1200:]}
    print(MARKER + json.dumps(out, ensure_ascii=False))
if __name__ == "__main__": main()
