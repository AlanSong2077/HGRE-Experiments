"""Simplified ReAct coder agent for the repeat-only experiment.

Tools:
  propose_plan(sleep_seconds, python_code)
  execute_plan() -> runs the official tests K times, measures total runtime.

M1 is intentionally not used. Success = all official test repetitions pass.
"""
from __future__ import annotations

import json
import signal
import time

from leetcode_runtime import config, executor

SYSTEM = (
    "You are a coding agent solving LeetCode problems.\\n"
    "You MUST explicitly decide two things and report them as the tool call:\\n"
    "  1) sleep xx  - the wait duration in seconds before running the tests;\\n"
    "  2) python yy - the complete Python solution.\\n"
    "Use the two tools in order:\\n"
    "1) propose_plan(sleep_seconds=xx, python_code=yy)\\n"
    "2) execute_plan()\\n"
    "There is no prescribed sleep range or default. Choose it yourself.\\n"
    "If the problem has no valid answer, return None (not an empty list/dict).\\n"
    "After execute_plan returns, reply with a short final answer."
)
FENCE = chr(96) * 3


def _load_llm():
    from langchain_deepseek import ChatDeepSeek
    return ChatDeepSeek(model=config.model_name(), api_key=config.api_key(),
                        api_base=config.base_url(), temperature=0,
                        timeout=30, max_retries=2)


def build_agent(problem, state, use_llm=True):
    from langgraph.prebuilt import create_react_agent

    def propose_plan(sleep_seconds: float, python_code: str) -> str:
        """Record the agent's sleep decision and complete Python solution."""
        try:
            seconds = float(sleep_seconds)
        except (TypeError, ValueError):
            seconds = 5.0
        state["plan"] = {"sleep_seconds": max(0.0, seconds), "code": str(python_code)}
        return "plan accepted: sleep=%ss, code_chars=%d" % (
            state["plan"]["sleep_seconds"], len(state["plan"]["code"]))

    def execute_plan() -> str:
        """Run the official tests K times and measure total runtime."""
        plan = state.get("plan")
        if not plan:
            return "error: no plan; call propose_plan first"
        code = plan["code"]
        k = int(problem.get("repeat_k") or 300)
        script = (problem.get("prompt") or "") + chr(10) + code + chr(10) + \
                 (problem.get("test") or "") + chr(10) + \
                 "for _ in range(%d):" % k + chr(10) + \
                 "    check(%s)" % (problem.get("entry_point") or "")
        job = executor.run_python_job(script, timeout_s=180.0)
        result = executor.wait_for_job(job, "native", 0.0,
                                          {"source": "native", "p50_ms": None},
                                          timeout_s=180.0)
        result["repeat_k"] = k
        result["requested_sleep_ms"] = int(round(plan["sleep_seconds"] * 1000))
        result["success"] = result.get("status") == "completed"
        state["last"] = result
        return json.dumps({"status": result["status"], "success": result["success"],
                           "runtime_ms": result.get("actual_runtime_ms"),
                           "returncode": result.get("returncode")})

    llm = _load_llm() if use_llm else None
    return create_react_agent(llm, [propose_plan, execute_plan], prompt=SYSTEM)


def task_prompt(problem):
    return ("Solve this LeetCode problem.\\n\\n" +
            str(problem.get("problem_description") or "") +
            "\\n\\nStarter code:\\n" + str(problem.get("starter_code") or ""))


def _extract_code(text):
    if FENCE + "python" in text:
        return text.split(FENCE + "python", 1)[1].split(FENCE, 1)[0].strip()
    if FENCE in text:
        return text.split(FENCE, 1)[1].split(FENCE, 1)[0].strip()
    return str(text).strip()


class AgentTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise AgentTimeout("trial exceeded 180 seconds")


def run_task(problem, state, use_llm=True):
    agent = build_agent(problem, state, use_llm=use_llm)
    started = time.time()
    limit = float(problem.get("trial_timeout_s") or 180.0)
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        result = agent.invoke({"messages": [("user", task_prompt(problem))]},
                              config={"recursion_limit": 20})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    wall_ms = (time.time() - started) * 1000.0
    tokens = 0
    steps = 0
    for message in result.get("messages", []):
        steps += 1
        usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, dict):
            tokens += int(usage.get("total_tokens") or 0)
    last = state.get("last")
    if last is None:
        text = ""
        for message in reversed(result.get("messages", [])):
            if getattr(message, "content", None):
                text = str(message.content)
                break
        code = _extract_code(text)
        k = int(problem.get("repeat_k") or 300)
        script = (problem.get("prompt") or "") + chr(10) + code + chr(10) + \
                 (problem.get("test") or "") + chr(10) + \
                 "for _ in range(%d):" % k + chr(10) + \
                 "    check(%s)" % (problem.get("entry_point") or "")
        job = executor.run_python_job(script, timeout_s=180.0)
        last = executor.wait_for_job(job, "native", 0.0,
                                        {"source": "native", "p50_ms": None},
                                        timeout_s=180.0)
        last["repeat_k"] = k
        last["requested_sleep_ms"] = 5000
        last["success"] = last.get("status") == "completed"
        state["last"] = last
        state["protocol"] = "fallback_text"
    return {"code": (state.get("plan") or {}).get("code", ""),
            "wall_ms": round(wall_ms, 1), "tokens": tokens, "steps": steps, **last}
