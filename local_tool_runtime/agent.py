"""ReAct agent for one controlled local-tool execution."""
from __future__ import annotations

import json
import time

from local_tool_runtime import config, executor

SYSTEM = (
    "You are a shell-command agent. The local workload command is fixed by the experiment. "
    "Choose only how long to sleep between checks. "
    "Call propose_plan(sleep_seconds=N) and execute_plan(); repeat while the job is still running. "
    "Rely on execute_plan output; do not invent completion."
)


def _load_llm():
    from langchain_deepseek import ChatDeepSeek
    return ChatDeepSeek(
        model=config.model_name(),
        api_key=config.api_key(),
        api_base=config.base_url(),
        temperature=0,
        timeout=30,
        max_retries=2,
    )


def build_agent(case, state):
    from langgraph.prebuilt import create_react_agent

    def propose_plan(sleep_seconds: float, zsh_command: str = "") -> str:
        try:
            seconds = float(sleep_seconds)
        except (TypeError, ValueError):
            seconds = 5.0
        state["plan"] = {"sleep_seconds": max(0.0, seconds)}
        return json.dumps({"accepted": True, "sleep_s": state["plan"]["sleep_seconds"]})

    def execute_plan() -> str:
        plan = state.get("plan")
        if not plan:
            return json.dumps({"status": "error", "error": "call propose_plan first"})
        job = state.get("job")
        if job is None:
            command = case.get("_fixed_command") or ""
            if not command:
                return json.dumps({"status": "error", "error": "fixed command is missing"})
            workdir = case.get("_workdir") or ""
            job = executor.start_job(
                command,
                workdir,
                timeout_s=float(case.get("exec_timeout_s") or 1200.0),
            )
            state["job"] = job
            state["command"] = command
            state["checks"] = 0
            state["sleeps_ms"] = []
            state["check_offsets_ms"] = []

        seconds = plan["sleep_seconds"]
        executor.sleep_seconds(seconds)
        state["sleeps_ms"].append(int(round(seconds * 1000)))
        state["checks"] += 1
        state["check_offsets_ms"].append(round((time.time() - job.started_at) * 1000.0, 1))

        rc = job.poll()
        elapsed = job.elapsed_ms()
        timed_out = job.timed_out or elapsed >= job.timeout_s * 1000.0
        if rc is None and timed_out:
            job.kill()
            rc = job.poll()
            result = {
                "status": "timeout",
                "success": False,
                "actual_runtime_ms": round(job.elapsed_ms(), 1),
                "zsh_command": state.get("command", ""),
                "requested_sleep_ms": sum(state["sleeps_ms"]),
                "sleeps_ms": list(state["sleeps_ms"]),
                "check_offsets_ms": list(state["check_offsets_ms"]),
                "checks": state["checks"],
                "workdir_bytes_before": job.workdir_bytes_before,
                "workdir_delta_bytes": job.workdir_delta_bytes(),
                "job_log_dir": str(job.log_dir),
                "stdout_tail": job.tail(job.stdout_path),
                "stderr_tail": job.tail(job.stderr_path),
            }
            state["last"] = result
            return json.dumps({"status": "timeout", "runtime_ms": result["actual_runtime_ms"]})
        if rc is None:
            return json.dumps({
                "status": "still_running",
                "elapsed_ms": round(elapsed, 1),
                "checks": state["checks"],
            })

        job.ended_at = job.ended_at or time.time()
        success = rc == 0
        result = {
            "status": "completed" if success else "failed",
            "success": bool(success),
            "returncode": rc,
            "actual_runtime_ms": round(job.elapsed_ms(), 1),
            "zsh_command": state.get("command", ""),
            "requested_sleep_ms": sum(state["sleeps_ms"]),
            "sleeps_ms": list(state["sleeps_ms"]),
            "check_offsets_ms": list(state["check_offsets_ms"]),
            "checks": state["checks"],
            "workdir_bytes_before": job.workdir_bytes_before,
            "workdir_delta_bytes": job.workdir_delta_bytes(),
            "job_log_dir": str(job.log_dir),
            "stdout_tail": job.tail(job.stdout_path),
            "stderr_tail": job.tail(job.stderr_path),
        }
        state["last"] = result
        return json.dumps({"status": result["status"], "returncode": rc,
                           "runtime_ms": result["actual_runtime_ms"]})

    return create_react_agent(_load_llm(), [propose_plan, execute_plan], prompt=SYSTEM)


def task_prompt(case):
    return (
        "Command type: %s\nWorkload: %s\n\nCurrent working directory: %s\n"
        "Start the fixed command as a background job, sleep, check, and repeat until done."
        % (case.get("command_type"),
           json.dumps(case.get("source") or {}, ensure_ascii=False),
           case.get("_workdir") or "")
    )


def run_case(case, state):
    agent = build_agent(case, state)
    started = time.time()
    result = None
    try:
        result = agent.invoke(
            {"messages": [("user", task_prompt(case))]},
            config={"recursion_limit": 120},
        )
    finally:
        job = state.get("job")
        if job is not None:
            job.poll()
            if job.proc.poll() is None:
                job.kill()
            job.close()
    wall_ms = (time.time() - started) * 1000.0
    last = state.get("last") or {
        "status": "failed",
        "success": False,
        "actual_runtime_ms": None,
        "zsh_command": state.get("command", ""),
        "requested_sleep_ms": sum(state.get("sleeps_ms") or []),
        "sleeps_ms": state.get("sleeps_ms") or [],
        "checks": state.get("checks", 0),
        "stdout_tail": "",
        "stderr_tail": "",
    }
    tokens = steps = 0
    for message in result.get("messages", []):
        steps += 1
        usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, dict):
            tokens += int(usage.get("total_tokens") or 0)
    return {"code": "", "wall_ms": round(wall_ms, 1), "tokens": tokens, "steps": steps, **last}
