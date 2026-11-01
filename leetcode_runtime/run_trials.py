"""Run the LeetCode repeat-execution experiment.

Each trial asks the agent for a solution, executes the official tests K times,
and records the measured runtime plus counterfactual polling policies.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from hgre_core.paths import ROOT
from leetcode_runtime import config, dataset, llm_prior, predictor

EXP_CONFIGS = [
    ("exp_50_x2", 50.0, 2.0),
    ("exp_100_x2", 100.0, 2.0),
    ("exp_200_x2", 200.0, 2.0),
    ("exp_100_x1.5", 100.0, 1.5),
    ("exp_100_x3", 100.0, 3.0),
    ("exp_500_x2", 500.0, 2.0),
]


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_trial_isolated(problem, history, k, timeout_s):
    payload = json.dumps({
        "problem": problem,
        "history": history,
        "k": k,
        "trial_timeout_s": timeout_s,
    }, ensure_ascii=False)
    proc = subprocess.Popen(
        [sys.executable, "-m", "leetcode_runtime.worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        cwd=str(ROOT),
    )
    try:
        out, err = proc.communicate(payload, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            out, err = proc.communicate(timeout=5)
        except Exception:
            out, err = "", ""
        return None, "trial_timeout"
    for line in (out or "").splitlines():
        if line.startswith("__TRIAL_RESULT__"):
            data = json.loads(line[len("__TRIAL_RESULT__"):])
            return (data["rec"], None) if data.get("ok") else (None, data.get("error") or "agent_failed")
    return None, "worker_no_result: " + (err or "")[-200:]


def simulate(chosen, actual, poll_ms=config.POLL_MS):
    if chosen is None or actual is None:
        return {"chosen_ms": chosen, "delay_ms": None, "queries": None}
    if chosen >= actual:
        return {"chosen_ms": chosen, "delay_ms": chosen - actual, "queries": 1}
    remaining = actual - chosen
    polls = int(math.ceil(remaining / poll_ms))
    return {"chosen_ms": chosen, "delay_ms": polls * poll_ms - remaining, "queries": 1 + polls}


def exp_backoff(actual, start, factor, max_interval=10000.0, max_checks=20):
    checkpoints, wait, total = [], float(start), 0.0
    while total < actual and len(checkpoints) < max_checks:
        checkpoints.append(wait)
        total += wait
        wait = min(max_interval, wait * factor)
    while total < actual and len(checkpoints) < 100000:
        checkpoints.append(max_interval)
        total += max_interval
    return {"chosen_ms": total, "delay_ms": max(0.0, total - actual), "queries": len(checkpoints)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="0 runs all problems")
    ap.add_argument("--reps", type=int, default=config.DEFAULT_REPS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(config.RESULTS))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trials_path = out / "trials.jsonl"
    strategies_path = out / "strategies.jsonl"
    failures_path = out / "failures.jsonl"

    problems = dataset.load_problems()
    rng = random.Random(args.seed)
    if args.sample:
        problems = rng.sample(problems, min(args.sample, len(problems)))
        problems.sort(key=lambda row: int(row["question_id"]))

    done = set()
    history = defaultdict(list)
    if trials_path.exists():
        for line in open(trials_path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((row["task_id"], int(row["rep"])))
            if row.get("success") and row.get("actual_runtime_ms") is not None:
                history[row["task_id"]].append({
                    "T_ms": float(row["actual_runtime_ms"]),
                    "k": int(row["k"]),
                })

    total = len(problems) * args.reps
    print(f"START problems={len(problems)} reps={args.reps} total={total} done={len(done)}", flush=True)
    start = time.time()
    done_count = len(done)
    failed = 0

    for problem in problems:
        task = problem["task_id"]
        for rep in range(1, args.reps + 1):
            if (task, rep) in done:
                continue
            k = rng.randint(config.K_MIN, config.K_MAX)
            p = dict(problem)
            p["repeat_k"] = k
            p["trial_timeout_s"] = config.TRIAL_TIMEOUT_S
            trial_start = time.time()
            rec, err = run_trial_isolated(p, history[task], k, config.TRIAL_TIMEOUT_S)
            if err:
                append_jsonl(failures_path, {"task_id": task, "rep": rep, "stage": "trial", "error": err[:400]})
                failed += 1
                continue
            actual = rec.get("actual_runtime_ms")
            if not rec.get("success") or actual is None:
                append_jsonl(failures_path, {"task_id": task, "rep": rep, "stage": "execution",
                                             "status": rec.get("status")})
                failed += 1
                continue

            code = rec.get("code") or ""
            prior = list(history[task])
            llm_ms = llm_prior.estimate(problem, code, k)
            m2 = predictor.compute(prior, k) if prior else None
            if prior and m2 is not None:
                m2_ms, m2_method, m2_source = m2["final_p50_ms"], m2["chosen_method"], "history"
            else:
                m2_ms, m2_method, m2_source = llm_ms, "llm_prior", "cold"

            trial = {
                "task_id": task,
                "question_id": problem.get("question_id"),
                "difficulty": problem.get("difficulty"),
                "rep": rep,
                "k": k,
                "success": True,
                "actual_runtime_ms": actual,
                "agent_wall_ms": rec.get("wall_ms"),
                "tokens": rec.get("tokens"),
                "steps": rec.get("steps"),
                "requested_sleep_ms": rec.get("requested_sleep_ms"),
                "llm_prior_ms": llm_ms,
                "m2_source": m2_source,
                "m2_method": m2_method,
                "pred_m2_ms": m2_ms,
                "m2_candidates": None if m2 is None else m2["candidates"],
                "m2_candidate_means": None if m2 is None else m2["candidate_means"],
                "history_n": len(prior),
            }
            policies = {"m2": m2_ms, "llm_prior": llm_ms, "llm_sleep": rec.get("requested_sleep_ms")}
            policy_rows = {}
            for name, chosen in policies.items():
                if name == "llm_sleep":
                    sim = simulate(chosen, actual)
                else:
                    sim = simulate(chosen, actual)
                policy_rows[name] = sim
                append_jsonl(strategies_path, {
                    "task_id": task, "difficulty": problem.get("difficulty"),
                    "rep": rep, "k": k, "policy": name,
                    "actual_runtime_ms": actual, **sim,
                })
            for name, wait, factor in EXP_CONFIGS:
                sim = exp_backoff(actual, wait, factor)
                policy_rows[name] = sim
                append_jsonl(strategies_path, {
                    "task_id": task, "difficulty": problem.get("difficulty"),
                    "rep": rep, "k": k, "policy": name,
                    "actual_runtime_ms": actual, **sim,
                })
            trial["policies"] = policy_rows
            append_jsonl(trials_path, trial)
            if rec.get("success") and actual is not None:
                history[task].append({"T_ms": actual, "k": k})
            done_count += 1
            elapsed = time.time() - start
            print(f"[{done_count}/{total}] {task} rep={rep} k={k} T={actual:.1f}ms", flush=True)

    print(f"FINISHED done={done_count} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
