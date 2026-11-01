"""Run the controlled local-tool experiment (csv/jsonl/gzip/matrix)."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from hgre_core.paths import ROOT
from local_tool_runtime import config, generate_data, llm_prior, predictor

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


def simulate(chosen, actual, poll=100):
    if chosen is None or actual is None:
        return {"chosen_ms": chosen, "delay_ms": None, "queries": None}
    if chosen >= actual:
        return {"chosen_ms": chosen, "delay_ms": chosen - actual, "queries": 1}
    remaining = actual - chosen
    polls = int(math.ceil(remaining / poll))
    return {"chosen_ms": chosen, "delay_ms": polls * poll - remaining, "queries": 1 + polls}


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


def agent_lag(rec, actual):
    for offset in rec.get("check_offsets_ms") or []:
        offset = float(offset)
        if offset >= actual:
            return round(offset - actual, 1)
    return None


def simulate_agent_sleep(rec, actual):
    offsets = [float(x) for x in (rec.get("check_offsets_ms") or [])]
    if not offsets:
        total = 0.0
        for ms in rec.get("sleeps_ms") or []:
            total += float(ms)
            offsets.append(total)
    if not offsets:
        return simulate(rec.get("requested_sleep_ms"), actual)
    for i, offset in enumerate(offsets):
        if offset >= actual:
            return {"chosen_ms": round(offset, 1), "delay_ms": round(offset - actual, 1),
                    "queries": i + 1, "detection_lag_ms": round(offset - actual, 1), "completed": True}
    return {"chosen_ms": round(offsets[-1], 1), "delay_ms": None,
            "queries": len(offsets), "detection_lag_ms": None, "completed": False}


def cleanup_trial(workdir, rec=None):
    try:
        if rec and rec.get("job_log_dir"):
            shutil.rmtree(rec["job_log_dir"], ignore_errors=True)
    except Exception:
        pass
    shutil.rmtree(workdir, ignore_errors=True)


def run_worker(case, timeout_s=1320.0):
    payload = json.dumps({"case": case}, ensure_ascii=False)
    proc = subprocess.Popen(
        [sys.executable, "-m", "local_tool_runtime.worker"],
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
        return None, "worker_timeout"
    for line in (out or "").splitlines():
        if line.startswith("__TRIAL_RESULT__"):
            data = json.loads(line[len("__TRIAL_RESULT__"):])
            return (data["rec"], None) if data.get("ok") else (None, data.get("error") or "worker_failed")
    return None, "no_result:" + (err or "")[-300:]


def main():
    generate_data.ensure()
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=config.DEFAULT_REPS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--case-ids", default="", help="comma-separated case ids")
    args = ap.parse_args()

    out = config.RESULTS
    out.mkdir(parents=True, exist_ok=True)
    trials_path = out / "trials.jsonl"
    strategies_path = out / "strategies.jsonl"
    failures_path = out / "failures.jsonl"

    cases = json.loads(config.CASES.read_text(encoding="utf-8"))
    if args.case_ids:
        wanted = {x.strip() for x in args.case_ids.split(",") if x.strip()}
        cases = [c for c in cases if c["case_id"] in wanted]
    groups = {}
    for case in cases:
        groups.setdefault(case["task_id"], []).append(case)
    ordered = []
    for task_id, versions in groups.items():
        versions = list(versions)
        random.Random("%d|%s" % (args.seed, task_id)).shuffle(versions)
        for index, case in enumerate(versions, 1):
            case = dict(case)
            case["execution_order"] = index
            ordered.append(case)
    cases = ordered

    config.RUNTIME.mkdir(parents=True, exist_ok=True)
    done = set()
    history = {}
    if trials_path.exists():
        for line in open(trials_path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((row["case_id"], int(row["rep"])))
            history.setdefault(row["task_id"], []).append({
                "T_ms": float(row["actual_runtime_ms"]),
                "size": row.get("predicted_input_size"),
            })

    total = len(cases) * args.reps
    print(f"START cases={len(cases)} reps={args.reps} total={total} done={len(done)}", flush=True)
    start = time.time()
    done_count = len(done)
    failed = 0

    for case in cases:
        case_id = case["case_id"]
        for rep in range(1, args.reps + 1):
            if (case_id, rep) in done:
                continue
            workdir = Path(tempfile.mkdtemp(prefix="hgre_local_", dir=str(config.RUNTIME)))
            payload = dict(case)
            payload["_workdir"] = str(workdir)
            payload["_fixed_command"] = case.get("fixed_command")
            predicted = case.get("input_size")
            predicted_source = case.get("input_size_source") or "case_input_size"

            rec, err = run_worker(payload, timeout_s=1320.0)
            if err:
                append_jsonl(failures_path, {"case_id": case_id, "rep": rep, "stage": "worker", "error": err[:400]})
                failed += 1
                cleanup_trial(workdir)
                continue
            actual = rec.get("actual_runtime_ms")
            if not rec.get("success") or actual is None:
                append_jsonl(failures_path, {"case_id": case_id, "rep": rep, "stage": "execution",
                                             "status": rec.get("status")})
                failed += 1
                cleanup_trial(workdir, rec)
                continue

            command = rec.get("zsh_command") or ""
            prior = list(history.get(case["task_id"], []))
            llm_ms, prior_source, prior_cache_hit = llm_prior.estimate(case, command)
            m2 = predictor.compute(prior, target_size=predicted) if prior else None
            sizes = [x.get("size") for x in prior if x.get("size") is not None]
            if predicted is not None and sizes:
                if predicted > max(sizes):
                    relation = "extrapolation_high"
                elif predicted < min(sizes):
                    relation = "extrapolation_low"
                else:
                    relation = "interpolation"
            else:
                relation = "unknown"
            if prior and m2:
                m2_ms, m2_method, m2_source = m2["final_p50_ms"], m2["chosen_method"], "history"
            else:
                m2_ms, m2_method, m2_source = llm_ms, "llm_prior", "cold"

            trial = {
                "case_id": case_id,
                "task_id": case.get("task_id"),
                "command_type": case.get("command_type"),
                "version": case.get("version"),
                "version_index": case.get("version_index"),
                "execution_order": case.get("execution_order"),
                "rep": rep,
                "source": case.get("source"),
                "command": command,
                "success": True,
                "actual_runtime_ms": actual,
                "agent_wall_ms": rec.get("wall_ms"),
                "tokens": rec.get("tokens"),
                "steps": rec.get("steps"),
                "requested_sleep_ms": rec.get("requested_sleep_ms"),
                "sleeps_ms": rec.get("sleeps_ms"),
                "check_offsets_ms": rec.get("check_offsets_ms"),
                "checks": rec.get("checks"),
                "returncode": rec.get("returncode"),
                "predicted_input_size": predicted,
                "predicted_input_size_unit": case.get("size_unit"),
                "predicted_input_source": predicted_source,
                "actual_workdir_bytes": rec.get("workdir_delta_bytes"),
                "llm_prior_ms": llm_ms,
                "prior_source": prior_source,
                "prior_fallback": prior_source == "constant_fallback",
                "prior_cache_hit": prior_cache_hit,
                "m2_source": m2_source,
                "m2_method": m2_method,
                "pred_m2_ms": m2_ms,
                "m2_candidates": None if not m2 else m2["candidates"],
                "m2_candidate_means": None if not m2 else m2["candidate_means"],
                "m2_ape": abs(m2_ms - actual) / actual if actual else None,
                "history_n": len(prior),
                "size_relation": relation,
                "agent_queries": rec.get("checks"),
                "agent_detection_lag_ms": agent_lag(rec, actual),
            }

            policies = {}
            for name, chosen in (("m2", m2_ms), ("llm_prior", llm_ms)):
                sim = simulate(chosen, actual)
                policies[name] = sim
                append_jsonl(strategies_path, {"case_id": case_id, "task_id": case.get("task_id"),
                                               "command_type": case.get("command_type"),
                                               "version": case.get("version"), "rep": rep,
                                               "policy": name, "actual_runtime_ms": actual, **sim})
            sleep_sim = simulate_agent_sleep(rec, actual)
            policies["native_sleep"] = sleep_sim
            append_jsonl(strategies_path, {"case_id": case_id, "task_id": case.get("task_id"),
                                           "command_type": case.get("command_type"),
                                           "version": case.get("version"), "rep": rep,
                                           "policy": "native_sleep",
                                           "actual_runtime_ms": actual, **sleep_sim})
            for name, wait, factor in EXP_CONFIGS:
                sim = exp_backoff(actual, wait, factor)
                policies[name] = sim
                append_jsonl(strategies_path, {"case_id": case_id, "task_id": case.get("task_id"),
                                               "command_type": case.get("command_type"),
                                               "version": case.get("version"), "rep": rep,
                                               "policy": name,
                                               "actual_runtime_ms": actual, **sim})
            trial["policies"] = policies
            append_jsonl(trials_path, trial)
            history.setdefault(case["task_id"], []).append({"T_ms": actual, "size": predicted})
            done_count += 1
            elapsed = time.time() - start
            print(f"[{done_count}/{total}] {case_id} rep={rep} T={actual:.1f}ms", flush=True)
            cleanup_trial(workdir, rec)

    print(f"FINISHED done={done_count} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
