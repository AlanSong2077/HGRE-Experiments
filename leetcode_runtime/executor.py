"""Execute a LeetCode trial and measure its actual runtime."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


WRAPPER = Path("/Users/alansong/Desktop/FORDEEPSEEK/HGRE-Experiments/leetcode_runtime/exec_wrapper.py")
POLL_MS = 100
MAX_WAIT_MS = 30_000


def run_python_job(script_text, timeout_s=30.0):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
    tmp.write(script_text)
    tmp.close()
    proc = subprocess.Popen([sys.executable, str(WRAPPER), tmp.name],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {"proc": proc, "path": tmp.name, "timeout_s": timeout_s}


def _parse_timing(stderr):
    for line in reversed((stderr or "").splitlines()):
        if line.startswith("__WP_TIMING__"):
            try:
                return json.loads(line[len("__WP_TIMING__"):])
            except Exception:
                return None
    return None


def wait_for_job(job, arm, requested_sleep_s, prediction, poll_ms=POLL_MS, timeout_s=None):
    proc = job["proc"]
    timeout_s = timeout_s or job.get("timeout_s") or 30.0
    requested_ms = int(round(requested_sleep_s * 1000))
    if arm == "native" or prediction.get("p50_ms") is None:
        chosen_ms = requested_ms
        source = "native" if arm == "native" else "fallback_native"
        fallback = None if arm == "native" else prediction.get("fallback_reason")
    else:
        chosen_ms = max(1, min(MAX_WAIT_MS, int(round(float(prediction["p50_ms"])))))
        source = prediction.get("source") or arm
        fallback = None
    # explicit check plan: first check at the chosen wait, then poll until timeout
    max_check_ms = max(chosen_ms, int(timeout_s * 1000))
    checkpoints = [chosen_ms] + list(range(chosen_ms + poll_ms, max_check_ms + 1, poll_ms))
    wait_started = time.time()
    detected_at = None
    checks = 0
    offsets = []
    for checkpoint in checkpoints:
        if proc.poll() is not None:
            detected_at = time.time()
            break
        elapsed_ms = (time.time() - wait_started) * 1000.0
        delta_ms = checkpoint - elapsed_ms
        if delta_ms > 0:
            time.sleep(delta_ms / 1000.0)
        checks += 1
        offsets.append(round((time.time() - wait_started) * 1000.0, 1))
        if proc.poll() is not None:
            detected_at = time.time()
            break
        if (time.time() - wait_started) > timeout_s:
            break
    if proc.poll() is None:
        if (time.time() - wait_started) > timeout_s:
            proc.kill()
    try:
        out, err = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    if detected_at is None:
        detected_at = time.time()
    wait_ms = (detected_at - wait_started) * 1000.0
    timing = _parse_timing(err)
    if timing and timing.get("end"):
        actual_runtime_ms = max(0.0, (float(timing["end"]) - float(timing["start"])) * 1000.0)
        detection_lag_ms = max(0.0, (detected_at - float(timing["end"])) * 1000.0)
        actual_started = float(timing["start"])
        actual_ended = float(timing["end"])
    else:
        actual_runtime_ms = None
        detection_lag_ms = None
        actual_started = None
        actual_ended = None
    rc = proc.returncode
    if actual_ended is None:
        status = "timeout" if (detected_at - wait_started) >= timeout_s or rc is None else "failed"
    elif (detected_at - wait_started) >= timeout_s and rc is None:
        status = "timeout"
    elif rc == 0:
        status = "completed"
    else:
        status = "failed"
    return {
        "requested_sleep_ms": requested_ms,
        "chosen_wait_ms": chosen_ms,
        "source": source,
        "fallback_reason": fallback,
        "pred_p50_ms": prediction.get("p50_ms"),
        "pred_p95_ms": prediction.get("p95_ms"),
        "pred_method": prediction.get("method"),
        "wait_ms": round(wait_ms, 1),
        "actual_runtime_ms": None if actual_runtime_ms is None else round(actual_runtime_ms, 1),
        "detection_lag_ms": None if detection_lag_ms is None else round(detection_lag_ms, 1),
        "checks": checks,
        "check_offsets_ms": offsets,
        "status": status,
        "returncode": rc,
        "success": status == "completed",
        "stdout_tail": (out or "")[-300:],
        "stderr_tail": (err or "")[-500:],
        "actual_started_epoch": actual_started,
        "actual_ended_epoch": actual_ended,
    }
