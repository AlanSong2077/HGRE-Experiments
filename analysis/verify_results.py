"""Recompute the paper's main tables from the packaged trial logs."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/alansong/Desktop/FORDEEPSEEK/HGRE-Experiments")
sys.path.insert(0, str(ROOT))

from hgre_core.paths import LEETCODE_RESULTS, LOCAL_TOOL_RESULTS


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def ape(pred, actual):
    return 100.0 * abs(pred - actual) / actual


def p90(values):
    return float(np.percentile(values, 90))


def main():
    leet = load(LEETCODE_RESULTS / "trials.jsonl")
    local = load(LOCAL_TOOL_RESULTS / "trials.jsonl")

    qids = sorted({row["question_id"] for row in leet})[:543]
    qset = set(qids)
    leet = [row for row in leet if row["question_id"] in qset]
    warm = [row for row in leet if row["m2_source"] == "history"]

    checks = []

    def check(label, actual, expected, tol):
        ok = abs(actual - expected) <= tol
        checks.append(ok)
        print(f"{label:38s} actual={actual:10.2f} expected={expected:10.2f} {'PASS' if ok else 'FAIL'}")

    print("=== SAMPLE ===")
    check("LeetCode trials", len(leet), 2622, 0)
    check("LeetCode problems", len(qset), 543, 0)
    check("cold-start", sum(row["m2_source"] == "cold" for row in leet), 549, 0)
    check("warm-start", len(warm), 2073, 0)
    check("full five-round problems", sum(len([x for x in leet if x["question_id"] == q]) == 5 for q in qset), 504, 0)
    check("local-tool trials", len(local), 120, 0)

    print("\n=== HEADLINE ===")
    h = np.array([ape(row["pred_m2_ms"], row["actual_runtime_ms"]) for row in warm])
    l = np.array([ape(row["llm_prior_ms"], row["actual_runtime_ms"]) for row in warm])
    check("HGRE warm MdAPE", float(np.median(h)), 5.0, 0.1)
    check("HGRE warm P90 APE", p90(h), 52.4, 0.2)
    check("LLM warm MdAPE", float(np.median(l)), 92.4, 0.1)
    check("LLM warm P90 APE", p90(l), 2696.9, 1.0)

    byq = {}
    for row in warm:
        byq.setdefault(row["question_id"], []).append(row)
    better = 0
    reductions = []
    for rows in byq.values():
        a = statistics.mean(ape(x["pred_m2_ms"], x["actual_runtime_ms"]) for x in rows)
        b = statistics.mean(ape(x["llm_prior_ms"], x["actual_runtime_ms"]) for x in rows)
        better += a < b
        if b:
            reductions.append((b - a) / b * 100)
    check("problems where HGRE is better", better, 522, 0)
    check("median per-problem reduction", statistics.median(reductions), 92.2, 0.1)

    print("\n=== TABLE I ===")
    for rep in (2, 3, 4, 5):
        for level in ("Easy", "Medium", "Hard"):
            rows = [x for x in warm if x["rep"] == rep and x["difficulty"] == level]
            hh = np.median([ape(x["pred_m2_ms"], x["actual_runtime_ms"]) for x in rows])
            ll = np.median([ape(x["llm_prior_ms"], x["actual_runtime_ms"]) for x in rows])
            print(f"R{rep} {level:6s} n={len(rows):4d} HGRE={hh:5.1f} LLM={ll:6.1f}")

    print("\n=== TABLE II / III ===")
    groups = {
        "Interpolation": [x for x in local if x["size_relation"] == "interpolation" and x["history_n"] > 0 and x["rep"] == 1],
        "Upward": [x for x in local if x["size_relation"] == "extrapolation_high"],
        "Downward": [x for x in local if x["size_relation"] == "extrapolation_low"],
        "Repeat": [x for x in local if x["rep"] in (2, 3)],
    }
    for name, rows in groups.items():
        hh = np.median([ape(x["pred_m2_ms"], x["actual_runtime_ms"]) for x in rows])
        ll = np.median([ape(x["llm_prior_ms"], x["actual_runtime_ms"]) for x in rows])
        print(f"{name:13s} n={len(rows):3d} HGRE={hh:7.1f} LLM={ll:7.1f}")

    print("\n=== POLLING ===")
    for name, rows in [("LeetCode warm", warm),
                       ("Local new", [x for x in local if x["rep"] == 1 and x["history_n"] > 0]),
                       ("Local repeat", [x for x in local if x["rep"] in (2, 3)])]:
        print(name, "n=", len(rows))
        for policy, label in (("m2", "HGRE"), ("llm_prior", "LLM"), ("exp_100_x2", "Exp.")):
            delays = [x["policies"][policy]["delay_ms"] for x in rows]
            checks_count = [x["policies"][policy]["queries"] for x in rows]
            print(f"  {label:4s} mean={np.mean(delays):7.1f} p90={np.percentile(delays,90):7.1f} checks={np.mean(checks_count):.2f}")

    print("\nRESULT:", "PASS" if all(checks) else "FAIL")


if __name__ == "__main__":
    main()
