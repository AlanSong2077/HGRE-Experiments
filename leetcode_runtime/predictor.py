"""HGRE adapter for repeated LeetCode executions."""
from __future__ import annotations

from hgre_core.predictor import predict


def compute(prior, k):
    rows = [{"T_ms": row["T_ms"], "size": row.get("k", row.get("size"))} for row in prior]
    return predict(rows, target_size=k, command_class="code_exec")
