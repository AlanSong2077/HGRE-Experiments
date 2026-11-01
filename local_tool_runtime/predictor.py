"""HGRE adapter for controlled local-tool executions."""
from __future__ import annotations

from hgre_core.predictor import predict


def compute(prior, target_size=None):
    rows = [{"T_ms": row["T_ms"], "size": row.get("size")} for row in prior]
    return predict(rows, target_size=target_size, command_class="download")
