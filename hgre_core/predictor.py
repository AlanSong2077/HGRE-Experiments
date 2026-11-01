"""Predict runtime from recent same-family executions."""
from __future__ import annotations

import math

from hgre_core import api, history as m2


def predict(prior, target_size=None, command_class="code_exec"):
    """prior: list of {"T_ms": float, "size": float|None}."""
    if not prior:
        return None
    records = []
    for i, row in enumerate(prior):
        size = row.get("size")
        records.append({
            "duration_ms": float(row["T_ms"]),
            "finishedAt": 1000 + i,
            "background": True,
            "command_class": command_class,
            "size": float(size) if size is not None else None,
        })
    target = {"startedAt": 10 ** 12, "command_class": command_class}
    if target_size is not None:
        target["size"] = float(target_size)
    inp = api.build_m2_input(records, target, min_duration_ms=1.0)
    if inp is None:
        return None

    predictor = m2.HistoryBasedPredictor()
    candidates, means = {}, {}
    for method in m2.METHODS:
        try:
            pred = predictor.predict(inp) if method == "auto" else predictor.predict(inp, method=method)
            if math.isfinite(float(pred.p50_ms)):
                candidates[method] = round(float(pred.p50_ms), 3)
                means[method] = round(float(pred.mean_ms), 3)
        except Exception:
            continue
    if not candidates:
        return None
    if "auto" in candidates:
        chosen, chosen_method = candidates["auto"], "auto"
    else:
        chosen_method = next(iter(candidates))
        chosen = candidates[chosen_method]
    return {
        "final_p50_ms": chosen,
        "chosen_method": chosen_method,
        "candidates": candidates,
        "candidate_means": means,
    }
