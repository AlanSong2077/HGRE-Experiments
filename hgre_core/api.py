"""M2-only adapter from online history records to PredictorInput.

M1 has been removed from the project. This module contains only the history
adapter used by the repeat-only experiment.
"""
from __future__ import annotations

from hgre_core import config
from hgre_core import history as m2

MIN_DURATION_MS = float(config.EXP.get("min_duration_ms", 100.0))
HISTORY_WINDOW = int(config.EXP.get("history_window", 5))


def _num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _past(records, started_at, command_class=None, min_duration_ms=None):
    out = []
    floor = MIN_DURATION_MS if min_duration_ms is None else float(min_duration_ms)
    for record in records or []:
        if not isinstance(record, dict) or record.get("background") is False:
            continue
        duration = _num(record.get("duration_ms"))
        finished = _num(record.get("finishedAt"))
        if duration is None or finished is None or duration < floor:
            continue
        if started_at is not None and finished > started_at:
            continue
        if command_class is not None and str(record.get("command_class") or "") != command_class:
            continue
        out.append(record)
    out.sort(key=lambda r: float(r.get("finishedAt")))
    return out


def _envelope(pred, source):
    return {"mean_ms": float(pred.mean_ms), "p50_ms": float(pred.p50_ms),
            "p95_ms": float(pred.p95_ms), "method": str(pred.method),
            "n_history": int(getattr(pred, "n_history", 0)),
            "confidence": float(getattr(pred, "confidence", 0.0)),
            "source": source}


def build_m2_input(history, record, min_duration_ms=None):
    started = _num(record.get("startedAt"))
    command_class = str(record.get("command_class") or "")
    rows = _past(history, started, command_class=command_class,
                 min_duration_ms=min_duration_ms)[-HISTORY_WINDOW:]
    if not rows:
        return None
    target_size = _num(record.get("size"))
    sizes_ok = target_size is not None and target_size > 0
    items = []
    for row in rows:
        size = _num(row.get("size"))
        entry = {"runtime_ms": float(row["duration_ms"])}
        if sizes_ok and size is not None and size > 0:
            entry["input_size"] = float(size)
        else:
            sizes_ok = False
        items.append(entry)
    if not sizes_ok:
        for entry in items:
            entry.pop("input_size", None)
    return config.PredictorInput(
        history=items, input_size=(float(target_size) if sizes_ok else None))


def predict_m2(history, record, min_duration_ms=None):
    inp = build_m2_input(history, record, min_duration_ms=min_duration_ms)
    if inp is None:
        return None
    return _envelope(m2.HistoryBasedPredictor().predict(inp), "m2")


def predict(kind, history, record, min_duration_ms=None):
    if kind != "m2":
        raise ValueError("only m2 is supported")
    return predict_m2(history, record, min_duration_ms=min_duration_ms)
