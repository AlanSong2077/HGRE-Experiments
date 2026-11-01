"""Minimal data contract for the M2-only repeat experiment."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PredictorInput:
    command: str = ""
    command_class: str = "other"
    complexity: str = "n"
    input_size: float | None = None
    history: list = field(default_factory=list)


@dataclass
class RuntimePrediction:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    method: str = "prior"
    n_history: int = 0
    confidence: float = 0.0


EXP = {
    "min_duration_ms": 100.0,
    "history_window": 5,
    "p95_target": 0.95,
}
