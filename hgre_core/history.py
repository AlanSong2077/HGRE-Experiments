"""History-based runtime predictor (shared HGRE core).

Input: PredictorInput(history=..., input_size=...).
Output: RuntimePrediction(mean_ms, p50_ms, p95_ms, method, ...).
"""
from __future__ import annotations

import math

import numpy as np

from hgre_core import config

__all__ = ["HistoryBasedPredictor", "METHODS"]

# 复杂度族形状（n 为缩放变量 x），作为历史时长的复杂度族。
_FAMILY_FUNCS = {
    "n": lambda v: v,
    "nlogn": lambda v: v * np.log2(v + 1.0),
    "n2": lambda v: v * v,
    "n3": lambda v: v * v * v,
}
_FAMILIES = tuple(_FAMILY_FUNCS)
_METHOD_ORDER = ("mean", "last", "ema", "linear", "poly2", "family")
METHODS = _METHOD_ORDER + ("family_ic", "auto")

_DEFAULT_PRIOR = (1000.0, 800.0, 1600.0)   # (mean, p50, p95) 空历史先验
_Z_95 = 1.6448536269514722
_MAX_FOLDS = 10                            # LOO 最多折数（控延迟）
_CLIP_LOW, _CLIP_HIGH = 0.02, 50.0         # 点预测裁剪系数（相对 max(hist)）


def _field(obj, name, default=None):
    """同时支持 dataclass / dict 的字段读取。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _parse_history(history):
    """-> (y, x, has_sizes)；y/x 为 np.ndarray，空历史返回 (empty, None, False)。"""
    ys, xs = [], []
    all_sizes = True
    for item in list(history or []):
        runtime = size = None
        try:
            if isinstance(item, dict):
                for key in ("runtime_ms", "duration_ms", "ms", "y", "value"):
                    if item.get(key) is not None:
                        runtime = float(item[key])
                        break
                if item.get("input_size") is not None:
                    size = float(item["input_size"])
            elif isinstance(item, (list, tuple)):
                vals = [float(v) for v in item if v is not None]
                if len(vals) == 1:
                    runtime = vals[0]
                elif len(vals) >= 2:
                    size, runtime = vals[0], vals[1]
            else:
                runtime = float(item)
        except (TypeError, ValueError):
            continue
        if runtime is None or not math.isfinite(runtime) or runtime <= 0:
            continue
        if size is None or not math.isfinite(size) or size <= 0:
            size = None
        ys.append(runtime)
        xs.append(size)
        if size is None:
            all_sizes = False
    if not ys:
        return np.empty(0), None, False
    y = np.asarray(ys, dtype=float)
    if all_sizes:
        x = np.asarray(xs, dtype=float)
    else:
        x = np.arange(1.0, len(ys) + 1.0)   # 退化为步序号
    return y, x, all_sizes


def _target_x(x, has_sizes, input_size):
    """预测点的缩放变量取值。"""
    if has_sizes:
        try:
            target = float(input_size)
        except (TypeError, ValueError):
            target = float("nan")
        if math.isfinite(target) and target > 0:
            return target
        return float(x[-1])       # 目标规模未知：停留在最后一个历史规模
    return float(x[-1] + 1.0)     # 无规模变量：预测下一步序号


def _fit_linear(y, x):
    """OLS y = a + b*x -> (a, b)；单点/零方差时 b=0。"""
    if len(y) < 2:
        return float(y[0]), 0.0
    xm, ym = float(np.mean(x)), float(np.mean(y))
    var = float(np.sum((x - xm) ** 2))
    if var <= 1e-12:
        return ym, 0.0
    b = float(np.sum((x - xm) * (y - ym)) / var)
    return ym - b * xm, b


def _fit_poly2(y, x):
    """OLS y = a + b*x + c*x^2 -> (a, b, c)；n<4 时退化为线性。"""
    if len(y) < 4:
        a, b = _fit_linear(y, x)
        return a, b, 0.0
    design = np.column_stack([np.ones_like(x), x, x * x])
    try:
        coef = np.linalg.solve(design.T @ design, design.T @ y)
    except (np.linalg.LinAlgError, ValueError):
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
    a, b, c = (float(v) for v in coef)
    return a, b, c


def _fit_family(y, f):
    """OLS y = a + c*f（闭合形式）-> (a, c)。"""
    fm, ym = float(np.mean(f)), float(np.mean(y))
    var = float(np.sum((f - fm) ** 2))
    if var <= 1e-12:
        return ym, 0.0
    c = float(np.sum((f - fm) * (y - ym)) / var)
    return ym - c * fm, c


def _family_infit(y, x, family):
    """单族样本内拟合 -> (family, a, c, median_APE)。"""
    f = _FAMILY_FUNCS[family](x)
    a, c = _fit_family(y, f)
    fitted = a + c * f
    if not np.all(np.isfinite(fitted)) or np.any(fitted <= 0):
        return family, a, c, float("inf")
    return family, a, c, float(np.median(np.abs(fitted - y) / y))


def _fit_best_family(y, x):
    """在四个族里选样本内 median APE 最小者 -> (family, a, c)。"""
    best = None
    for family in _FAMILIES:
        family, a, c, score = _family_infit(y, x, family)
        if not math.isfinite(score):
            continue
        if best is None or score < best[0] * (1.0 - 1e-9) - 1e-15:
            best = (score, family, a, c)
    if best is None:                       # 全部退化：常数模型
        return "n", float(np.mean(y)), 0.0
    return best[1], best[2], best[3]


def _select_family_loo(y, x):
    """用 LOO median APE 选族（抗过拟合）；样本不足时退化为样本内选族。"""
    n = len(y)
    if n < 4:
        return _fit_best_family(y, x)[0]
    folds = range(n) if n <= _MAX_FOLDS else range(n - _MAX_FOLDS, n)
    best_family, best_score = _FAMILIES[0], float("inf")
    for family in _FAMILIES:
        errors = []
        for i in folds:
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            a, c = _fit_family(y[mask], _FAMILY_FUNCS[family](x[mask]))
            pred = a + c * float(_FAMILY_FUNCS[family](float(x[i])))
            if math.isfinite(pred) and pred > 0:
                errors.append(abs(pred - float(y[i])) / float(y[i]))
        if errors and float(np.median(errors)) < best_score - 1e-12:
            best_family, best_score = family, float(np.median(errors))
    return best_family


def _available(method, n):
    """该方法在 n 个样本上是否可用。"""
    if n < 1:
        return False
    if method in ("mean", "last", "ema"):
        return True
    if method == "poly2":
        return n >= 4
    if method in ("family", "family_ic"):
        return n >= 3
    return n >= 2                  # linear


def _effective_method(method, n):
    """样本不足时的降级（保证不抛异常）。"""
    if method in ("family", "family_ic") and n < 3:
        method = "linear" if n >= 2 else "mean"
    if method == "poly2" and n < 4:
        method = "linear" if n >= 2 else "mean"
    if method == "linear" and n < 2:
        method = "mean"
    return method


class HistoryBasedPredictor:
    """历史序列 -> config.RuntimePrediction（毫秒）。

    用法::

        p = HistoryBasedPredictor()
        pred = p.predict(config.PredictorInput(history=[120.0, 90.0, 110.0]))
        pred = p.predict(inp, method="family")     # 消融实验用

    无状态、确定性；重复调用结果完全一致；不修改输入对象。
    """

    name = "M2"

    def __init__(self, ema_alpha: float = 0.5, p95_target: float = 0.95,
                 sigma_fallback: float = 0.25):
        self.ema_alpha = float(ema_alpha)
        self.p95_target = float(p95_target)
        self.sigma_fallback = float(sigma_fallback)

    # ------------------------------------------------------------ 方法实现
    def _forecast(self, method, y, x, target_x):
        """单个方法外推 -> (point_ms, fitted_values, detail)。"""
        if method == "mean":
            mu = float(np.mean(y))
            return mu, np.full(y.size, mu), ""
        if method == "last":
            fitted = np.concatenate(([float(y[0])], y[:-1]))
            return float(y[-1]), fitted, ""
        if method == "ema":
            alpha = self.ema_alpha
            e = np.empty_like(y)
            e[0] = y[0]
            for i in range(1, y.size):
                e[i] = alpha * y[i] + (1.0 - alpha) * e[i - 1]
            fitted = np.concatenate(([float(e[0])], e[:-1]))
            return float(e[-1]), fitted, ""
        if method == "linear":
            a, b = _fit_linear(y, x)
            return float(a + b * target_x), a + b * x, ""
        if method == "poly2":
            a, b, c = _fit_poly2(y, x)
            fitted = a + b * x + c * x * x
            return float(a + b * target_x + c * target_x * target_x), fitted, ""
        if method == "family":
            family, a, c = _fit_best_family(y, x)
            fitted = a + c * _FAMILY_FUNCS[family](x)
            point = a + c * float(_FAMILY_FUNCS[family](target_x))
            return float(point), fitted, family
        if method == "family_ic":
            family = _select_family_loo(y, x)
            f = _FAMILY_FUNCS[family](x)
            a, c = _fit_family(y, f)
            point = a + c * float(_FAMILY_FUNCS[family](target_x))
            return float(point), a + c * f, family
        raise ValueError(f"unknown method: {method}")

    def _loo_choice(self, y, x):
        """对方法族逐点留一，按 median APE 选方法（确定性，成本 << 1ms）。"""
        n = len(y)
        folds = range(n) if n <= _MAX_FOLDS else range(n - _MAX_FOLDS, n)
        best_method, best_score = "mean", float("inf")
        for method in _METHOD_ORDER:
            errors = []
            for i in folds:
                if not _available(method, n - 1):
                    continue
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                try:
                    point, _fitted, _detail = self._forecast(
                        method, y[mask], x[mask], float(x[i]))
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    continue
                if math.isfinite(point) and point > 0:
                    errors.append(abs(point - float(y[i])) / float(y[i]))
            if not errors:
                continue
            score = float(np.median(errors))
            if score < best_score - 1e-12:      # 平手时偏好更简约（顺序靠前）
                best_method, best_score = method, score
        return best_method

    def _auto_choice(self, y, x, has_sizes):
        """auto 规则（见模块 docstring）。"""
        n = int(y.size)
        if n <= 1:
            return "mean"
        if has_sizes and bool(np.allclose(x, x[0])):
            return "last"                       # 规模没变：最近一次最可靠
        if n in (2, 3):
            return "linear"
        if n >= 5:
            return self._loo_choice(y, x)
        return "family"

    # ------------------------------------------------------------ P95 / 置信度
    def _p95(self, y, fitted, p50):
        """-> (p95, used_conformal, m_residuals)。"""
        mask = np.isfinite(fitted) & (fitted > 0) & np.isfinite(y) & (y > 0)
        residual = np.log(y[mask]) - np.log(fitted[mask])
        m = int(residual.size)
        k = int(math.ceil((m + 1) * self.p95_target))
        if m > 0 and k <= m:                    # conformal 序统计量
            q, used = float(np.sort(residual)[k - 1]), True
        else:                                   # 样本不足：lognormal sigma 兜底
            q, used = _Z_95 * self.sigma_fallback, False
        return float(p50 * math.exp(q)), used, m

    def _confidence(self, n, used_conformal, chosen):
        """覆盖度启发式（非校准概率）：样本越多、有 conformal 支持越自信。"""
        conf = 0.25 + 0.11 * min(int(n), 5)
        if used_conformal:
            conf += 0.05
        else:
            conf *= 0.92
        if chosen in ("mean", "last", "ema"):
            conf *= 0.97
        return float(min(0.95, max(0.05, conf)))

    # ------------------------------------------------------------ 预测
    def predict(self, inp, method: str = "auto"):
        """-> config.RuntimePrediction。非法输入不抛异常，走先验/降级。"""
        y, x, has_sizes = _parse_history(_field(inp, "history", None))
        n = int(y.size)
        if n == 0:                              # 空历史：冷启动先验
            mean_ms, p50_ms, p95_ms = _DEFAULT_PRIOR
            return config.RuntimePrediction(
                mean_ms=mean_ms, p50_ms=p50_ms, p95_ms=p95_ms,
                method="prior", n_history=0, confidence=0.10)

        target_x = _target_x(x, has_sizes, _field(inp, "input_size", None))
        requested = str(method or "auto").strip().lower()
        if requested == "auto":
            chosen = _effective_method(self._auto_choice(y, x, has_sizes), n)
            prefix = "auto:"
        elif requested in METHODS:
            chosen = _effective_method(requested, n)
            prefix = ""
        else:
            chosen, prefix = "mean", "fallback:"

        point, fitted, detail = self._forecast(chosen, y, x, target_x)
        if not math.isfinite(point) or point <= 0:   # 外推失败：退回均值
            chosen, prefix, detail = "mean", "fallback:", ""
            point, fitted, detail = self._forecast(chosen, y, x, target_x)

        # 防爆裁剪：相对历史量级
        max_hist = float(np.max(y))
        low, high = _CLIP_LOW * max_hist, _CLIP_HIGH * max_hist
        point = float(min(high, max(low, point)))
        p50 = max(1.0, point)
        p95_raw, used_conformal, _m = self._p95(y, fitted, p50)
        p95 = float(min(high, max(p50, p95_raw)))

        label = prefix + chosen + ((":" + detail) if detail else "")
        return config.RuntimePrediction(
            mean_ms=p50, p50_ms=p50, p95_ms=p95, method=label,
            n_history=n,
            confidence=self._confidence(n, used_conformal, chosen))
