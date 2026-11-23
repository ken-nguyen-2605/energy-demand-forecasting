"""Shared evaluation utilities for every forecasting model (Part C).

All team members should evaluate their models through this module so that the
numbers in the report are computed identically:

    from evaluate import evaluate, ResultsLog
    log = ResultsLog()
    log.add("Ridge", "test", y_true, y_pred)
    log.to_frame()

Metrics
-------
MAE    mean absolute error (MW)
RMSE   root mean squared error (MW)          - penalises large errors
MAPE   mean absolute percentage error (%)    - scale free, load is never ~0
sMAPE  symmetric MAPE (%)
R2     coefficient of determination
MaxAE  maximum absolute error (MW)           - worst hour
ME     mean error, forecast - actual (MW)    - bias
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ("MAE", "RMSE", "MAPE", "sMAPE", "R2", "MaxAE", "ME")


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if not (np.isfinite(yt).all() and np.isfinite(yp).all()):
        raise ValueError("y_true / y_pred contain NaN or inf")
    return yt, yp


def mae(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    if np.any(yt == 0):
        raise ValueError("MAPE undefined when y_true contains zeros")
    return float(np.mean(np.abs((yt - yp) / yt)) * 100)


def smape(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    denom = (np.abs(yt) + np.abs(yp)) / 2
    return float(np.mean(np.abs(yt - yp) / denom) * 100)


def r2(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def max_ae(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.max(np.abs(yt - yp)))


def me(y_true, y_pred) -> float:
    """Bias (forecast - actual); positive means over-forecasting."""
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(yp - yt))


def evaluate(y_true, y_pred) -> dict[str, float]:
    """Compute every metric in ``METRICS`` and return them as a dict."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "MaxAE": max_ae(y_true, y_pred),
        "ME": me(y_true, y_pred),
    }


@dataclass
class ResultsLog:
    """Accumulates (model, split, metrics) rows and exports a tidy table."""

    rows: list[dict] = field(default_factory=list)

    def add(self, model: str, split: str, y_true, y_pred, **extra) -> dict[str, float]:
        scores = evaluate(y_true, y_pred)
        self.rows.append({"model": model, "split": split, **scores, **extra})
        return scores

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def pivot(self, metric: str = "MAE") -> pd.DataFrame:
        return self.to_frame().pivot(index="model", columns="split", values=metric)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False, float_format="%.4f")


# ----------------------------------------------------------------------------
# Diagnostics / plots
# ----------------------------------------------------------------------------
def error_by_group(time: pd.Series, y_true, y_pred, by: str = "hour", tz: str = "Europe/Madrid") -> pd.Series:
    """MAE grouped by local ``hour``, ``dayofweek`` or ``month``."""
    yt, yp = _as_arrays(y_true, y_pred)
    local = pd.to_datetime(time).dt.tz_convert(tz)
    key = {"hour": local.dt.hour, "dayofweek": local.dt.day_of_week, "month": local.dt.month}[by]
    return pd.Series(np.abs(yt - yp), index=key.to_numpy()).groupby(level=0).mean()


def plot_forecast(time, y_true, preds: dict[str, np.ndarray], out_path: Path, title: str = "",
                  start: str | None = None, days: int = 14) -> None:
    """Overlay actual load and model predictions on a ``days``-long window."""
    t = pd.to_datetime(pd.Series(time)).reset_index(drop=True)
    start_ts = pd.Timestamp(start, tz="UTC") if start else t.iloc[0]
    mask = (t >= start_ts) & (t < start_ts + pd.Timedelta(days=days))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(t[mask], np.asarray(y_true)[mask], color="black", lw=1.8, label="Actual")
    for name, yp in preds.items():
        ax.plot(t[mask], np.asarray(yp)[mask], lw=1.1, alpha=0.9, label=name)
    ax.set_ylabel("Total load (MW)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(ncol=min(len(preds) + 1, 5), fontsize=9, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, out_path: Path, title: str = "") -> None:
    """Residual histogram and predicted-vs-actual scatter."""
    yt, yp = _as_arrays(y_true, y_pred)
    res = yt - yp
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(res, bins=60, color="steelblue", alpha=0.85)
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set_xlabel("Residual y - ŷ (MW)")
    axes[0].set_ylabel("Hours")
    axes[0].set_title(f"Residuals (mean = {res.mean():.0f}, sd = {res.std():.0f})")
    axes[1].scatter(yt, yp, s=3, alpha=0.3)
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    axes[1].plot([lo, hi], [lo, hi], color="red", lw=1)
    axes[1].set_xlabel("Actual (MW)")
    axes[1].set_ylabel("Predicted (MW)")
    axes[1].set_title("Predicted vs actual")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_bars(table: pd.DataFrame, metric: str, out_path: Path, split: str = "test") -> None:
    """Horizontal bar chart of one metric across models for a given split."""
    sub = table[table["split"] == split].sort_values(metric, ascending=metric != "R2")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(sub) + 1.2))
    ax.barh(sub["model"], sub[metric], color="steelblue")
    for i, v in enumerate(sub[metric]):
        ax.text(v, i, f" {v:,.2f}" if metric in ("MAPE", "sMAPE", "R2") else f" {v:,.0f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(f"{metric} on {split} set")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
