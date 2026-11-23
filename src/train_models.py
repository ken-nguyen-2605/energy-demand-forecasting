"""Baseline, Ridge / Lasso and Random Forest models (Part C).

Protocol (identical for every model so results are comparable):
1. Hyper-parameters are selected by fitting on **train** and scoring MAE on **val**.
2. The selected configuration is refit on **train + val** and scored once on **test**.
3. All metrics come from ``evaluate.py``.

Outputs
-------
results/metrics.csv              all models x {val, test} x metrics
results/tuning_*.csv             validation curves of the hyper-parameter search
results/ablation_features.csv    feature-family ablation (Part B justification)
results/horizon_comparison.csv   1h-ahead vs 24h-ahead
results/feature_importance_rf.csv, results/lasso_coefficients.csv
results/predictions_test.parquet test-set predictions of every model (for Part E)
plots/*.png                      figures used in the report

Usage:  uv run python src/train_models.py [--horizon 24] [--fast]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluate import ResultsLog, error_by_group, evaluate, plot_forecast, plot_metric_bars, plot_residuals
from features import TARGET, TSO_FORECAST, FeatureConfig, feature_columns, make_datasets

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
PLOT_DIR = ROOT_DIR / "plots"
SEED = 0

CATEGORICAL = ["cal_hour", "cal_dayofweek", "cal_month", "cal_hour_x_offday"]


# ----------------------------------------------------------------------------
# Model factories
# ----------------------------------------------------------------------------
def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Hour x off-day interaction: the daily profile on weekends/holidays differs."""
    df = df.copy()
    df["cal_hour_x_offday"] = df["cal_hour"] * 10 + df["cal_is_offday"]
    return df


def make_linear(model_cls, alpha: float, numeric: list[str]) -> Pipeline:
    """Standardise numeric features, one-hot calendar categories, then Ridge/Lasso.

    The scaler is fit inside the pipeline -> only on the rows passed to ``fit``
    (train, or train+val for the final model), never on the evaluation split.
    """
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), CATEGORICAL),
        ]
    )
    kwargs = {"alpha": alpha}
    if model_cls is Lasso:
        kwargs.update(max_iter=20_000, tol=1e-3, selection="cyclic")
    return Pipeline([("pre", pre), ("model", model_cls(**kwargs))])


def make_rf(n_estimators: int = 300, **params) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=SEED, **params)


# ----------------------------------------------------------------------------
# Experiment helpers
# ----------------------------------------------------------------------------
def timed_fit(model, X, y):
    t0 = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t0


def tune_linear(model_cls, alphas, train, val, cols, numeric) -> tuple[float, pd.DataFrame]:
    rows = []
    for a in alphas:
        m, _ = timed_fit(make_linear(model_cls, a, numeric), add_interactions(train[cols]), train[TARGET])
        pred = m.predict(add_interactions(val[cols]))
        rows.append({"alpha": a, **evaluate(val[TARGET], pred)})
        print(f"  {model_cls.__name__:5s} alpha={a:<10.4g} val MAE={rows[-1]['MAE']:.1f}")
    curve = pd.DataFrame(rows)
    return float(curve.loc[curve["MAE"].idxmin(), "alpha"]), curve


def tune_rf(grid: dict, train, val, cols, n_estimators: int) -> tuple[dict, pd.DataFrame]:
    rows = []
    keys = list(grid)
    for values in itertools.product(*grid.values()):
        params = dict(zip(keys, values))
        m, secs = timed_fit(make_rf(n_estimators, **params), train[cols], train[TARGET])
        rows.append({**params, **evaluate(val[TARGET], m.predict(val[cols])), "fit_s": secs})
        print(f"  RF {params} val MAE={rows[-1]['MAE']:.1f} ({secs:.1f}s)")
    curve = pd.DataFrame(rows)
    best = curve.loc[curve["MAE"].idxmin(), keys].to_dict()
    best = {k: (None if pd.isna(v) else (int(v) if isinstance(v, float) and float(v).is_integer() and k != "max_features" else v))
            for k, v in best.items()}
    return best, curve


# ----------------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------------
def run_main(horizon: int, fast: bool) -> dict:
    cfg = FeatureConfig(horizon=horizon)
    data = make_datasets(cfg)
    train, val, test = data["train"], data["val"], data["test"]
    trval = pd.concat([train, val], ignore_index=True)
    cols = feature_columns(train)
    numeric = [c for c in cols if c not in CATEGORICAL]
    print(f"Features ({len(cols)}): train {train.shape}, val {val.shape}, test {test.shape}")

    log = ResultsLog()
    preds_test: dict[str, np.ndarray] = {}
    summary: dict = {"horizon": horizon, "n_features": len(cols), "features": cols,
                     "n_train": len(train), "n_val": len(val), "n_test": len(test)}

    # ---- 1. Baselines (no fitting) --------------------------------------------
    baselines = {
        "Naive (t-24h)": "lag_24",
        "Seasonal naive (t-168h)": "lag_168",
        "Mean of last 4 weeks": "lag_same_hour_4w_mean",
    }
    for name, col in baselines.items():
        log.add(name, "val", val[TARGET], val[col], family="baseline")
        log.add(name, "test", test[TARGET], test[col], family="baseline")
        preds_test[name] = test[col].to_numpy()
    log.add("TSO forecast (reference)", "val", val[TARGET], val[TSO_FORECAST], family="reference")
    log.add("TSO forecast (reference)", "test", test[TARGET], test[TSO_FORECAST], family="reference")
    preds_test["TSO forecast (reference)"] = test[TSO_FORECAST].to_numpy()

    # ---- 2. Ridge / Lasso ------------------------------------------------------
    alphas_ridge = np.logspace(-2, 4, 13)
    alphas_lasso = np.logspace(-1, 2.5, 8) if not fast else [1.0, 10.0]
    print("Tuning Ridge ...")
    best_ridge, curve_ridge = tune_linear(Ridge, alphas_ridge, train, val, cols, numeric)
    print("Tuning Lasso ...")
    best_lasso, curve_lasso = tune_linear(Lasso, alphas_lasso, train, val, cols, numeric)
    curve_ridge.to_csv(RESULTS_DIR / "tuning_ridge.csv", index=False)
    curve_lasso.to_csv(RESULTS_DIR / "tuning_lasso.csv", index=False)

    lasso_final = None
    for name, cls, alpha in [("Ridge", Ridge, best_ridge), ("Lasso", Lasso, best_lasso)]:
        m_val, _ = timed_fit(make_linear(cls, alpha, numeric), add_interactions(train[cols]), train[TARGET])
        log.add(name, "val", val[TARGET], m_val.predict(add_interactions(val[cols])), family="linear", alpha=alpha)
        m, secs = timed_fit(make_linear(cls, alpha, numeric), add_interactions(trval[cols]), trval[TARGET])
        p = m.predict(add_interactions(test[cols]))
        log.add(name, "test", test[TARGET], p, family="linear", alpha=alpha, fit_s=secs)
        preds_test[name] = p
        summary[f"{name.lower()}_alpha"] = alpha
        if cls is Lasso:
            lasso_final = m

    # Lasso coefficients (feature selection view)
    names = lasso_final.named_steps["pre"].get_feature_names_out()
    coefs = pd.DataFrame({"feature": names, "coef": lasso_final.named_steps["model"].coef_})
    coefs["abs"] = coefs["coef"].abs()
    coefs.sort_values("abs", ascending=False).to_csv(RESULTS_DIR / "lasso_coefficients.csv", index=False)
    summary["lasso_nonzero"] = int((coefs["coef"] != 0).sum())
    summary["lasso_total_coefs"] = int(len(coefs))

    # ---- 3. Random Forest -----------------------------------------------------
    grid = (
        {"max_depth": [None], "min_samples_leaf": [1], "max_features": [0.5]}
        if fast
        else {"max_depth": [12, 20, None], "min_samples_leaf": [1, 5], "max_features": [0.33, 0.5, 1.0]}
    )
    n_trees = 100 if fast else 200
    print("Tuning Random Forest ...")
    best_rf, curve_rf = tune_rf(grid, train, val, cols, n_trees)
    curve_rf.to_csv(RESULTS_DIR / "tuning_rf.csv", index=False)
    summary["rf_best_params"] = best_rf

    n_final = 100 if fast else 500
    rf_val, _ = timed_fit(make_rf(n_final, **best_rf), train[cols], train[TARGET])
    log.add("Random Forest", "val", val[TARGET], rf_val.predict(val[cols]), family="tree")
    rf, secs = timed_fit(make_rf(n_final, **best_rf), trval[cols], trval[TARGET])
    p_rf = rf.predict(test[cols])
    log.add("Random Forest", "test", test[TARGET], p_rf, family="tree", fit_s=secs)
    preds_test["Random Forest"] = p_rf

    # Feature importance: impurity-based + permutation (on val, model fit on train)
    print("Permutation importance ...")
    perm = permutation_importance(rf_val, val[cols], val[TARGET], scoring="neg_mean_absolute_error",
                                  n_repeats=3 if fast else 5, random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({
        "feature": cols,
        "impurity": rf_val.feature_importances_,
        "perm_mae_increase": perm.importances_mean,
        "perm_std": perm.importances_std,
    }).sort_values("perm_mae_increase", ascending=False)
    imp.to_csv(RESULTS_DIR / "feature_importance_rf.csv", index=False)

    # ---- 4. Save tables, predictions, plots ----------------------------------
    table = log.to_frame()
    table.to_csv(RESULTS_DIR / "metrics.csv", index=False, float_format="%.4f")
    pred_df = pd.DataFrame({"time": test["time"], "actual": test[TARGET], **preds_test})
    pred_df.to_parquet(RESULTS_DIR / "predictions_test.parquet", index=False)

    err_hour = pd.DataFrame({k: error_by_group(test["time"], test[TARGET], v, "hour") for k, v in preds_test.items()})
    err_hour.to_csv(RESULTS_DIR / "mae_by_hour_test.csv")
    err_dow = pd.DataFrame({k: error_by_group(test["time"], test[TARGET], v, "dayofweek") for k, v in preds_test.items()})
    err_dow.to_csv(RESULTS_DIR / "mae_by_dow_test.csv")

    make_plots(table, test, preds_test, curve_ridge, curve_lasso, curve_rf, imp, err_hour)
    return summary


def make_plots(table, test, preds, curve_ridge, curve_lasso, curve_rf, imp, err_hour) -> None:
    import matplotlib.pyplot as plt

    plot_metric_bars(table, "MAE", PLOT_DIR / "c_mae_test.png")
    plot_metric_bars(table, "MAPE", PLOT_DIR / "c_mape_test.png")
    plot_forecast(test["time"], test[TARGET],
                  {k: preds[k] for k in ["Naive (t-24h)", "Ridge", "Random Forest"]},
                  PLOT_DIR / "c_forecast_2weeks.png", title="Test set: two weeks of day-ahead forecasts",
                  start="2018-11-05")
    plot_forecast(test["time"], test[TARGET],
                  {k: preds[k] for k in ["Seasonal naive (t-168h)", "Ridge", "Random Forest"]},
                  PLOT_DIR / "c_forecast_holiday.png", title="Test set: week containing 12 Oct (national holiday)",
                  start="2018-10-08", days=7)
    plot_residuals(test[TARGET], preds["Random Forest"], PLOT_DIR / "c_residuals_rf.png", "Random Forest - test set")
    plot_residuals(test[TARGET], preds["Ridge"], PLOT_DIR / "c_residuals_ridge.png", "Ridge - test set")

    # Validation curves of linear models
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(curve_ridge["alpha"], curve_ridge["MAE"], "o-", label="Ridge")
    ax.semilogx(curve_lasso["alpha"], curve_lasso["MAE"], "s-", label="Lasso")
    ax.set_xlabel(r"Regularisation strength $\alpha$")
    ax.set_ylabel("Validation MAE (MW)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "c_tuning_linear.png", dpi=150)
    plt.close(fig)

    # RF tuning heatmap-like plot
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (depth, leaf), g in curve_rf.groupby(["max_depth", "min_samples_leaf"], dropna=False):
        ax.plot(g["max_features"].astype(str), g["MAE"], "o-", label=f"depth={depth}, leaf={leaf}")
    ax.set_xlabel("max_features")
    ax.set_ylabel("Validation MAE (MW)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "c_tuning_rf.png", dpi=150)
    plt.close(fig)

    # Permutation importance top-15
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(top["feature"], top["perm_mae_increase"], xerr=top["perm_std"], color="seagreen")
    ax.set_xlabel("Increase of validation MAE when permuted (MW)")
    ax.set_title("Random Forest - permutation importance (top 15)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "b_feature_importance_rf.png", dpi=150)
    plt.close(fig)

    # MAE by local hour
    fig, ax = plt.subplots(figsize=(9, 4))
    for k in ["Naive (t-24h)", "Ridge", "Random Forest", "TSO forecast (reference)"]:
        ax.plot(err_hour.index, err_hour[k], "o-", ms=3, label=k)
    ax.set_xlabel("Local hour (Europe/Madrid)")
    ax.set_ylabel("MAE (MW)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "c_mae_by_hour.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Part-B experiments: feature-family ablation and forecast horizon
# ----------------------------------------------------------------------------
def run_ablation(horizon: int, rf_params: dict, ridge_alpha: float, fast: bool) -> pd.DataFrame:
    """Fit Ridge and RF on train with different feature families, score on val."""
    setups = {
        "Calendar only": dict(use_calendar=True, use_lags=False, use_weather=False),
        "Weather only": dict(use_calendar=False, use_lags=False, use_weather=True),
        "Lags only": dict(use_calendar=False, use_lags=True, use_weather=False),
        "Calendar + Weather": dict(use_calendar=True, use_lags=False, use_weather=True),
        "Calendar + Lags": dict(use_calendar=True, use_lags=True, use_weather=False),
        "Lags + Weather": dict(use_calendar=False, use_lags=True, use_weather=True),
        "All (Calendar + Lags + Weather)": dict(use_calendar=True, use_lags=True, use_weather=True),
    }
    rows = []
    for name, flags in setups.items():
        data = make_datasets(FeatureConfig(horizon=horizon, **flags))
        tr, va = data["train"], data["val"]
        cols = feature_columns(tr)
        rf, _ = timed_fit(make_rf(100, **rf_params), tr[cols], tr[TARGET])
        s_rf = evaluate(va[TARGET], rf.predict(va[cols]))
        row = {"features": name, "n_features": len(cols), "RF_MAE": s_rf["MAE"], "RF_MAPE": s_rf["MAPE"]}
        if flags["use_calendar"]:  # linear pipeline needs calendar categories
            numeric = [c for c in cols if c not in CATEGORICAL]
            lin, _ = timed_fit(make_linear(Ridge, ridge_alpha, numeric), add_interactions(tr[cols]), tr[TARGET])
            s_lin = evaluate(va[TARGET], lin.predict(add_interactions(va[cols])))
        else:
            from sklearn.preprocessing import StandardScaler as _S
            lin = Pipeline([("sc", _S()), ("m", Ridge(alpha=ridge_alpha))]).fit(tr[cols], tr[TARGET])
            s_lin = evaluate(va[TARGET], lin.predict(va[cols]))
        row.update({"Ridge_MAE": s_lin["MAE"], "Ridge_MAPE": s_lin["MAPE"]})
        rows.append(row)
        print(f"  {name:35s} n={len(cols):2d}  RF MAE={s_rf['MAE']:.0f}  Ridge MAE={s_lin['MAE']:.0f}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "ablation_features.csv", index=False, float_format="%.4f")
    return out


def run_horizon_comparison(rf_params: dict, ridge_alpha: float) -> pd.DataFrame:
    """Same models at 1-hour-ahead vs 24-hour-ahead (scored on val)."""
    rows = []
    for h in (1, 24):
        data = make_datasets(FeatureConfig(horizon=h))
        tr, va = data["train"], data["val"]
        cols = feature_columns(tr)
        numeric = [c for c in cols if c not in CATEGORICAL]
        naive = evaluate(va[TARGET], va[f"lag_{h}"])
        lin, _ = timed_fit(make_linear(Ridge, ridge_alpha, numeric), add_interactions(tr[cols]), tr[TARGET])
        s_lin = evaluate(va[TARGET], lin.predict(add_interactions(va[cols])))
        rf, _ = timed_fit(make_rf(100, **rf_params), tr[cols], tr[TARGET])
        s_rf = evaluate(va[TARGET], rf.predict(va[cols]))
        for model, s in [(f"Naive (t-{h}h)", naive), ("Ridge", s_lin), ("Random Forest", s_rf)]:
            rows.append({"horizon_h": h, "model": model, "MAE": s["MAE"], "RMSE": s["RMSE"], "MAPE": s["MAPE"]})
        print(f"  h={h:2d}: naive {naive['MAE']:.0f}, Ridge {s_lin['MAE']:.0f}, RF {s_rf['MAE']:.0f}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "horizon_comparison.csv", index=False, float_format="%.4f")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--fast", action="store_true", help="tiny grids, for smoke tests")
    args = ap.parse_args()
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    summary = run_main(args.horizon, args.fast)
    print("\nFeature-family ablation (val MAE) ...")
    run_ablation(args.horizon, summary["rf_best_params"], summary["ridge_alpha"], args.fast)
    print("\nHorizon comparison ...")
    run_horizon_comparison(summary["rf_best_params"], summary["ridge_alpha"])
    summary["runtime_s"] = round(time.perf_counter() - t0, 1)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    table = pd.read_csv(RESULTS_DIR / "metrics.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n", table[["model", "split", "MAE", "RMSE", "MAPE", "R2"]].round(3).to_string(index=False))
    print(f"\nDone in {summary['runtime_s']}s")


if __name__ == "__main__":
    main()
