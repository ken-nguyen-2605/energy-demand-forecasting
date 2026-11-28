"""Part D — Gradient Boosting (LightGBM & XGBoost) Pipeline.

Modular step-by-step execution strictly following the shared team protocol
(src/features.py, src/evaluate.py, src/train_models.py):
- Step 1 (`--step 1`): Off-the-shelf default LightGBM & XGBoost + learning curves (train -> val).
- Step 2 (`--step 2`): Controlled Loss Comparison (2A) + Optuna TPE Tuning (2B) (train -> val).
  Test set remains strictly LOCKED during Step 1 and Step 2.
- Step 3 (`--step 3`): Final evaluation on test (refit frozen Step 2 configs on train+val -> test ONCE),
  5-seed stability check, horizon comparison (h=1 vs h=24), and diagnostic plots.
- Step 4 (`--step 4`): Exact TreeSHAP explainability on test set (global, grouped by 3 families,
  RF permutation comparison, dependence plots, and 3 local waterfall case studies).
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr
import shap
import xgboost as xgb

try:
    from src.evaluate import (
        ResultsLog,
        error_by_group,
        evaluate,
        plot_forecast,
        plot_metric_bars,
        plot_residuals,
    )
    from src.features import TARGET, FeatureConfig, feature_columns, make_datasets
except ImportError:
    from evaluate import (
        ResultsLog,
        error_by_group,
        evaluate,
        plot_forecast,
        plot_metric_bars,
        plot_residuals,
    )
    from features import TARGET, FeatureConfig, feature_columns, make_datasets

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
PLOT_DIR = ROOT_DIR / "plots"
SEED = 0
LGBM_CAT_COLS = ["cal_hour", "cal_dayofweek", "cal_month"]
KNOWN_TEST_SENSOR_GLITCH_UTC = "2018-06-05 10:00"


def _objective_short_name(obj_name: str) -> str:
    return {
        "huber": "Huber",
        "reg:pseudohubererror": "Pseudo-Huber",
        "regression_l1": "L1",
        "reg:absoluteerror": "L1",
        "regression": "L2",
        "reg:squarederror": "L2",
    }.get(obj_name, obj_name)


def _fit_lgbm(params: dict, X: pd.DataFrame, y: pd.Series, cat_cols: list[str] | None = None, **fit_kwargs):
    """Instantiate and fit LGBMRegressor with optional categorical_feature support."""
    model = lgb.LGBMRegressor(**params)
    kwargs = dict(fit_kwargs)
    if cat_cols:
        present = [c for c in cat_cols if c in X.columns]
        if present:
            kwargs["categorical_feature"] = present
    return timed_fit(model, X, y, **kwargs)


def _fit_xgb(params: dict, X: pd.DataFrame, y: pd.Series, **fit_kwargs):
    """Instantiate and fit XGBRegressor, computing median base_score dynamically for Pseudo-Huber."""
    p = dict(params)
    if p.get("objective") == "reg:pseudohubererror":
        p["base_score"] = float(np.median(y))
    model = xgb.XGBRegressor(**p)
    return timed_fit(model, X, y, **fit_kwargs)


def timed_fit(model, X, y, **fit_kwargs):
    """Fit an estimator and return (fitted_model, elapsed_seconds)."""
    t0 = time.perf_counter()
    model.fit(X, y, **fit_kwargs)
    return model, time.perf_counter() - t0


def _read_part_c_rf_val_mae() -> float | None:
    c_metrics = RESULTS_DIR / "metrics.csv"
    if c_metrics.exists():
        df_c = pd.read_csv(c_metrics)
        rf_row = df_c.query("model == 'Random Forest' and split == 'val'")
        if not rf_row.empty:
            return float(rf_row["MAE"].iloc[0])
    return None


# ============================================================================
# STEP 1: Default (Untuned) Boosting & Learning Curves (train -> val)
# ============================================================================
def step1_untuned_and_learning_curve(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cols: list[str],
    lc_trees: int = 600,
    lc_lr: float = 0.1,
) -> tuple[pd.DataFrame, dict]:
    """Run default LightGBM & XGBoost on train -> val and plot MAE learning curves."""
    X_tr, y_tr = train[cols], train[TARGET]
    X_va, y_va = val[cols], val[TARGET]

    log = ResultsLog()

    lgb_def = lgb.LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1)
    lgb_def, lgb_s = timed_fit(lgb_def, X_tr, y_tr)
    log.add("LightGBM (default)", "train", y_tr, lgb_def.predict(X_tr), family="boosting_step1", fit_s=lgb_s)
    log.add("LightGBM (default)", "val", y_va, lgb_def.predict(X_va), family="boosting_step1", fit_s=lgb_s)

    xgb_def = xgb.XGBRegressor(tree_method="hist", random_state=SEED, n_jobs=-1)
    xgb_def, xgb_s = timed_fit(xgb_def, X_tr, y_tr)
    log.add("XGBoost (default)", "train", y_tr, xgb_def.predict(X_tr), family="boosting_step1", fit_s=xgb_s)
    log.add("XGBoost (default)", "val", y_va, xgb_def.predict(X_va), family="boosting_step1", fit_s=xgb_s)

    table = log.to_frame()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "d_step1_untuned.csv", index=False, float_format="%.4f")

    # Learning Curve Diagnosis: 1 -> lc_trees at lr=0.1
    lgb_lc = lgb.LGBMRegressor(
        n_estimators=lc_trees, learning_rate=lc_lr, random_state=SEED, n_jobs=-1, verbose=-1
    )
    lgb_evals: dict = {}
    lgb_lc.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_va, y_va)],
        eval_names=["train", "val"],
        eval_metric="l1",
        callbacks=[lgb.record_evaluation(lgb_evals)],
    )
    lgb_tr_mae = np.asarray(lgb_evals["train"]["l1"])
    lgb_va_mae = np.asarray(lgb_evals["val"]["l1"])

    xgb_lc = xgb.XGBRegressor(
        n_estimators=lc_trees, learning_rate=lc_lr, tree_method="hist",
        eval_metric="mae", random_state=SEED, n_jobs=-1,
    )
    xgb_lc.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], verbose=False)
    xgb_evals = xgb_lc.evals_result()
    xgb_tr_mae = np.asarray(xgb_evals["validation_0"]["mae"])
    xgb_va_mae = np.asarray(xgb_evals["validation_1"]["mae"])

    rf_val_mae = _read_part_c_rf_val_mae()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    iters = np.arange(1, lc_trees + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    lc_summary = {}
    for ax, name, tr_curve, va_curve in [
        (axes[0], "LightGBM (num_leaves=31, lr=0.1, L2 loss)", lgb_tr_mae, lgb_va_mae),
        (axes[1], "XGBoost (max_depth=6, lr=0.1, L2 loss)", xgb_tr_mae, xgb_va_mae),
    ]:
        best_idx = int(np.argmin(va_curve))
        best_tree = best_idx + 1
        best_va = float(va_curve[best_idx])
        lc_summary[name] = {
            "best_tree": best_tree,
            "best_val_mae": best_va,
            "train_mae_at_best": float(tr_curve[best_idx]),
            "val_mae_at_end": float(va_curve[-1]),
            "train_mae_at_end": float(tr_curve[-1]),
        }
        ax.plot(iters, tr_curve, label="Train MAE", color="#1f77b4", lw=1.8)
        ax.plot(iters, va_curve, label="Validation MAE", color="#d62728", lw=2.0)
        ax.axvline(best_tree, color="green", ls="--", lw=1.2, label=f"Best Val @ tree {best_tree} ({best_va:,.0f} MW)")
        if rf_val_mae is not None:
            ax.axhline(rf_val_mae, color="gray", ls=":", lw=1.4, label=f"Part C RF Val MAE ({rf_val_mae:,.0f} MW)")
        ax.set_title(name, fontsize=10.5)
        ax.set_xlabel("Number of Trees (Boosting Iterations)")
        ax.set_ylabel("MAE (MW)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("Step 1 Diagnosis: Train vs. Validation MAE Learning Curves (No Regularization Tuning)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "d_step1_learning_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return table, lc_summary


# ============================================================================
# STEP 2A: Controlled Loss Function Comparison (L2 vs L1 vs Huber on train->val)
# ============================================================================
def step2a_loss_comparison(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """Compare L2 (MSE), L1 (MAE), and Huber (delta=500 MW) under fixed tree settings."""
    X_tr, y_tr = train[cols], train[TARGET]
    X_va, y_va = val[cols], val[TARGET]
    rows: list[dict] = []

    # 1. LightGBM fixed configuration (only `objective` varies)
    for loss_label, obj_name in [
        ("L2 (MSE)", "regression"),
        ("L1 (MAE)", "regression_l1"),
        ("Huber (delta=500 MW)", "huber"),
    ]:
        params = {
            "objective": obj_name,
            "n_estimators": 3000,
            "learning_rate": 0.05,
            "max_depth": 8,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        if obj_name == "huber":
            params["alpha"] = 500.0

        m, fit_s = _fit_lgbm(
            params, X_tr, y_tr, cat_cols=LGBM_CAT_COLS,
            eval_set=[(X_va, y_va)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
        )
        best_iter = int(m.best_iteration_ or params["n_estimators"])
        s_tr = evaluate(y_tr, m.predict(X_tr))
        s_va = evaluate(y_va, m.predict(X_va))
        rows.append(
            {
                "model_family": "LightGBM",
                "loss": loss_label,
                "objective": obj_name,
                "best_iteration": best_iter,
                "train_MAE": s_tr["MAE"],
                "train_RMSE": s_tr["RMSE"],
                "val_MAE": s_va["MAE"],
                "val_RMSE": s_va["RMSE"],
                "val_MAPE": s_va["MAPE"],
                "val_sMAPE": s_va["sMAPE"],
                "val_R2": s_va["R2"],
                "val_MaxAE": s_va["MaxAE"],
                "val_ME": s_va["ME"],
                "gap_MAE": s_va["MAE"] - s_tr["MAE"],
                "fit_s": fit_s,
            }
        )

    # 2. XGBoost fixed configuration (only `objective` varies)
    for loss_label, obj_name in [
        ("L2 (MSE)", "reg:squarederror"),
        ("L1 (MAE)", "reg:absoluteerror"),
        ("Huber (delta=500 MW)", "reg:pseudohubererror"),
    ]:
        params = {
            "objective": obj_name,
            "eval_metric": "mae",
            "n_estimators": 3000,
            "learning_rate": 0.05,
            "max_depth": 7,
            "min_child_weight": 15.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "early_stopping_rounds": 80,
            "tree_method": "hist",
            "random_state": SEED,
            "n_jobs": -1,
        }
        if obj_name == "reg:pseudohubererror":
            params["huber_slope"] = 500.0
            params["max_delta_step"] = 1500.0

        m, fit_s = _fit_xgb(params, X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best_iter = int(m.best_iteration + 1)
        s_tr = evaluate(y_tr, m.predict(X_tr))
        s_va = evaluate(y_va, m.predict(X_va))
        rows.append(
            {
                "model_family": "XGBoost",
                "loss": loss_label,
                "objective": obj_name,
                "best_iteration": best_iter,
                "train_MAE": s_tr["MAE"],
                "train_RMSE": s_tr["RMSE"],
                "val_MAE": s_va["MAE"],
                "val_RMSE": s_va["RMSE"],
                "val_MAPE": s_va["MAPE"],
                "val_sMAPE": s_va["sMAPE"],
                "val_R2": s_va["R2"],
                "val_MaxAE": s_va["MaxAE"],
                "val_ME": s_va["ME"],
                "gap_MAE": s_va["MAE"] - s_tr["MAE"],
                "fit_s": fit_s,
            }
        )

    df_loss = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_loss.to_csv(RESULTS_DIR / "d_step2_loss_comparison.csv", index=False, float_format="%.4f")
    return df_loss


# ============================================================================
# STEP 2B: Optuna TPE Hyperparameter Tuning on train -> val
# ============================================================================
def _tune_lgbm_optuna(
    train: pd.DataFrame, val: pd.DataFrame, cols: list[str], n_trials: int = 40
) -> tuple[dict, list[str], pd.DataFrame]:
    X_tr, y_tr = train[cols], train[TARGET]
    X_va, y_va = val[cols], val[TARGET]
    rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        obj_name = trial.suggest_categorical("objective", ["regression_l1", "regression", "huber"])
        use_cat = trial.suggest_categorical("use_categorical", [True, False])
        max_depth = trial.suggest_int("max_depth", 5, 12)
        max_leaves = min(127, (2**max_depth) - 1)
        num_leaves = trial.suggest_int("num_leaves", 23, max(24, max_leaves))

        params = {
            "objective": obj_name,
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 15.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        huber_alpha = np.nan
        if obj_name == "huber":
            huber_alpha = trial.suggest_float("alpha", 150.0, 1200.0, log=True)
            params["alpha"] = huber_alpha

        cat_cols = LGBM_CAT_COLS if use_cat else []
        m, fit_s = _fit_lgbm(
            params, X_tr, y_tr, cat_cols=cat_cols,
            eval_set=[(X_va, y_va)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
        )
        best_iter = int(m.best_iteration_ or params["n_estimators"])
        s_tr = evaluate(y_tr, m.predict(X_tr))
        s_va = evaluate(y_va, m.predict(X_va))

        rows.append(
            {
                "trial": trial.number,
                "objective": obj_name,
                "use_categorical": use_cat,
                "alpha": huber_alpha,
                "learning_rate": params["learning_rate"],
                "max_depth": max_depth,
                "num_leaves": num_leaves,
                "min_child_samples": params["min_child_samples"],
                "subsample": params["subsample"],
                "colsample_bytree": params["colsample_bytree"],
                "reg_alpha": params["reg_alpha"],
                "reg_lambda": params["reg_lambda"],
                "best_iteration": best_iter,
                "train_MAE": s_tr["MAE"],
                "train_RMSE": s_tr["RMSE"],
                **s_va,
                "gap_MAE": s_va["MAE"] - s_tr["MAE"],
                "fit_s": fit_s,
            }
        )
        return s_va["MAE"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    for init_obj, init_cat in [
        ("regression_l1", True),
        ("regression", True),
        ("huber", True),
        ("huber", False),
    ]:
        seed_trial = {
            "objective": init_obj,
            "use_categorical": init_cat,
            "max_depth": 8,
            "num_leaves": 63,
            "learning_rate": 0.04,
            "min_child_samples": 15,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }
        if init_obj == "huber":
            seed_trial["alpha"] = 500.0
        study.enqueue_trial(seed_trial)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    curve = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    best = curve.loc[curve["MAE"].idxmin()]
    best_params = {
        "objective": str(best["objective"]),
        "n_estimators": int(best["best_iteration"]),
        "learning_rate": float(best["learning_rate"]),
        "max_depth": int(best["max_depth"]),
        "num_leaves": int(best["num_leaves"]),
        "min_child_samples": int(best["min_child_samples"]),
        "subsample": float(best["subsample"]),
        "subsample_freq": 1,
        "colsample_bytree": float(best["colsample_bytree"]),
        "reg_alpha": float(best["reg_alpha"]),
        "reg_lambda": float(best["reg_lambda"]),
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    if best_params["objective"] == "huber":
        best_params["alpha"] = float(best["alpha"])
    best_cat_cols = LGBM_CAT_COLS if bool(best["use_categorical"]) else []

    return best_params, best_cat_cols, curve


def _tune_xgb_optuna(
    train: pd.DataFrame, val: pd.DataFrame, cols: list[str], n_trials: int = 30
) -> tuple[dict, pd.DataFrame]:
    X_tr, y_tr = train[cols], train[TARGET]
    X_va, y_va = val[cols], val[TARGET]
    rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        obj_name = trial.suggest_categorical(
            "objective", ["reg:absoluteerror", "reg:squarederror", "reg:pseudohubererror"]
        )
        params = {
            "objective": obj_name,
            "eval_metric": "mae",
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.07, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 15.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 30.0, log=True),
            "early_stopping_rounds": 80,
            "tree_method": "hist",
            "random_state": SEED,
            "n_jobs": -1,
        }
        huber_slope = np.nan
        if obj_name == "reg:pseudohubererror":
            huber_slope = trial.suggest_float("huber_slope", 150.0, 1200.0, log=True)
            params["huber_slope"] = huber_slope
            params["max_delta_step"] = 1500.0

        m, fit_s = _fit_xgb(params, X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best_iter = int(m.best_iteration + 1)
        s_tr = evaluate(y_tr, m.predict(X_tr))
        s_va = evaluate(y_va, m.predict(X_va))

        rows.append(
            {
                "trial": trial.number,
                "objective": obj_name,
                "huber_slope": huber_slope,
                "learning_rate": params["learning_rate"],
                "max_depth": params["max_depth"],
                "min_child_weight": params["min_child_weight"],
                "subsample": params["subsample"],
                "colsample_bytree": params["colsample_bytree"],
                "reg_alpha": params["reg_alpha"],
                "reg_lambda": params["reg_lambda"],
                "best_iteration": best_iter,
                "train_MAE": s_tr["MAE"],
                "train_RMSE": s_tr["RMSE"],
                **s_va,
                "gap_MAE": s_va["MAE"] - s_tr["MAE"],
                "fit_s": fit_s,
            }
        )
        return s_va["MAE"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    for init_obj in ("reg:absoluteerror", "reg:squarederror", "reg:pseudohubererror"):
        seed_trial = {
            "objective": init_obj,
            "learning_rate": 0.04,
            "max_depth": 7,
            "min_child_weight": 10.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }
        if init_obj == "reg:pseudohubererror":
            seed_trial["huber_slope"] = 500.0
        study.enqueue_trial(seed_trial)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    curve = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    best = curve.loc[curve["MAE"].idxmin()]
    best_params = {
        "objective": str(best["objective"]),
        "n_estimators": int(best["best_iteration"]),
        "learning_rate": float(best["learning_rate"]),
        "max_depth": int(best["max_depth"]),
        "min_child_weight": float(best["min_child_weight"]),
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "reg_alpha": float(best["reg_alpha"]),
        "reg_lambda": float(best["reg_lambda"]),
        "tree_method": "hist",
        "random_state": SEED,
        "n_jobs": -1,
    }
    if best_params["objective"] == "reg:pseudohubererror":
        best_params["huber_slope"] = float(best["huber_slope"])
        best_params["max_delta_step"] = 1500.0

    return best_params, curve


def step2b_optuna_tuning(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cols: list[str],
    lgb_trials: int = 40,
    xgb_trials: int = 30,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """Run Optuna TPE hyperparameter tuning for LightGBM & XGBoost on train -> val."""
    t0 = time.perf_counter()
    best_lgb, best_lgb_cats, curve_lgb = _tune_lgbm_optuna(train, val, cols, n_trials=lgb_trials)
    t_lgb = time.perf_counter() - t0
    print(
        f"  Finished LightGBM Optuna ({lgb_trials} trials) in {t_lgb:.1f}s -> "
        f"best Val MAE = {curve_lgb['MAE'].min():.2f} MW (categorical={bool(best_lgb_cats)})"
    )

    t0 = time.perf_counter()
    best_xgb, curve_xgb = _tune_xgb_optuna(train, val, cols, n_trials=xgb_trials)
    t_xgb = time.perf_counter() - t0
    print(f"  Finished XGBoost  Optuna ({xgb_trials} trials) in {t_xgb:.1f}s -> best Val MAE = {curve_xgb['MAE'].min():.2f} MW")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    curve_lgb.to_csv(RESULTS_DIR / "d_step2_tuning_lgbm.csv", index=False, float_format="%.4f")
    curve_xgb.to_csv(RESULTS_DIR / "d_step2_tuning_xgb.csv", index=False, float_format="%.4f")

    best_lgb_row = curve_lgb.loc[curve_lgb["MAE"].idxmin()].to_dict()
    best_xgb_row = curve_xgb.loc[curve_xgb["MAE"].idxmin()].to_dict()
    summary_keys = ("MAE", "RMSE", "MAPE", "sMAPE", "R2", "MaxAE", "ME", "train_MAE", "gap_MAE", "best_iteration")
    payload = {
        "lgbm_best_params": best_lgb,
        "lgbm_categorical_features": best_lgb_cats,
        "xgb_best_params": best_xgb,
        "lgbm_best_val_summary": {k: round(float(best_lgb_row[k]), 4) for k in summary_keys if k in best_lgb_row},
        "xgb_best_val_summary": {k: round(float(best_xgb_row[k]), 4) for k in summary_keys if k in best_xgb_row},
    }
    (RESULTS_DIR / "d_step2_best_params.json").write_text(json.dumps(payload, indent=2))

    # Plot Optuna Convergence Curves vs. Step 1 Default & Part C Random Forest
    rf_val_mae = _read_part_c_rf_val_mae()
    step1_defaults: dict[str, float] = {}
    s1_path = RESULTS_DIR / "d_step1_untuned.csv"
    if s1_path.exists():
        df_s1 = pd.read_csv(s1_path).query("split == 'val'")
        for _, r in df_s1.iterrows():
            step1_defaults[r["model"]] = float(r["MAE"])

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    palette = {
        "regression_l1": "#1f77b4",
        "reg:absoluteerror": "#1f77b4",
        "regression": "#ff7f0e",
        "reg:squarederror": "#ff7f0e",
        "huber": "#2ca02c",
        "reg:pseudohubererror": "#2ca02c",
    }
    short_lbl = {
        "regression_l1": "L1 (MAE)",
        "reg:absoluteerror": "L1 (MAE)",
        "regression": "L2 (MSE)",
        "reg:squarederror": "L2 (MSE)",
        "huber": "Huber",
        "reg:pseudohubererror": "Huber",
    }

    for ax, curve, title, def_key in [
        (axes[0], curve_lgb, f"LightGBM — Optuna TPE ({lgb_trials} Trials)", "LightGBM (default)"),
        (axes[1], curve_xgb, f"XGBoost — Optuna TPE ({xgb_trials} Trials)", "XGBoost (default)"),
    ]:
        for obj, grp in curve.groupby("objective"):
            ax.scatter(
                grp["trial"] + 1, grp["MAE"],
                color=palette.get(obj, "purple"), s=36, alpha=0.82,
                label=f"Trial: {short_lbl.get(obj, obj)}",
            )
        ax.plot(curve["trial"] + 1, curve["MAE"].cummin(), color="black", lw=2.1, label=f"Best Val MAE ({curve['MAE'].min():,.0f} MW)")
        if def_key in step1_defaults:
            ax.axhline(step1_defaults[def_key], color="#d62728", ls="--", lw=1.3, label=f"Step 1 Default ({step1_defaults[def_key]:,.0f} MW)")
        if rf_val_mae is not None:
            ax.axhline(rf_val_mae, color="gray", ls=":", lw=1.5, label=f"Part C RF Val MAE ({rf_val_mae:,.0f} MW)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Optuna Trial Number")
        ax.set_ylabel("Validation MAE (MW)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Step 2B: Hyperparameter Tuning Convergence on Validation Set (Train -> Val)", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "d_step2_tuning_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return best_lgb, best_xgb, curve_lgb, curve_xgb


# ============================================================================
# STEP 3: Final Test Evaluation (ONCE), 5-Seed Stability & Horizon Comparison
# ============================================================================
def _load_frozen_step2_params() -> tuple[dict, dict, list[str]]:
    p_path = RESULTS_DIR / "d_step2_best_params.json"
    if not p_path.exists():
        raise FileNotFoundError(f"Frozen Step 2 params not found at {p_path}. Run --step 2 first.")
    payload = json.loads(p_path.read_text())
    return (
        payload["lgbm_best_params"],
        payload["xgb_best_params"],
        list(payload.get("lgbm_categorical_features", [])),
    )


def step3_final_test_evaluation(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict:
    """
    Task 3A: Fit on train -> val AND refit on train+val -> test (ONCE) via ResultsLog.
    Task 3B: 5-seed stability check on train+val -> test (mean ± std).
    Task 3C: Extended horizon comparison (h=1 vs h=24 on val).
    Task 3D: 6 diagnostic figures using evaluate.py plotting utilities.
    """
    best_lgb, best_xgb, lgb_cat_cols = _load_frozen_step2_params()
    trval = pd.concat([train, val], ignore_index=True)

    # ---- Task 3A: Official Val + Test Evaluation via evaluate.ResultsLog ----
    log = ResultsLog()
    preds_boost_test: dict[str, np.ndarray] = {}

    # 1. Default LightGBM (train->val and train+val->test for Untuned vs Tuned ablation)
    m_lgb_def_va, s_lgb_def_va = timed_fit(
        lgb.LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1), train[cols], train[TARGET]
    )
    log.add("LightGBM (default)", "val", val[TARGET], m_lgb_def_va.predict(val[cols]), family="boosting_default", fit_s=s_lgb_def_va)
    m_lgb_def_te, s_lgb_def_te = timed_fit(
        lgb.LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1), trval[cols], trval[TARGET]
    )
    log.add("LightGBM (default)", "test", test[TARGET], m_lgb_def_te.predict(test[cols]), family="boosting_default", fit_s=s_lgb_def_te)

    # 2. Default XGBoost
    m_xgb_def_va, s_xgb_def_va = timed_fit(
        xgb.XGBRegressor(tree_method="hist", random_state=SEED, n_jobs=-1), train[cols], train[TARGET]
    )
    log.add("XGBoost (default)", "val", val[TARGET], m_xgb_def_va.predict(val[cols]), family="boosting_default", fit_s=s_xgb_def_va)
    m_xgb_def_te, s_xgb_def_te = timed_fit(
        xgb.XGBRegressor(tree_method="hist", random_state=SEED, n_jobs=-1), trval[cols], trval[TARGET]
    )
    log.add("XGBoost (default)", "test", test[TARGET], m_xgb_def_te.predict(test[cols]), family="boosting_default", fit_s=s_xgb_def_te)

    # 3. Tuned LightGBM (frozen Step 2 config)
    m_lgb_va, s_lgb_va = _fit_lgbm(best_lgb, train[cols], train[TARGET], cat_cols=lgb_cat_cols)
    log.add("LightGBM", "val", val[TARGET], m_lgb_va.predict(val[cols]), family="boosting", fit_s=s_lgb_va)
    m_lgb_te, s_lgb_te = _fit_lgbm(best_lgb, trval[cols], trval[TARGET], cat_cols=lgb_cat_cols)
    p_lgb_test = m_lgb_te.predict(test[cols])
    log.add("LightGBM", "test", test[TARGET], p_lgb_test, family="boosting", fit_s=s_lgb_te)
    preds_boost_test["LightGBM"] = p_lgb_test

    # 4. Tuned XGBoost (frozen Step 2 config)
    m_xgb_va, s_xgb_va = _fit_xgb(best_xgb, train[cols], train[TARGET])
    log.add("XGBoost", "val", val[TARGET], m_xgb_va.predict(val[cols]), family="boosting", fit_s=s_xgb_va)
    m_xgb_te, s_xgb_te = _fit_xgb(best_xgb, trval[cols], trval[TARGET])
    p_xgb_test = m_xgb_te.predict(test[cols])
    log.add("XGBoost", "test", test[TARGET], p_xgb_test, family="boosting", fit_s=s_xgb_te)
    preds_boost_test["XGBoost"] = p_xgb_test

    boost_table = log.to_frame()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    boost_table.to_csv(RESULTS_DIR / "metrics_boosting.csv", index=False, float_format="%.4f")

    # Merge Part C metrics.csv + Part D metrics_boosting.csv into metrics_all.csv (non-destructive)
    c_metrics_path = RESULTS_DIR / "metrics.csv"
    if c_metrics_path.exists():
        c_table = pd.read_csv(c_metrics_path)
        all_table = pd.concat([c_table, boost_table], ignore_index=True)
    else:
        all_table = boost_table.copy()
    all_table.to_csv(RESULTS_DIR / "metrics_all.csv", index=False, float_format="%.4f")

    # Merge Part C test predictions + Part D test predictions by `time` into predictions_test_all.parquet
    c_preds_path = RESULTS_DIR / "predictions_test.parquet"
    preds_all: dict[str, np.ndarray] = {}
    if c_preds_path.exists():
        pred_df = pd.read_parquet(c_preds_path)
        boost_pred_df = pd.DataFrame({"time": test["time"].to_numpy(), **preds_boost_test})
        pred_df = pred_df.drop(columns=[c for c in preds_boost_test if c in pred_df.columns])
        pred_df = pred_df.merge(boost_pred_df, on="time", how="left", validate="one_to_one")
        assert len(pred_df) == len(test), "Row count mismatch after merging predictions by time"
        for c in pred_df.columns:
            if c not in ("time", "actual"):
                preds_all[c] = pred_df[c].to_numpy()
        pred_df.to_parquet(RESULTS_DIR / "predictions_test_all.parquet", index=False)
    else:
        preds_all.update(preds_boost_test)

    # Disaggregated MAE by local hour & day-of-week using evaluate.error_by_group()
    err_hour_all = pd.DataFrame(
        {k: error_by_group(test["time"], test[TARGET], v, "hour") for k, v in preds_all.items()}
    )
    err_hour_all.to_csv(RESULTS_DIR / "mae_by_hour_all.csv", float_format="%.4f")
    err_dow_all = pd.DataFrame(
        {k: error_by_group(test["time"], test[TARGET], v, "dayofweek") for k, v in preds_all.items()}
    )
    err_dow_all.to_csv(RESULTS_DIR / "mae_by_dow_all.csv", float_format="%.4f")

    # ---- Task 3B: 5-Seed Stability Check on train+val -> test ----
    ms_rows: list[dict] = []
    for m_name, is_lgb, base_p in [
        ("LightGBM", True, best_lgb),
        ("XGBoost", False, best_xgb),
    ]:
        seed_scores = []
        for s in seeds:
            p = dict(base_p)
            p["random_state"] = s
            if is_lgb:
                m, _ = _fit_lgbm(p, trval[cols], trval[TARGET], cat_cols=lgb_cat_cols)
            else:
                m, _ = _fit_xgb(p, trval[cols], trval[TARGET])
            seed_scores.append(evaluate(test[TARGET], m.predict(test[cols])))
        df_s = pd.DataFrame(seed_scores)
        ms_rows.append(
            {
                "model": m_name,
                "n_seeds": len(seeds),
                "MAE_mean": float(df_s["MAE"].mean()),
                "MAE_std": float(df_s["MAE"].std()),
                "RMSE_mean": float(df_s["RMSE"].mean()),
                "RMSE_std": float(df_s["RMSE"].std()),
                "MAPE_mean": float(df_s["MAPE"].mean()),
                "MAPE_std": float(df_s["MAPE"].std()),
                "R2_mean": float(df_s["R2"].mean()),
                "R2_std": float(df_s["R2"].std()),
                "ME_mean": float(df_s["ME"].mean()),
                "ME_std": float(df_s["ME"].std()),
            }
        )
    multiseed_df = pd.DataFrame(ms_rows)
    multiseed_df.to_csv(RESULTS_DIR / "d_step3_multiseed.csv", index=False, float_format="%.4f")

    # ---- Task 3C: Extended Horizon Comparison (h=1 vs h=24 on val) ----
    h_rows: list[dict] = []
    c_horizon_path = RESULTS_DIR / "horizon_comparison.csv"
    if c_horizon_path.exists():
        h_rows.extend(pd.read_csv(c_horizon_path).to_dict("records"))

    for h in (1, 24):
        h_data = make_datasets(FeatureConfig(horizon=h))
        h_tr, h_va = h_data["train"], h_data["val"]
        h_cols = feature_columns(h_tr)

        m_lgb_h, _ = _fit_lgbm(best_lgb, h_tr[h_cols], h_tr[TARGET], cat_cols=lgb_cat_cols)
        s_lgb_h = evaluate(h_va[TARGET], m_lgb_h.predict(h_va[h_cols]))
        h_rows.append({"horizon_h": h, "model": "LightGBM", "MAE": s_lgb_h["MAE"], "RMSE": s_lgb_h["RMSE"], "MAPE": s_lgb_h["MAPE"]})

        m_xgb_h, _ = _fit_xgb(best_xgb, h_tr[h_cols], h_tr[TARGET])
        s_xgb_h = evaluate(h_va[TARGET], m_xgb_h.predict(h_va[h_cols]))
        h_rows.append({"horizon_h": h, "model": "XGBoost", "MAE": s_xgb_h["MAE"], "RMSE": s_xgb_h["RMSE"], "MAPE": s_xgb_h["MAPE"]})

    horizon_df = (
        pd.DataFrame(h_rows)
        .drop_duplicates(subset=["horizon_h", "model"], keep="last")
        .sort_values(["horizon_h", "MAE"])
        .reset_index(drop=True)
    )
    horizon_df.to_csv(RESULTS_DIR / "horizon_comparison_all.csv", index=False, float_format="%.4f")

    # ---- Task 3D: 6 Diagnostic Plots (using evaluate.py functions) ----
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    # Exclude default untuned rows from the clean all-models bar chart (keep main models + TSO reference)
    main_bar_table = all_table[~all_table["model"].str.contains(r"\(default\)")].copy()
    plot_metric_bars(main_bar_table, "MAE", PLOT_DIR / "d_step3_mae_test_all.png", split="test")
    plot_metric_bars(main_bar_table, "MAPE", PLOT_DIR / "d_step3_mape_test_all.png", split="test")

    overlay = {k: preds_all[k] for k in ["Random Forest", "LightGBM", "XGBoost"] if k in preds_all}
    plot_forecast(
        test["time"], test[TARGET], overlay,
        PLOT_DIR / "d_step3_forecast_2weeks.png",
        title="Test set: two weeks of day-ahead forecasts (Random Forest vs. LightGBM vs. XGBoost)",
        start="2018-11-05", days=14,
    )
    plot_forecast(
        test["time"], test[TARGET], overlay,
        PLOT_DIR / "d_step3_forecast_holiday.png",
        title="Test set: week containing 12 Oct National Holiday (RF vs. LightGBM vs. XGBoost)",
        start="2018-10-08", days=7,
    )

    # Hourly MAE comparison plot
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    for k in ["Naive (t-24h)", "Ridge", "Random Forest", "LightGBM", "XGBoost"]:
        if k in err_hour_all.columns:
            lw = 2.1 if k in ("LightGBM", "XGBoost") else 1.25
            ax.plot(err_hour_all.index, err_hour_all[k], "o-", ms=3.5, lw=lw, label=k)
    ax.set_xlabel("Local hour (Europe/Madrid)")
    ax.set_ylabel("Test MAE (MW)")
    ax.set_title("Hourly Test MAE Across Model Families (Day-Ahead, h = 24h)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "d_step3_mae_by_hour_all.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Residual diagnostics for the best boosting model on Test (selected by Val MAE per protocol!)
    val_mae_lgb = float(boost_table.query("model == 'LightGBM' and split == 'val'")["MAE"].iloc[0])
    val_mae_xgb = float(boost_table.query("model == 'XGBoost' and split == 'val'")["MAE"].iloc[0])
    best_name = "LightGBM" if val_mae_lgb <= val_mae_xgb else "XGBoost"
    plot_residuals(
        test[TARGET], preds_boost_test[best_name],
        PLOT_DIR / "d_step3_residuals.png",
        f"{best_name} (Tuned) — Test Set Residuals",
    )

    return {
        "all_table": all_table,
        "boost_table": boost_table,
        "multiseed_df": multiseed_df,
        "horizon_df": horizon_df,
        "err_hour_all": err_hour_all,
        "err_dow_all": err_dow_all,
        "best_by_val": best_name,
    }


# ============================================================================
# STEP 4: Exact TreeSHAP Explainability on Test Set (+ Fair Val Comparison)
# ============================================================================
def _feature_family(col: str) -> str:
    if col.startswith("cal_"):
        return "Calendar (18)"
    if col.startswith(("lag_", "roll_", "diff_")):
        return "Lag / Rolling (13)"
    if col.startswith("w_"):
        return "Weather (11)"
    return "Other"


def step4_shap_explainability(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
) -> dict:
    """
    Task 4A: Compute exact TreeSHAP on full Test set (5,260 h) for LightGBM & XGBoost,
             plus matched Validation TreeSHAP (train -> val) for fair comparison with
             Part C's RF Permutation Importance (per docs/HANDOFF_PART_D.md).
    Task 4B: Global importance table vs. Part C RF Permutation Importance + 3-family group SHAP.
    Task 4C: Non-linear Dependence plots (lag_24 x cal_is_offday & combined 5-var temperature U-curve).
    Task 4D: 3 Local SHAP Waterfall plots (Peak load, 12 Oct National Holiday, Worst genuine error
             excluding the documented 2018-06-05 10:00 UTC sensor glitch).
    """
    best_lgb, best_xgb, lgb_cat_cols = _load_frozen_step2_params()
    trval = pd.concat([train, val], ignore_index=True)

    # Fit frozen winning configurations on train + val (identical to Step 3 final models)
    lgb_model, _ = _fit_lgbm(best_lgb, trval[cols], trval[TARGET], cat_cols=lgb_cat_cols)
    xgb_model, _ = _fit_xgb(best_xgb, trval[cols], trval[TARGET])

    # Also fit LightGBM on train -> val for fair apples-to-apples comparison with RF permutation importance
    lgb_val_model, _ = _fit_lgbm(best_lgb, train[cols], train[TARGET], cat_cols=lgb_cat_cols)

    X_va = val[cols].copy()
    X_te = test[cols].copy()
    y_te = test[TARGET].to_numpy()
    y_pred_lgb = lgb_model.predict(X_te)
    y_pred_xgb = xgb_model.predict(X_te)

    # ---- Task 4A: Exact TreeSHAP + Local Accuracy Axiom Check ----
    t0 = time.perf_counter()
    lgb_explainer = shap.TreeExplainer(lgb_model)
    lgb_exp = lgb_explainer(X_te)
    t_lgb_shap = time.perf_counter() - t0

    lgb_recon = lgb_exp.base_values + lgb_exp.values.sum(axis=1)
    lgb_max_err = float(np.max(np.abs(y_pred_lgb - lgb_recon)))
    assert lgb_max_err < 1e-2, f"LightGBM TreeSHAP additivity error too large: {lgb_max_err}"

    lgb_val_exp = shap.TreeExplainer(lgb_val_model)(X_va)
    lgb_val_recon = lgb_val_exp.base_values + lgb_val_exp.values.sum(axis=1)
    lgb_val_max_err = float(np.max(np.abs(lgb_val_model.predict(X_va) - lgb_val_recon)))
    assert lgb_val_max_err < 1e-2, f"LightGBM Val TreeSHAP additivity error too large: {lgb_val_max_err}"

    t0 = time.perf_counter()
    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_exp = xgb_explainer(X_te)
    t_xgb_shap = time.perf_counter() - t0

    xgb_recon = xgb_exp.base_values + xgb_exp.values.sum(axis=1)
    xgb_max_err = float(np.max(np.abs(y_pred_xgb - xgb_recon)))
    assert xgb_max_err < 1.0, f"XGBoost TreeSHAP additivity error too large: {xgb_max_err}"

    lgb_mean_abs = np.abs(lgb_exp.values).mean(axis=0)
    lgb_val_mean_abs = np.abs(lgb_val_exp.values).mean(axis=0)
    xgb_mean_abs = np.abs(xgb_exp.values).mean(axis=0)

    rho_lgb_xgb, _ = spearmanr(lgb_mean_abs, xgb_mean_abs)

    # ---- Task 4B: Global Importance Table vs RF Permutation Importance + 3-Family Group SHAP ----
    shap_df = pd.DataFrame(
        {
            "feature": cols,
            "family": [_feature_family(c) for c in cols],
            "lgbm_mean_abs_shap_mw": lgb_mean_abs,
            "lgbm_share_pct": 100.0 * lgb_mean_abs / lgb_mean_abs.sum(),
            "lgbm_val_mean_abs_shap_mw": lgb_val_mean_abs,
            "lgbm_val_share_pct": 100.0 * lgb_val_mean_abs / lgb_val_mean_abs.sum(),
            "xgb_mean_abs_shap_mw": xgb_mean_abs,
            "xgb_share_pct": 100.0 * xgb_mean_abs / xgb_mean_abs.sum(),
        }
    )
    shap_df["lgbm_val_shap_rank"] = shap_df["lgbm_val_mean_abs_shap_mw"].rank(ascending=False, method="min").astype(int)
    shap_df["xgb_shap_rank"] = shap_df["xgb_mean_abs_shap_mw"].rank(ascending=False, method="min").astype(int)
    shap_df = shap_df.sort_values("lgbm_mean_abs_shap_mw", ascending=False).reset_index(drop=True)
    shap_df["lgbm_shap_rank"] = np.arange(1, len(shap_df) + 1)

    rho_lgb_rf_val = np.nan
    rho_lgb_rf_test = np.nan
    rf_imp_path = RESULTS_DIR / "feature_importance_rf.csv"
    if rf_imp_path.exists():
        rf_imp = pd.read_csv(rf_imp_path)
        rf_imp["rf_perm_rank"] = np.arange(1, len(rf_imp) + 1)
        shap_df = shap_df.merge(
            rf_imp[["feature", "perm_mae_increase", "impurity", "rf_perm_rank"]],
            on="feature",
            how="left",
        )
        valid = shap_df.dropna(subset=["perm_mae_increase"])
        rho_lgb_rf_val, _ = spearmanr(valid["lgbm_val_mean_abs_shap_mw"], valid["perm_mae_increase"])
        rho_lgb_rf_test, _ = spearmanr(valid["lgbm_mean_abs_shap_mw"], valid["perm_mae_increase"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(RESULTS_DIR / "d_step4_shap_importance.csv", index=False, float_format="%.4f")

    # Group-level additive SHAP across Part B's 3 feature families (42 features)
    grp_rows: list[dict] = []
    for fam in ["Lag / Rolling (13)", "Calendar (18)", "Weather (11)"]:
        idxs = [i for i, c in enumerate(cols) if _feature_family(c) == fam]
        net_group_shap = lgb_exp.values[:, idxs].sum(axis=1)  # exact additive sum per hour!
        mean_abs_net = float(np.mean(np.abs(net_group_shap)))
        sum_indiv_abs = float(lgb_mean_abs[idxs].sum())
        xgb_net = float(np.mean(np.abs(xgb_exp.values[:, idxs].sum(axis=1))))
        grp_rows.append(
            {
                "feature_family": fam,
                "n_features": len(idxs),
                "lgbm_mean_abs_net_group_mw": mean_abs_net,
                "lgbm_sum_indiv_mean_abs_mw": sum_indiv_abs,
                "lgbm_sum_indiv_share_pct": 100.0 * sum_indiv_abs / float(lgb_mean_abs.sum()),
                "xgb_mean_abs_net_group_mw": xgb_net,
            }
        )
    group_df = pd.DataFrame(grp_rows)
    group_df["lgbm_net_group_share_pct"] = (
        100.0 * group_df["lgbm_mean_abs_net_group_mw"] / group_df["lgbm_mean_abs_net_group_mw"].sum()
    )
    group_df.to_csv(RESULTS_DIR / "d_step4_shap_by_group.csv", index=False, float_format="%.4f")

    # Plot 4B-1: Beeswarm Summary (Top 15 on Test Set)
    lgb_obj_short = _objective_short_name(best_lgb["objective"])
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9.5, 6.4))
    shap.plots.beeswarm(lgb_exp, max_display=15, show=False)
    plt.title(f"LightGBM (Tuned — {lgb_obj_short}) — TreeSHAP Beeswarm Summary (Top 15, Test Set)", fontsize=11.5)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "d_step4_shap_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Plot 4B-2: Fair Apples-to-Apples Comparison on Val (train -> val): LightGBM TreeSHAP vs. RF Permutation
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6))
    top15_val = shap_df.sort_values("lgbm_val_mean_abs_shap_mw", ascending=False).head(15).iloc[::-1]
    axes[0].barh(top15_val["feature"], top15_val["lgbm_val_mean_abs_shap_mw"], color="steelblue")
    axes[0].set_xlabel("Mean |SHAP value| on Validation Set (MW)")
    axes[0].set_title("LightGBM (train → val) — TreeSHAP Importance (Top 15)")
    axes[0].grid(axis="x", alpha=0.3)

    if "perm_mae_increase" in shap_df.columns:
        top15_rf = shap_df.sort_values("perm_mae_increase", ascending=False).head(15).iloc[::-1]
        axes[1].barh(top15_rf["feature"], top15_rf["perm_mae_increase"], color="seagreen")
        axes[1].set_xlabel("Validation MAE Increase When Permuted (MW)")
        axes[1].set_title("Random Forest (train → val, Part C) — Permutation (Top 15)")
        axes[1].grid(axis="x", alpha=0.3)
    fig.suptitle(
        f"Matched Validation Feature Attribution: LightGBM TreeSHAP vs. RF Permutation (Spearman r = {rho_lgb_rf_val:.3f})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "d_step4_shap_vs_rf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- Task 4C: Non-linear Dependence & Interaction Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    idx_lag24 = cols.index("lag_24")
    sc0 = axes[0].scatter(
        X_te["lag_24"],
        lgb_exp.values[:, idx_lag24],
        c=X_te["cal_is_offday"],
        cmap="coolwarm",
        s=9,
        alpha=0.5,
    )
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_xlabel("lag_24: Electricity Load 24h Earlier (MW)")
    axes[0].set_ylabel("SHAP Value of lag_24 (MW)")
    axes[0].set_title("SHAP Dependence: lag_24 × cal_is_offday Interaction")
    axes[0].grid(alpha=0.25)
    cbar0 = plt.colorbar(sc0, ax=axes[0], ticks=[0, 1])
    cbar0.ax.set_yticklabels(["Workday (0)", "Off-day (1)"])

    # Combined 5-feature temperature SHAP to avoid collinear dilution
    temp_cols = [c for c in ["w_temp_c", "w_temp_c_sq", "w_hdh", "w_cdh", "w_temp_roll24"] if c in cols]
    temp_idxs = [cols.index(c) for c in temp_cols]
    temp_shap_sum = lgb_exp.values[:, temp_idxs].sum(axis=1)
    sc1 = axes[1].scatter(
        X_te["w_temp_c"],
        temp_shap_sum,
        c=X_te["cal_hour"],
        cmap="viridis",
        s=9,
        alpha=0.5,
    )
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].axvline(18.0, color="dimgray", lw=1.0, ls=":", label="Heating base (18°C)")
    axes[1].axvline(22.0, color="firebrick", lw=1.0, ls="-.", label="Cooling base (22°C)")
    axes[1].set_xlabel("w_temp_c: 5-City Mean Temperature (°C)")
    axes[1].set_ylabel("Combined SHAP of 5 Temperature Features (MW)")
    axes[1].set_title("SHAP Dependence: U-Shaped Temperature Response × Hour")
    axes[1].legend(loc="upper center", fontsize=8.5, framealpha=0.88)
    axes[1].grid(alpha=0.25)
    plt.colorbar(sc1, ax=axes[1], label="Local Hour (0–23)")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "d_step4_shap_dependence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- Task 4D: Local SHAP Waterfall Case Studies ----
    abs_err = np.abs(y_te - y_pred_lgb)
    idx_peak = int(np.argmax(y_te))

    utc_ts = pd.to_datetime(test["time"], utc=True)
    local_ts = utc_ts.dt.tz_convert("Europe/Madrid")
    hol_mask = (local_ts.dt.strftime("%Y-%m-%d") == "2018-10-12") & (local_ts.dt.hour == 12)
    if hol_mask.any():
        idx_holiday = int(np.flatnonzero(hol_mask.to_numpy())[0])
    else:
        idx_holiday = int(np.flatnonzero((X_te["cal_is_holiday"] == 1).to_numpy())[0])

    # Documented 1-hour sensor glitch in docs/HANDOFF_PART_D.md (2018-06-05 10:00 UTC)
    glitch_mask = utc_ts.dt.strftime("%Y-%m-%d %H:%M") == KNOWN_TEST_SENSOR_GLITCH_UTC
    abs_err_clean = abs_err.copy()
    if glitch_mask.any():
        idx_glitch = int(np.flatnonzero(glitch_mask.to_numpy())[0])
        abs_err_clean[idx_glitch] = -1.0
    else:
        idx_glitch = None
    idx_worst_genuine = int(np.argmax(abs_err_clean))

    local_rows: list[dict] = []
    cases_to_plot = [
        ("peak", idx_peak, "Peak Demand Hour on Test Set", True),
        ("holiday", idx_holiday, "National Holiday (12 Oct 2018, 12:00 Local)", True),
        ("worst_error", idx_worst_genuine, "Worst Genuine Forecast Error (Excl. Sensor Glitch)", True),
    ]
    if idx_glitch is not None:
        cases_to_plot.append(("sensor_glitch", idx_glitch, "Documented Sensor Glitch (05 Jun 2018 10:00 UTC)", False))

    for tag, idx, title_label, make_plot in cases_to_plot:
        ts_str = local_ts.iloc[idx].strftime("%Y-%m-%d %H:%M %Z (%a)")
        actual_mw = float(y_te[idx])
        pred_mw = float(y_pred_lgb[idx])
        err_mw = actual_mw - pred_mw
        base_mw = float(lgb_exp.base_values[idx])

        vals_row = lgb_exp.values[idx]
        top_order = np.argsort(np.abs(vals_row))[::-1][:5]
        drivers_str = "; ".join(
            f"{cols[j]}={X_te.iloc[idx, j]:.1f} ({vals_row[j]:+.0f} MW)" for j in top_order
        )
        local_rows.append(
            {
                "case": tag,
                "test_row_idx": idx,
                "local_time": ts_str,
                "base_value_mw": base_mw,
                "actual_mw": actual_mw,
                "pred_mw": pred_mw,
                "residual_y_minus_yhat_mw": err_mw,
                "top_5_shap_drivers": drivers_str,
            }
        )

        if make_plot:
            plt.figure(figsize=(11.0, 6.2))
            shap.plots.waterfall(lgb_exp[idx], max_display=12, show=False)
            fig = plt.gcf()
            fig.set_size_inches(11.0, 6.2)
            ax_main = fig.axes[0]
            x0, x1 = ax_main.get_xlim()
            span = max(x1 - x0, 1.0)
            new_xlim = (x0 - 0.16 * span, x1 + 0.12 * span)
            for ax_sub in fig.axes:
                ax_sub.set_xlim(new_xlim)
                for txt in ax_sub.texts:
                    txt.set_fontsize(9.2)
            fig.suptitle(
                f"Local TreeSHAP Waterfall — {title_label}\n"
                f"{ts_str} | Actual: {actual_mw:,.0f} MW, Pred: {pred_mw:,.0f} MW (y - ŷ: {err_mw:+,.0f} MW)",
                fontsize=11.0,
                y=0.99,
            )
            fig.subplots_adjust(top=0.84)
            fig.savefig(PLOT_DIR / f"d_step4_waterfall_{tag}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(RESULTS_DIR / "d_step4_local_cases.csv", index=False)

    return {
        "shap_df": shap_df,
        "group_df": group_df,
        "local_df": local_df,
        "lgb_max_additivity_err": lgb_max_err,
        "lgb_val_max_additivity_err": lgb_val_max_err,
        "xgb_max_additivity_err": xgb_max_err,
        "spearman_lgb_xgb": float(rho_lgb_xgb),
        "spearman_lgb_rf_val": float(rho_lgb_rf_val),
        "spearman_lgb_rf_test": float(rho_lgb_rf_test),
        "lgb_base_value": float(lgb_exp.base_values[0]),
        "lgb_shap_time_s": t_lgb_shap,
        "xgb_shap_time_s": t_xgb_shap,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["1", "2", "3", "4", "all"], default="4")
    ap.add_argument("--lgb-trials", type=int, default=40)
    ap.add_argument("--xgb-trials", type=int, default=30)
    args = ap.parse_args()

    np.random.seed(SEED)
    data = make_datasets(FeatureConfig(horizon=24))
    train, val, test = data["train"], data["val"], data["test"]
    cols = feature_columns(train)

    test_status = "UNLOCKED" if args.step in ("3", "4", "all") else "LOCKED"
    print(
        f"Loaded Part B features (horizon=24h, {len(cols)} features): "
        f"train={train.shape}, val={val.shape}, test={test.shape} [{test_status}]"
    )

    if args.step in ("1", "all"):
        print("\n=== RUNNING STEP 1: DEFAULT BOOSTING & LEARNING CURVES ===")
        step1_table, _ = step1_untuned_and_learning_curve(train, val, cols)
        print(step1_table[["model", "split", "MAE", "RMSE", "MAPE", "R2", "ME", "fit_s"]].round(3).to_string(index=False))

    if args.step in ("2", "all"):
        print("\n=== RUNNING STEP 2A: CONTROLLED LOSS COMPARISON (L2 vs L1 vs Huber) ===")
        df_loss = step2a_loss_comparison(train, val, cols)
        cols_2a = ["model_family", "loss", "best_iteration", "train_MAE", "val_MAE", "gap_MAE", "val_RMSE", "val_MAPE", "val_R2", "val_ME", "fit_s"]
        print(df_loss[cols_2a].round(3).to_string(index=False))

        print(f"\n=== RUNNING STEP 2B: OPTUNA TPE TUNING (LGBM={args.lgb_trials}, XGB={args.xgb_trials}) ===")
        best_lgb, best_xgb, curve_lgb, curve_xgb = step2b_optuna_tuning(
            train, val, cols, lgb_trials=args.lgb_trials, xgb_trials=args.xgb_trials
        )
        print("\n--- Best LightGBM Configuration (Validation) ---")
        print(json.dumps(best_lgb, indent=2))
        print("\n--- Best XGBoost Configuration (Validation) ---")
        print(json.dumps(best_xgb, indent=2))

    if args.step in ("3", "all"):
        print("\n=== RUNNING STEP 3: FINAL TEST EVALUATION, 5-SEED STABILITY & HORIZON COMPARISON ===")
        out3 = step3_final_test_evaluation(train, val, test, cols)
        with pd.option_context("display.width", 220, "display.max_columns", 20):
            print("\n--- 3A. COMBINED METRICS ON TEST SPLIT (PART C + PART D) ---")
            te_tbl = out3["all_table"].query("split == 'test'")[["model", "MAE", "RMSE", "MAPE", "sMAPE", "R2", "MaxAE", "ME", "fit_s"]]
            print(te_tbl.round(3).to_string(index=False))

            print("\n--- 3B. 5-SEED STABILITY CHECK ON TEST (SEEDS 0..4) ---")
            print(out3["multiseed_df"].round(3).to_string(index=False))

            print("\n--- 3C. HORIZON COMPARISON (h=1h vs h=24h ON VALIDATION) ---")
            print(out3["horizon_df"].round(3).to_string(index=False))

    if args.step in ("4", "all"):
        print("\n=== RUNNING STEP 4: EXACT TREESHAP EXPLAINABILITY ON TEST SET (+ FAIR VAL COMPARISON) ===")
        out4 = step4_shap_explainability(train, val, test, cols)
        print(
            f"  TreeSHAP runtime : LightGBM = {out4['lgb_shap_time_s']:.2f}s | XGBoost = {out4['xgb_shap_time_s']:.2f}s\n"
            f"  Base value (phi0): LightGBM = {out4['lgb_base_value']:,.2f} MW\n"
            f"  Local Accuracy   : max |y_hat - (phi0 + sum phi_j)| -> "
            f"LGBM(test) = {out4['lgb_max_additivity_err']:.2e} MW | LGBM(val) = {out4['lgb_val_max_additivity_err']:.2e} MW | "
            f"XGB(test) = {out4['xgb_max_additivity_err']:.2e} MW\n"
            f"  Rank Agreement   : Spearman(LGBM Test SHAP, XGB Test SHAP) = {out4['spearman_lgb_xgb']:.4f} | "
            f"Spearman(LGBM Val SHAP, RF Val Perm [fair]) = {out4['spearman_lgb_rf_val']:.4f} | "
            f"Spearman(LGBM Test SHAP, RF Val Perm) = {out4['spearman_lgb_rf_test']:.4f}"
        )
        with pd.option_context("display.width", 220, "display.max_columns", 20):
            print("\n--- 4B. TOP 15 FEATURES: LIGHTGBM SHAP (TEST & VAL) vs XGBOOST SHAP vs RF PERMUTATION ---")
            show_shap = [
                "lgbm_shap_rank", "feature", "family", "lgbm_mean_abs_shap_mw",
                "lgbm_share_pct", "lgbm_val_shap_rank", "xgb_shap_rank", "perm_mae_increase", "rf_perm_rank"
            ]
            show_shap = [c for c in show_shap if c in out4["shap_df"].columns]
            print(out4["shap_df"][show_shap].head(15).round(2).to_string(index=False))

            print("\n--- 4B (cont). GROUP-AGGREGATED SHAP ACROSS PART B'S 3 FEATURE FAMILIES ---")
            print(out4["group_df"].round(2).to_string(index=False))

            print("\n--- 4D. LOCAL CASE STUDIES ON TEST ---")
            for _, r in out4["local_df"].iterrows():
                print(
                    f"  [{r['case']:13s}] {r['local_time']} | Actual={r['actual_mw']:,.0f} MW, "
                    f"Pred={r['pred_mw']:,.0f} MW (Err={r['residual_y_minus_yhat_mw']:+,.0f} MW)\n"
                    f"                  Top 5 SHAP: {r['top_5_shap_drivers']}"
                )


if __name__ == "__main__":
    main()
