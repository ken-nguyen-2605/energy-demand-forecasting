"""Tests for Part D Gradient Boosting (LightGBM, XGBoost) & Exact TreeSHAP pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import shap
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate import ResultsLog, evaluate  # noqa: E402
from src.features import TARGET, TSO_FORECAST, FeatureConfig, feature_columns, make_datasets  # noqa: E402
from src.train_boosting import _feature_family  # noqa: E402

RESULTS_DIR = ROOT_DIR / "results"
PLOT_DIR = ROOT_DIR / "plots"
SPLITS_EXIST = (ROOT_DIR / "data" / "splits" / "train.parquet").exists()


# ----------------------------------------------------------------------------
# 1. Feature family partition helper (42 features from updated Part B)
# ----------------------------------------------------------------------------
@pytest.mark.skipif(not SPLITS_EXIST, reason="splits not generated")
def test_feature_family_partition_covers_all_42_features():
    data = make_datasets(FeatureConfig(horizon=24))
    cols = feature_columns(data["train"])
    assert len(cols) == 42
    counts = pd.Series([_feature_family(c) for c in cols]).value_counts().to_dict()
    assert counts == {
        "Calendar (18)": 18,
        "Lag / Rolling (13)": 13,
        "Weather (11)": 11,
    }


# ----------------------------------------------------------------------------
# 2. Regression test: XGBoost Pseudo-Huber numerical stability on MW scale
# ----------------------------------------------------------------------------
def test_xgb_pseudohuber_numerical_stability():
    """Ensure base_score=median(y) & max_delta_step prevent Newton-step blowup."""
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = 28_000.0 + 1_200.0 * X["f0"] - 800.0 * X["f1"] + rng.normal(0, 300, size=n)
    X_tr, y_tr = X.iloc[:300], y.iloc[:300]
    X_va, y_va = X.iloc[300:], y.iloc[300:]

    model = xgb.XGBRegressor(
        objective="reg:pseudohubererror",
        huber_slope=1000.0,
        base_score=float(np.median(y_tr)),
        max_delta_step=2000.0,
        n_estimators=80,
        learning_rate=0.05,
        max_depth=5,
        tree_method="hist",
        random_state=0,
        n_jobs=1,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    m = evaluate(y_va, pred)
    assert np.isfinite(m["MAE"])
    assert m["MAE"] < 1_000.0
    assert m["R2"] > 0.70


# ----------------------------------------------------------------------------
# 3. ResultsLog schema compatibility for Part D outputs
# ----------------------------------------------------------------------------
def test_metrics_schema_compatible_with_evaluate():
    required_cols = {"model", "split", "MAE", "RMSE", "MAPE", "sMAPE", "R2", "MaxAE", "ME"}

    df_boost = pd.read_csv(RESULTS_DIR / "metrics_boosting.csv")
    assert required_cols <= set(df_boost.columns)
    assert not df_boost[list(required_cols - {"model", "split"})].isna().any().any()
    assert {"LightGBM", "XGBoost"} <= set(df_boost["model"])

    df_all = pd.read_csv(RESULTS_DIR / "metrics_all.csv")
    assert required_cols <= set(df_all.columns)
    assert not df_all[list(required_cols - {"model", "split"})].isna().any().any()
    expected_models = {
        "Naive (t-24h)",
        "Seasonal naive (t-168h)",
        "Mean of last 4 weeks",
        "TSO forecast (reference)",
        "Ridge",
        "Lasso",
        "Random Forest",
        "LightGBM",
        "XGBoost",
    }
    assert expected_models <= set(df_all.query("split == 'test'")["model"])


# ----------------------------------------------------------------------------
# 4. Tuned LightGBM & XGBoost outperform Part C Random Forest on Test
# ----------------------------------------------------------------------------
def test_tuned_boosting_beats_random_forest_on_test():
    df_all = pd.read_csv(RESULTS_DIR / "metrics_all.csv").query("split == 'test'").set_index("model")
    rf = df_all.loc["Random Forest"]
    for model_name in ["LightGBM", "XGBoost"]:
        row = df_all.loc[model_name]
        assert row["MAE"] < rf["MAE"], f"{model_name} MAE should beat RF"
        assert row["RMSE"] < rf["RMSE"], f"{model_name} RMSE should beat RF"
        assert row["MAPE"] < rf["MAPE"], f"{model_name} MAPE should beat RF"
        assert row["R2"] > rf["R2"], f"{model_name} R2 should beat RF"


# ----------------------------------------------------------------------------
# 5. Merged Test Predictions Parquet alignment for Person E
# ----------------------------------------------------------------------------
def test_predictions_parquet_aligned_for_person_e():
    pred_c = pd.read_parquet(RESULTS_DIR / "predictions_test.parquet")
    pred_all = pd.read_parquet(RESULTS_DIR / "predictions_test_all.parquet")

    assert len(pred_all) == 5260
    assert len(pred_all) == len(pred_c)
    assert (pred_all["time"].to_numpy() == pred_c["time"].to_numpy()).all()
    assert np.allclose(pred_all["actual"].to_numpy(), pred_c["actual"].to_numpy(), atol=1e-6)
    assert {"LightGBM", "XGBoost", "Random Forest"} <= set(pred_all.columns)
    assert not pred_all[["actual", "LightGBM", "XGBoost"]].isna().any().any()

    # Check ResultsLog roundtrip from saved parquet matches metrics_all.csv
    df_all = pd.read_csv(RESULTS_DIR / "metrics_all.csv").query("split == 'test'").set_index("model")
    log = ResultsLog()
    log.add("LightGBM", "test", pred_all["actual"], pred_all["LightGBM"])
    log.add("XGBoost", "test", pred_all["actual"], pred_all["XGBoost"])
    rt = log.to_frame().set_index("model")
    for m in ("LightGBM", "XGBoost"):
        assert rt.loc[m, "MAE"] == pytest.approx(df_all.loc[m, "MAE"], abs=1e-2)


# ----------------------------------------------------------------------------
# 6. Exact TreeSHAP local additivity axiom & global share sum
# ----------------------------------------------------------------------------
def test_treeshap_local_additivity_axiom():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 6)), columns=[f"x{i}" for i in range(6)])
    y = 28_000.0 + 1_000.0 * X["x0"] - 600.0 * X["x1"] + 300.0 * X["x2"] ** 2
    model = lgb.LGBMRegressor(
        objective="huber",
        alpha=1000.0,
        n_estimators=50,
        learning_rate=0.08,
        num_leaves=15,
        random_state=0,
        n_jobs=1,
        verbose=-1,
    ).fit(X.iloc[:240], y.iloc[:240])

    X_eval = X.iloc[240:]
    preds = model.predict(X_eval)
    exp = shap.TreeExplainer(model)(X_eval)
    reconstructed = exp.base_values + exp.values.sum(axis=1)
    assert float(np.max(np.abs(preds - reconstructed))) < 1e-6

    # Also verify saved Step 4 SHAP table sums cleanly to 100% across 42 features
    df_shap = pd.read_csv(RESULTS_DIR / "d_step4_shap_importance.csv")
    assert len(df_shap) == 42
    assert df_shap["lgbm_share_pct"].sum() == pytest.approx(100.0, abs=0.1)
    assert df_shap["lgbm_val_share_pct"].sum() == pytest.approx(100.0, abs=0.1)
    assert df_shap["xgb_share_pct"].sum() == pytest.approx(100.0, abs=0.1)

    df_grp = pd.read_csv(RESULTS_DIR / "d_step4_shap_by_group.csv")
    assert len(df_grp) == 3
    assert df_grp["n_features"].sum() == 42
    assert df_grp["lgbm_net_group_share_pct"].sum() == pytest.approx(100.0, abs=0.1)


# ----------------------------------------------------------------------------
# 7. Step 4 Local Waterfall Cases consistency
# ----------------------------------------------------------------------------
def test_step4_local_cases_consistency():
    df_cases = pd.read_csv(RESULTS_DIR / "d_step4_local_cases.csv")
    assert {"peak", "holiday", "worst_error"} <= set(df_cases["case"])
    diff = df_cases["actual_mw"] - df_cases["pred_mw"]
    assert np.allclose(diff.to_numpy(), df_cases["residual_y_minus_yhat_mw"].to_numpy(), atol=1e-4)
    assert (df_cases["base_value_mw"] > 20_000.0).all()


# ----------------------------------------------------------------------------
# 8. All Part D artifacts (15 data files, 14 PNG plots, notebook) exist
# ----------------------------------------------------------------------------
def test_all_part_d_outputs_exist_and_nonempty():
    expected_results = [
        "d_step1_untuned.csv",
        "d_step2_loss_comparison.csv",
        "d_step2_tuning_lgbm.csv",
        "d_step2_tuning_xgb.csv",
        "d_step2_best_params.json",
        "metrics_boosting.csv",
        "metrics_all.csv",
        "predictions_test_all.parquet",
        "mae_by_hour_all.csv",
        "mae_by_dow_all.csv",
        "d_step3_multiseed.csv",
        "horizon_comparison_all.csv",
        "d_step4_shap_importance.csv",
        "d_step4_shap_by_group.csv",
        "d_step4_local_cases.csv",
    ]
    for fname in expected_results:
        p = RESULTS_DIR / fname
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty result: {fname}"

    expected_plots = [
        "d_step1_learning_curve.png",
        "d_step2_tuning_curves.png",
        "d_step3_mae_test_all.png",
        "d_step3_mape_test_all.png",
        "d_step3_forecast_2weeks.png",
        "d_step3_forecast_holiday.png",
        "d_step3_mae_by_hour_all.png",
        "d_step3_residuals.png",
        "d_step4_shap_beeswarm.png",
        "d_step4_shap_vs_rf.png",
        "d_step4_shap_dependence.png",
        "d_step4_waterfall_peak.png",
        "d_step4_waterfall_holiday.png",
        "d_step4_waterfall_worst_error.png",
    ]
    for fname in expected_plots:
        p = PLOT_DIR / fname
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty plot: {fname}"

    nb_path = ROOT_DIR / "src" / "boosting_shap.ipynb"
    assert nb_path.exists() and nb_path.stat().st_size > 0
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 6
    assert [c["execution_count"] for c in code_cells] == list(range(1, len(code_cells) + 1))
