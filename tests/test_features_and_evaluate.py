"""Tests for feature engineering (Part B) and the shared evaluation module (Part C)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import ResultsLog, evaluate, mae, mape, r2, rmse, smape  # noqa: E402
from src.features import (  # noqa: E402
    TARGET,
    TSO_FORECAST,
    FeatureConfig,
    add_calendar_features,
    build_feature_frame,
    feature_columns,
    make_datasets,
)


# ----------------------------------------------------------------------------
# evaluate.py
# ----------------------------------------------------------------------------
def test_metrics_known_values():
    y = np.array([100.0, 200.0, 300.0, 400.0])
    p = np.array([110.0, 190.0, 330.0, 400.0])
    assert mae(y, p) == pytest.approx(12.5)
    assert rmse(y, p) == pytest.approx(np.sqrt((100 + 100 + 900) / 4))
    assert mape(y, p) == pytest.approx((10 + 5 + 10 + 0) / 4)
    assert smape(y, y) == 0.0
    assert r2(y, y) == 1.0
    assert r2(y, np.full(4, y.mean())) == pytest.approx(0.0)


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        mae([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        mae([1, np.nan], [1, 2])
    with pytest.raises(ValueError):
        mape([0, 1], [1, 1])


def test_results_log_roundtrip():
    log = ResultsLog()
    log.add("m1", "val", [1, 2, 3], [1, 2, 4])
    log.add("m1", "test", [1, 2, 3], [1, 2, 3])
    df = log.to_frame()
    assert set(df.columns) >= {"model", "split", "MAE", "RMSE", "MAPE", "sMAPE", "R2", "MaxAE"}
    assert log.pivot("MAE").loc["m1", "test"] == 0.0
    assert evaluate([1, 2], [1, 2])["MAE"] == 0.0


# ----------------------------------------------------------------------------
# features.py on a synthetic series (fast, no data files needed)
# ----------------------------------------------------------------------------
def _synthetic(n_hours: int = 24 * 60, start: str = "2016-01-01") -> pd.DataFrame:
    t = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    y = np.arange(n_hours, dtype=float) + 20_000
    return pd.DataFrame({
        "time": t, TARGET: y, TSO_FORECAST: y,
        "temp_mean": 288.15, "temp_min": 285.0, "temp_max": 291.0,
        "humidity_mean": 60.0, "pressure_mean": 1015.0, "wind_speed_mean": 3.0, "clouds_all_mean": 20.0,
    })


@pytest.mark.parametrize("horizon", [1, 24])
def test_lags_never_look_ahead(horizon):
    df = _synthetic()
    cfg = FeatureConfig(horizon=horizon)
    f = build_feature_frame(df, cfg)
    i = 24 * 50  # a row with complete history; y increases by 1 every hour
    y_i = f.loc[i, TARGET]
    for lag in cfg.effective_lags:
        assert f.loc[i, f"lag_{lag}"] == y_i - lag
    assert min(cfg.effective_lags) >= horizon
    # rolling mean of the window ending at t-horizon
    assert f.loc[i, "roll_mean_24"] == pytest.approx(y_i - horizon - 11.5)
    # no lag column may equal the current target
    lag_cols = [c for c in f.columns if c.startswith(("lag_", "roll_", "diff_"))]
    assert not (f.loc[i, lag_cols] >= y_i).any()


def test_target_and_benchmark_not_in_features():
    f = build_feature_frame(_synthetic(), FeatureConfig())
    cols = feature_columns(f)
    assert TARGET not in cols and TSO_FORECAST not in cols and "time" not in cols


def test_calendar_uses_local_time_and_holidays():
    t = pd.Series(pd.to_datetime(["2018-01-01 08:00", "2018-07-15 22:30", "2018-10-12 10:00"], utc=True))
    cal = add_calendar_features(pd.DataFrame({"time": t}))
    # 08:00 UTC in winter = 09:00 CET ; 22:30 UTC in summer = 00:30 CEST next day
    assert cal["cal_hour"].tolist() == [9, 0, 12]
    assert cal["cal_is_dst"].tolist() == [0, 1, 1]
    assert cal["cal_is_holiday"].tolist() == [1, 0, 1]  # New Year, -, Fiesta Nacional
    assert cal.loc[1, "cal_dayofweek"] == 0  # 2018-07-16 is a Monday


# ----------------------------------------------------------------------------
# features.py on the real splits
# ----------------------------------------------------------------------------
@pytest.mark.skipif(not (Path(__file__).resolve().parent.parent / "data/splits/train.parquet").exists(),
                    reason="splits not generated")
def test_make_datasets_real_splits():
    data = make_datasets(FeatureConfig(horizon=24))
    tr, va, te = data["train"], data["val"], data["test"]
    assert tr["time"].max() < va["time"].min() < te["time"].min()
    assert len(va) == 5260 and len(te) == 5260
    cols = feature_columns(tr)
    for part in (tr, va, te):
        assert not part[cols].isna().any().any()
    assert not any(c.startswith("lag_") and int(c.split("_")[1]) < 24 for c in cols if c[4:].isdigit())


def test_special_days_and_bridge():
    t = pd.Series(pd.to_datetime(["2018-03-29 10:00", "2018-04-02 10:00", "2018-12-23 23:30",
                                  "2018-12-31 10:00", "2017-01-02 10:00", "2018-05-02 10:00"], utc=True))
    cal = add_calendar_features(pd.DataFrame({"time": t}))
    # Holy Thursday, Easter Monday, 24/12 in Madrid, 31/12, Monday after Sunday 1/1, normal day
    assert cal["cal_is_special"].tolist() == [1, 1, 1, 1, 1, 0]
    assert cal["cal_is_holiday"].sum() == 0
    assert cal.loc[3, "cal_is_bridge"] == 1  # Mon 31/12/2018, between Sunday and 1/1/2019


def test_good_friday_and_lag_day_flags():
    t = pd.Series(pd.to_datetime(["2016-03-25 10:00", "2018-05-02 10:00", "2018-05-08 10:00"], utc=True))
    cal = add_calendar_features(pd.DataFrame({"time": t}))
    assert cal.loc[0, "cal_is_holiday"] == 1
    assert cal.loc[1, "cal_lag1d_is_off"] == 1  # day after 1 May
    assert cal.loc[2, "cal_lag7d_is_off"] == 1  # one week after 1 May


def test_weekly_lags_follow_local_clock_across_dst():
    df = _synthetic(24 * 41, start="2018-03-10")
    df[TARGET] = df["time"].dt.tz_convert("Europe/Madrid").dt.hour * 1000.0
    f = build_feature_frame(df, FeatureConfig(horizon=24))
    week_after_change = f["time"].between(pd.Timestamp("2018-03-26", tz="UTC"), pd.Timestamp("2018-03-31 23:00", tz="UTC"))
    assert (f.loc[week_after_change, "lag_168"] == f.loc[week_after_change, TARGET]).all()


def test_weekly_lag_never_inside_horizon():
    df = _synthetic(24 * 45, start="2018-03-01")
    p = df.index[df["time"] == pd.Timestamp("2018-03-23 19:00", tz="UTC")][0]  # one week before the DST change
    changed = df.copy()
    changed.loc[p:, TARGET] += 5000.0
    cfg = FeatureConfig(horizon=168)
    base, f = build_feature_frame(df, cfg), build_feature_frame(changed, cfg)
    cols = [c for c in feature_columns(base) if c.startswith(("lag_", "roll_", "diff_"))]
    pd.testing.assert_frame_equal(base.loc[: p + 167, cols], f.loc[: p + 167, cols])
