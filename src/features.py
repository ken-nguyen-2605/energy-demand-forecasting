"""Feature engineering for hourly energy-demand forecasting (Part B).

Builds three feature families on top of the chronological splits produced by
``data_preprocessing.py``:

1. Calendar features  - local (Europe/Madrid) hour, weekday, month, day of year,
   weekend / Spanish national holiday flags and cyclical sin/cos encodings.
2. Lag features       - past values of the target (``total load actual``) and
   rolling statistics, all shifted by at least the forecast ``horizon`` so that
   every feature is known at the moment the forecast is issued.
3. Weather features   - temperature in Celsius, heating / cooling degree hours,
   squared temperature (U-shaped load response), temperature range, humidity,
   wind, clouds and smoothed temperature.

Leakage rules
-------------
* Lags/rolling windows are computed on the concatenated train -> val -> test
  series (strictly chronological) and only look backwards; the first rows of
  train whose window is incomplete are dropped. Values from the end of train
  feeding the first lags of val are *past* observations, which is legitimate.
* Columns that are not available at forecast time or that leak the target
  (``total load forecast``, ``price actual``, generation columns, ...) are never
  used as model inputs. ``total load forecast`` is kept only as a benchmark.
* Weather at the target hour is treated as known (perfect weather forecast
  assumption, standard in load-forecasting studies; see the report).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT_DIR / "data" / "splits"

TARGET = "total load actual"
TSO_FORECAST = "total load forecast"
LOCAL_TZ = "Europe/Madrid"

# Spanish *national* public holidays 2014-2018 (regional holidays are ignored).
# Good Friday: 2015-04-03, 2016-03-25, 2017-04-14, 2018-03-30.
_FIXED_HOLIDAYS = ["01-01", "01-06", "05-01", "08-15", "10-12", "11-01", "12-06", "12-08", "12-25"]
_GOOD_FRIDAYS = ["2015-04-03", "2016-03-25", "2017-04-14", "2018-03-30"]
SPAIN_HOLIDAYS = pd.DatetimeIndex(
    sorted(
        [f"{y}-{md}" for y in range(2014, 2019) for md in _FIXED_HOLIDAYS] + _GOOD_FRIDAYS
    )
)


@dataclass
class FeatureConfig:
    """Hyper-parameters of the feature pipeline."""

    horizon: int = 24  # forecast lead time in hours (24 = day-ahead)
    lags: tuple[int, ...] = (1, 2, 3, 24, 48, 72, 168, 336)
    roll_windows: tuple[int, ...] = (24, 168)
    comfort_temp_c: float = 18.0  # base temperature for heating/cooling degree hours
    use_calendar: bool = True
    use_lags: bool = True
    use_weather: bool = True
    extra_drop: list[str] = field(default_factory=list)

    @property
    def effective_lags(self) -> list[int]:
        """Only lags >= horizon are observable at forecast time."""
        return sorted({lag for lag in self.lags if lag >= self.horizon} | {self.horizon})


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_splits(split_dir: Path = SPLIT_DIR) -> dict[str, pd.DataFrame]:
    """Read the chronological splits written by ``data_preprocessing.py``."""
    return {name: pd.read_parquet(split_dir / f"{name}.parquet") for name in ("train", "val", "test")}


# ----------------------------------------------------------------------------
# Feature families
# ----------------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features in *local* Spanish time (consumption follows local clocks)."""
    local = df["time"].dt.tz_convert(LOCAL_TZ)
    local_date = local.dt.tz_localize(None).dt.normalize()
    hour = local.dt.hour
    dow = local.dt.day_of_week
    month = local.dt.month
    doy = local.dt.day_of_year
    is_holiday = local_date.isin(SPAIN_HOLIDAYS)

    out = pd.DataFrame(index=df.index)
    out["cal_hour"] = hour
    out["cal_dayofweek"] = dow
    out["cal_month"] = month
    out["cal_dayofyear"] = doy
    out["cal_is_weekend"] = (dow >= 5).astype(int)
    out["cal_is_holiday"] = is_holiday.astype(int)
    out["cal_is_offday"] = ((dow >= 5) | is_holiday).astype(int)
    # Bridge days ("puentes"): a working day squeezed between an off-day and a holiday.
    prev_off = (local_date - pd.Timedelta(days=1)).isin(SPAIN_HOLIDAYS) | ((dow - 1) % 7 >= 5)
    next_off = (local_date + pd.Timedelta(days=1)).isin(SPAIN_HOLIDAYS) | ((dow + 1) % 7 >= 5)
    out["cal_is_bridge"] = ((~(dow >= 5)) & (~is_holiday) & prev_off & next_off).astype(int)
    utc_offset = local.dt.tz_localize(None) - df["time"].dt.tz_localize(None)
    out["cal_is_dst"] = (utc_offset == pd.Timedelta(hours=2)).astype(int)  # CEST = UTC+2
    # Cyclical encodings so that 23h is close to 0h, December close to January.
    out["cal_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["cal_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["cal_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["cal_dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["cal_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["cal_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def add_lag_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Target lags and rolling statistics, all shifted by >= horizon hours."""
    y = df[TARGET]
    out = pd.DataFrame(index=df.index)
    for lag in cfg.effective_lags:
        out[f"lag_{lag}"] = y.shift(lag)
    base = y.shift(cfg.horizon)  # last value observable when forecasting
    for w in cfg.roll_windows:
        out[f"roll_mean_{w}"] = base.rolling(w, min_periods=w).mean()
        out[f"roll_std_{w}"] = base.rolling(w, min_periods=w).std()
    out["roll_min_24"] = base.rolling(24, min_periods=24).min()
    out["roll_max_24"] = base.rolling(24, min_periods=24).max()
    # Same hour, average over the last 4 weeks that are observable.
    weekly_lags = [k * 168 for k in range(1, 5) if k * 168 >= cfg.horizon]
    out["lag_same_hour_4w_mean"] = pd.concat([y.shift(k) for k in weekly_lags], axis=1).mean(axis=1, skipna=False)
    # Day-over-day difference of the last observable day (trend proxy).
    out["diff_24"] = base - y.shift(cfg.horizon + 24)
    return out


def add_weather_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Domain weather features derived from the 5-city national averages."""
    t_c = df["temp_mean"] - 273.15
    out = pd.DataFrame(index=df.index)
    out["w_temp_c"] = t_c
    out["w_temp_c_sq"] = (t_c - cfg.comfort_temp_c) ** 2
    out["w_hdh"] = (cfg.comfort_temp_c - t_c).clip(lower=0)  # heating degree hours
    out["w_cdh"] = (t_c - cfg.comfort_temp_c - 4).clip(lower=0)  # cooling degree hours (base 22 C)
    out["w_temp_range"] = df["temp_max"] - df["temp_min"]
    out["w_humidity"] = df["humidity_mean"]
    out["w_pressure"] = df["pressure_mean"]
    out["w_wind_speed"] = df["wind_speed_mean"]
    out["w_clouds"] = df["clouds_all_mean"]
    # Thermal inertia of buildings: smoothed temperature over the past day.
    out["w_temp_roll24"] = t_c.rolling(24, min_periods=1).mean()
    out["w_temp_roll24_diff"] = t_c - out["w_temp_roll24"]
    return out


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def build_feature_frame(full: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Return ``time``, target, TSO benchmark and all engineered features."""
    full = full.sort_values("time").reset_index(drop=True)
    assert full["time"].is_unique, "Duplicate timestamps in concatenated splits"
    parts = [full[["time", TARGET, TSO_FORECAST]]]
    if cfg.use_calendar:
        parts.append(add_calendar_features(full))
    if cfg.use_lags:
        parts.append(add_lag_features(full, cfg))
    if cfg.use_weather:
        parts.append(add_weather_features(full, cfg))
    feats = pd.concat(parts, axis=1)
    return feats.drop(columns=[c for c in cfg.extra_drop if c in feats.columns])


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """All model-input columns (everything except time, target and benchmark)."""
    return [c for c in frame.columns if c not in ("time", TARGET, TSO_FORECAST)]


def make_datasets(cfg: FeatureConfig | None = None, split_dir: Path = SPLIT_DIR) -> dict[str, pd.DataFrame]:
    """Build features on the full timeline and cut them back into train/val/test.

    Rows whose lag window is incomplete (the first 4 weeks = 672 h of train) are dropped.
    """
    cfg = cfg or FeatureConfig()
    splits = load_splits(split_dir)
    bounds = {name: (df["time"].min(), df["time"].max()) for name, df in splits.items()}
    full = pd.concat(splits.values(), ignore_index=True)
    frame = build_feature_frame(full, cfg)

    out: dict[str, pd.DataFrame] = {}
    for name, (t0, t1) in bounds.items():
        part = frame[(frame["time"] >= t0) & (frame["time"] <= t1)]
        part = part.dropna(subset=feature_columns(frame)).reset_index(drop=True)
        out[name] = part
    assert out["train"]["time"].max() < out["val"]["time"].min() < out["test"]["time"].min()
    assert len(out["val"]) == len(splits["val"]) and len(out["test"]) == len(splits["test"]), (
        "Val/test rows must not be dropped by lag warm-up"
    )
    return out


def xy(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature frame into (X, y)."""
    return df[cols], df[TARGET]


if __name__ == "__main__":
    cfg = FeatureConfig()
    data = make_datasets(cfg)
    cols = feature_columns(data["train"])
    print(f"Horizon = {cfg.horizon}h, effective lags = {cfg.effective_lags}")
    print(f"{len(cols)} features: {cols}")
    for name, df in data.items():
        print(f"{name:5s}: {df.shape} ({df['time'].min()} -> {df['time'].max()})")
