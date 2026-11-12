"""Energy & Weather Data Preprocessing, EDA, and Chronological Split Pipeline.

Implements data cleaning, 4-panel Seaborn EDA figure generation, chronological
Train / Validation / Test splitting, Parquet columnar storage, and leakage-free
float32 standardization.
"""

from dataclasses import dataclass
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend for headless script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class PipelineConfig:
    """Configuration for paths and chronological split ratios."""

    data_dir: Path = ROOT_DIR / "data"
    out_dir: Path = ROOT_DIR / "data/splits"
    plot_dir: Path = ROOT_DIR / "plots"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 0


def timed(fn):
    """Decorator to measure execution time of pipeline functions."""

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        print(f"[TIMED] {fn.__name__}: {time.perf_counter() - t0:.4f}s")
        return out

    return wrapper


@timed
def clean_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Clean energy dataset using method chaining and integrity assertions."""
    before_rows = len(df)
    drop_cols = [
        # 100% NaN columns (35,064 missing)
        "generation hydro pumped storage aggregated",
        "forecast wind offshore eday ahead",
        # All-zero columns
        "generation fossil coal-derived gas",
        "generation fossil oil shale",
        "generation fossil peat",
        "generation geothermal",
        "generation marine",
        "generation wind offshore",
    ]
    out = (
        df.drop(columns=drop_cols)
        .assign(time=lambda d: pd.to_datetime(d["time"], utc=True))
        .sort_values("time")
        .reset_index(drop=True)
        .ffill()
        .bfill()
        .assign(
            hour=lambda d: d["time"].dt.hour,
            month=lambda d: d["time"].dt.month,
            dayofweek=lambda d: d["time"].dt.day_of_week,
        )
    )
    assert len(out) == before_rows, f"Row count changed: {before_rows} -> {len(out)}"
    assert out["time"].is_unique, "Energy timestamps must be unique!"
    assert out.isna().sum().sum() == 0, "NaNs remain in cleaned energy table!"
    return out


@timed
def clean_and_aggregate_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Clean 5-city weather data, compute city temperature deviation, and aggregate hourly."""
    w = (
        df.assign(
            time=lambda d: pd.to_datetime(d["dt_iso"], utc=True),
            city_name=lambda d: d["city_name"].str.strip().astype("category"),
        )
        .drop_duplicates(subset=["time", "city_name"], keep="first")
        .sort_values(["city_name", "time"])
        .reset_index(drop=True)
    )
    # 5 cities x 35,064 hours = 175,320 rows (3,076 duplicate keys removed)
    assert len(w) == 35064 * 5, f"Expected 175320 rows, got {len(w)}"

    # Mask impossible sensor readings using parenthesized boolean masks and .loc[mask, col]
    bad_pressure = (w["pressure"] < 900) | (w["pressure"] > 1080)
    bad_wind = w["wind_speed"] > 50
    print(f"Weather outliers masked -> pressure: {bad_pressure.sum()}, wind_speed: {bad_wind.sum()}")
    w.loc[bad_pressure, "pressure"] = np.nan
    w.loc[bad_wind, "wind_speed"] = np.nan

    # Forward-fill within each city
    w[["pressure", "wind_speed"]] = (
        w.groupby("city_name", observed=True)[["pressure", "wind_speed"]].ffill().bfill()
    )

    # Compute per-row city temperature deviation via transform
    w["temp_dev_from_city_mean"] = (
        w["temp"] - w.groupby("city_name", observed=True)["temp"].transform("mean")
    )

    # Collapse 5 cities into 1 national hourly lookup row via agg
    weather_hourly = w.groupby("time", as_index=False).agg(
        temp_mean=("temp", "mean"),
        temp_min=("temp_min", "min"),
        temp_max=("temp_max", "max"),
        temp_dev_mean=("temp_dev_from_city_mean", "mean"),
        pressure_mean=("pressure", "mean"),
        humidity_mean=("humidity", "mean"),
        wind_speed_mean=("wind_speed", "mean"),
        clouds_all_mean=("clouds_all", "mean"),
    )
    return weather_hourly


@timed
def plot_eda_four_panel(df: pd.DataFrame, out_path: Path) -> None:
    """Generate and save the 4-panel Seaborn EDA figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Distribution (histplot)
    sns.histplot(data=df, x="total load actual", kde=True, ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("1. Distribution: Total Load Actual (MW)")

    # 2. Relationship (scatterplot)
    sample_df = df.iloc[::10]
    sns.scatterplot(
        data=sample_df,
        x="temp_mean",
        y="total load actual",
        hue="hour",
        palette="viridis",
        alpha=0.6,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("2. Relationship: Mean Temp (K) vs Total Load")

    # 3. Category Comparison (boxplot)
    sns.boxplot(
        data=df,
        x="dayofweek",
        y="price actual",
        hue="dayofweek",
        legend=False,
        ax=axes[1, 0],
        palette="Set2",
    )
    axes[1, 0].set_title("3. Category Comparison: Price Actual by Day of Week (0=Mon)")

    # 4. Correlation Matrix (heatmap)
    corr_cols = [
        "total load actual",
        "total load forecast",
        "price actual",
        "price day ahead",
        "generation solar",
        "generation wind onshore",
        "generation fossil gas",
        "temp_mean",
        "humidity_mean",
        "wind_speed_mean",
    ]
    sns.heatmap(
        df[corr_cols].corr(),
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("4. Correlation Matrix (Spot Forecast vs Actual!)")

    fig.suptitle("Energy & Weather: 4-Panel EDA Diagnostic Figure", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved 4-panel EDA figure to {out_path}")


def main() -> None:
    cfg = PipelineConfig()
    _rng = np.random.default_rng(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.plot_dir.mkdir(parents=True, exist_ok=True)

    energy_raw = pd.read_csv(cfg.data_dir / "energy_dataset.csv")
    weather_raw = pd.read_csv(cfg.data_dir / "weather_features.csv")
    print(f"Raw shapes -> energy: {energy_raw.shape}, weather: {weather_raw.shape}")

    energy = clean_energy(energy_raw)
    weather_hourly = clean_and_aggregate_weather(weather_raw)

    # Validate 1-to-1 left join contract
    assert weather_hourly["time"].is_unique, "Lookup key time is not unique!"
    before = len(energy)
    merged = energy.merge(weather_hourly, on="time", how="left", validate="one_to_one")
    assert len(merged) == before, f"Merge changed row count: {before} -> {len(merged)}"
    print(f"Unmatched rows after left join: {merged['temp_mean'].isna().sum()}")
    print(f"Merged shape: {merged.shape}")

    # Generate 4-panel Seaborn EDA figure
    plot_eda_four_panel(merged, cfg.plot_dir / "eda_4panel.png")

    # Chronological Train / Val / Test split (70% / 15% / 15%)
    n = len(merged)
    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))

    train_df = merged.iloc[:train_end]
    val_df = merged.iloc[train_end:val_end]
    test_df = merged.iloc[val_end:]

    assert len(train_df) + len(val_df) + len(test_df) == n, "Split sizes do not sum to total!"
    assert train_df["time"].max() < val_df["time"].min() < test_df["time"].min(), "Temporal leakage!"

    # Save splits in both Parquet and CSV formats
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df.to_parquet(cfg.out_dir / f"{name}.parquet", index=False)
        split_df.to_csv(cfg.out_dir / f"{name}.csv", index=False)
        print(f"Saved {name}: {split_df.shape} ({split_df['time'].min()} -> {split_df['time'].max()})")

    # Read narrow feature subset from Parquet, standardize using train statistics, and cast to float32
    feature_cols = [
        "temp_mean",
        "pressure_mean",
        "humidity_mean",
        "wind_speed_mean",
        "generation solar",
        "generation wind onshore",
    ]
    X_train_raw = pd.read_parquet(cfg.out_dir / "train.parquet", columns=feature_cols).to_numpy()
    X_val_raw = pd.read_parquet(cfg.out_dir / "val.parquet", columns=feature_cols).to_numpy()
    X_test_raw = pd.read_parquet(cfg.out_dir / "test.parquet", columns=feature_cols).to_numpy()

    mu = X_train_raw.mean(axis=0)
    sd = X_train_raw.std(axis=0)

    X_train = np.ascontiguousarray((X_train_raw - mu) / sd, dtype=np.float32)
    X_val = np.ascontiguousarray((X_val_raw - mu) / sd, dtype=np.float32)
    X_test = np.ascontiguousarray((X_test_raw - mu) / sd, dtype=np.float32)

    assert X_train.dtype == np.float32
    assert not np.isnan(X_train).any() and not np.isnan(X_val).any() and not np.isnan(X_test).any()
    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(X_train.std(axis=0), 1.0, atol=1e-5)
    print(f"Verified float32 standardized arrays -> X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")


if __name__ == "__main__":
    main()
