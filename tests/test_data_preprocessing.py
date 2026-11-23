import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preprocessing import fix_day_month_swap  # noqa: E402


def test_splits_exist_and_chronological():
    splits_dir = Path("data/splits")
    train = pd.read_parquet(splits_dir / "train.parquet")
    val = pd.read_parquet(splits_dir / "val.parquet")
    test = pd.read_parquet(splits_dir / "test.parquet")

    # 1. Row counts sum to 35,064 (no rows duplicated or dropped during left join)
    assert len(train) + len(val) + len(test) == 35064

    # 2. Strict chronological ordering
    assert train["time"].max() < val["time"].min()
    assert val["time"].max() < test["time"].min()

    # 3. No NaNs in key target or weather features
    for split in (train, val, test):
        assert not split["total load actual"].isna().any()
        assert not split["temp_mean"].isna().any()


def test_leakage_free_float32_standardization():
    splits_dir = Path("data/splits")
    feature_cols = ["temp_mean", "pressure_mean", "humidity_mean", "wind_speed_mean"]

    # Parquet projection pushdown
    X_train = np.ascontiguousarray(
        pd.read_parquet(splits_dir / "train.parquet", columns=feature_cols).to_numpy(),
        dtype=np.float32,
    )
    X_val = np.ascontiguousarray(
        pd.read_parquet(splits_dir / "val.parquet", columns=feature_cols).to_numpy(),
        dtype=np.float32,
    )

    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + np.float32(1e-8)

    X_train_std = (X_train - mu) / sd
    X_val_std = (X_val - mu) / sd

    assert X_train_std.dtype == np.float32
    assert X_val_std.dtype == np.float32
    assert not np.isnan(X_train_std).any()
    assert np.allclose(X_train_std.mean(axis=0), 0.0, atol=1e-3)


def test_fix_day_month_swap():
    df = pd.DataFrame({
        "time": ["2016-02-06 10:00:00+01:00", "2016-06-02 10:00:00+02:00", "2016-02-15 10:00:00+01:00"],
        "total load actual": [1.0, 2.0, 3.0],
        "price actual": [10.0, 20.0, 30.0],
    })
    out = fix_day_month_swap(df)
    assert out["total load actual"].tolist() == [2.0, 1.0, 3.0]  # day 15 cannot be swapped
    assert out["price actual"].tolist() == [10.0, 20.0, 30.0]
