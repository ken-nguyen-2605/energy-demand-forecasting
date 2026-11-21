# Energy Consumption, Generation, Prices & Weather Forecasting

End-to-end data preprocessing, Exploratory Data Analysis (EDA), feature engineering, and machine learning pipeline for the [Hourly Energy Demand Generation and Weather (Spain)](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) dataset.

---

## 1. Project Workflow (4 Stages)

1. **Data Preprocessing & EDA (`src/data_preprocessing.py`, `src/eda.ipynb`):**
   - Download raw datasets, clean missing values and sensor outliers, merge national energy and 5-city weather data on UTC timestamps, generate the 4-panel EDA figure (`plots/eda_4panel.png`), and create chronological **Train / Validation / Test** splits (`data/splits/`).
2. **Feature Engineering (`src/features.py`):**
   - Calendar features in local Spanish time (hour, weekday, month, holidays, bridge days, DST, sin/cos encodings), target lag/rolling features shifted by the forecast horizon (default **24 h = day-ahead**), and domain weather features (°C, heating/cooling degree hours, temperature², thermal inertia).
3. **Baseline & Linear / Tree Models (`src/evaluate.py`, `src/train_models.py`):**
   - Shared evaluation module (MAE, RMSE, MAPE, sMAPE, R², MaxAE + diagnostic plots) and training of Baselines (naive t-24h, seasonal naive t-168h, 4-week mean), Ridge/Lasso and Random Forest with validation-based tuning, feature-family ablation and 1h-vs-24h horizon comparison.
4. **Gradient Boosting, Hyperparameter Tuning & SHAP:**
   - Train and tune **LightGBM** and **XGBoost** models, run **SHAP** feature attribution analysis, and verify end-to-end reproducibility.

---

## 2. Repository Structure

```text
.
├── data/
│   ├── energy_dataset.csv               # Raw national hourly energy data (35,064 rows x 29 cols)
│   ├── weather_features.csv             # Raw 5-city hourly weather data (178,396 rows x 17 cols)
│   └── splits/                          # Cleaned, merged & chronologically split datasets
│       ├── train.parquet / .csv         # First 70%: 24,544 rows (2014-12-31 -> 2017-10-19)
│       ├── val.parquet   / .csv         # Next 15%:   5,260 rows (2017-10-19 -> 2018-05-26)
│       └── test.parquet  / .csv         # Final 15%:  5,260 rows (2018-05-26 -> 2018-12-31)
├── docs/
│   ├── en/                              # English documentation (1.DATA_PREPROCESSING.md, ...)
│   └── vi/                              # Vietnamese documentation (1.DATA_PREPROCESSING.md, ...)
├── plots/
│   └── eda_4panel.png                   # 4-panel Seaborn EDA figure
├── src/
│   ├── data_preprocessing.py            # Automated script for cleaning, merging, EDA plotting & splitting
│   ├── eda.ipynb                        # Interactive Jupyter Notebook for data exploration & cleaning
│   ├── features.py                      # Calendar / lag / weather feature engineering (leakage-safe)
│   ├── evaluate.py                      # Shared metrics, ResultsLog and diagnostic plots (use for ALL models)
│   └── train_models.py                  # Baselines, Ridge/Lasso, Random Forest, ablation, horizon study
├── results/                             # metrics.csv, tuning_*.csv, ablation, importances, test predictions
├── tests/
│   ├── test_data_preprocessing.py       # Automated pytest suite verifying data preprocessing & splits
│   └── test_features_and_evaluate.py    # No-look-ahead lag tests, calendar/timezone tests, metric tests
├── pyproject.toml                       # Project dependencies managed by uv (Python >= 3.12)
└── uv.lock                              # Reproducible dependency lockfile
```

---

## 3. Dataset Setup

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) and extract `energy_dataset.csv` and `weather_features.csv` into the `data/` folder:

```text
data/
├── energy_dataset.csv
└── weather_features.csv
```

---

## 4. Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Run data cleaning, 4-panel EDA plot, and chronological split
uv run python src/data_preprocessing.py

# 3. Build features (prints the feature list and split shapes)
uv run python src/features.py

# 4. Train baselines, Ridge/Lasso, Random Forest (+ ablation & horizon study), ~5-10 min
uv run python src/train_models.py            # add --fast for a quick smoke test

# 5. Run automated pipeline verification tests
uv run pytest -q
```

---

## 5. Using the shared modules (for Part D / E)

```python
import sys; sys.path.append("src")
from features import FeatureConfig, make_datasets, feature_columns, TARGET
from evaluate import ResultsLog

data = make_datasets(FeatureConfig(horizon=24))          # same features for every model
cols = feature_columns(data["train"])
log = ResultsLog()
# ... fit on data["train"], tune on data["val"], refit on train+val ...
log.add("LightGBM", "test", data["test"][TARGET], y_pred)
```

Protocol used for every model: tune hyper-parameters on **train → val** (MAE), refit the chosen
configuration on **train + val**, score once on **test**. `total load forecast` (TSO) is a reference
benchmark only and is never used as an input feature.
