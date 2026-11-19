# Energy Consumption, Generation, Prices & Weather Forecasting

End-to-end data preprocessing, Exploratory Data Analysis (EDA), feature engineering, and machine learning pipeline for the [Hourly Energy Demand Generation and Weather (Spain)](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) dataset.

---

## 1. Project Workflow (4 Stages)

1. **Data Preprocessing & EDA (`src/data_preprocessing.py`, `src/eda.ipynb`):**
   - Download raw datasets, clean missing values and sensor outliers, merge national energy and 5-city weather data on UTC timestamps, generate the 4-panel EDA figure (`plots/eda_4panel.png`), and create chronological **Train / Validation / Test** splits (`data/splits/`).
2. **Feature Engineering:**
   - Construct calendar features, time-series lag/rolling features, and domain-specific weather features from the chronological splits.
3. **Baseline & Linear / Tree Models (`src/evaluate.py`):**
   - Implement the shared evaluation module (`evaluate.py`) and train Baseline, Ridge/Lasso, and Random Forest models.
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
│   └── eda.ipynb                        # Interactive Jupyter Notebook for data exploration & cleaning
├── tests/
│   └── test_data_preprocessing.py       # Automated pytest suite verifying data preprocessing & splits
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

# 3. Run automated pipeline verification tests
uv run pytest -q
```
