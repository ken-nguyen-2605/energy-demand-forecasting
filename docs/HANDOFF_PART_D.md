# Bàn giao phần C cho phần D (LightGBM, XGBoost, SHAP)

Nhánh `fix/data_alignment_and_baselines`. Bài toán: dự báo `total load actual` trước **24 giờ** (lag ≥ 24 giờ, thời tiết tại giờ mục tiêu coi như dự báo hoàn hảo).

## Dữ liệu đã sửa

File Kaggle `energy_dataset.csv` bị đảo ngày và tháng ở các ngày 1–12 cho mọi cột ENTSO-E và cột `price day ahead`. Chỉ `price actual` đúng. `src/data_preprocessing.py` đã sửa việc này trong `fix_day_month_swap()` (12.672 giờ). Hãy luôn tạo splits bằng script này:

```bash
uv run python src/data_preprocessing.py
```

Không dùng `src/eda.ipynb` để tạo splits: notebook vẫn đọc CSV gốc chưa sửa và sẽ ghi đè `data/splits`.

## Lấy dữ liệu và đặc trưng

```python
import sys; sys.path.insert(0, "src")
from features import FeatureConfig, make_datasets, feature_columns, TARGET   # TARGET = "total load actual"
from evaluate import ResultsLog, error_by_group, plot_forecast, plot_metric_bars

data = make_datasets(FeatureConfig(horizon=24))      # dict: train 23.872 / val 5.260 / test 5.260 dòng
cols = feature_columns(data["train"])                # 42 đặc trưng, giống results/summary.json["features"]
```

- `time` là UTC. `total load forecast` (TSO) chỉ để so sánh, **không** đưa vào đầu vào.
- `cal_hour`, `cal_dayofweek`, `cal_month` là số nguyên. Muốn LightGBM coi là categorical thì phải khai báo rõ.
- Cột mới của phần B: `cal_is_special` (ngày nghỉ thực tế không phải lễ quốc gia), `cal_lag1d_is_off`, `cal_lag7d_is_off` (ngày cách 1 / 7 ngày có phải ngày nghỉ).
- `lag_168`, `lag_336`, `lag_same_hour_4w_mean` lấy cùng giờ đồng hồ Madrid của 7/14/21/28 ngày trước (cách thực 167–169 giờ quanh ngày đổi giờ). Baseline `Seasonal naive (t-168h)` và `Mean of last 4 weeks` dùng chính các cột này; tên cột giữ nguyên để merge.

## Giao thức (giống phần C để so sánh công bằng)

1. Tinh chỉnh: fit trên `train`, chọn theo MAE trên `val`. Ghi dòng `split="val"` bằng mô hình fit trên train.
2. Fit lại cấu hình đã chọn trên `pd.concat([train, val], ignore_index=True)` rồi dự báo `test` **đúng một lần**, ghi dòng `split="test"` kèm `fit_s`.
3. Nếu dùng early stopping: lấy `best_iteration` trên val, rồi fit lại trên train+val với số cây cố định đó. Không dùng test cho early stopping.
4. Seed = 0 (như `train_models.py`).

## Ghi kết quả, không ghi đè kết quả của C

`train_models.py` ghi đè toàn bộ `results/` và `plots/` mỗi lần chạy, kể cả với `--fast`. Phần D nên ghi file riêng:

```python
log = ResultsLog()
log.add("LightGBM", "val", y_val, p_val, family="boosting")
log.add("LightGBM", "test", y_test, p_test, family="boosting", fit_s=secs)
log.save(Path("results/metrics_boosting.csv"))
# bảng chung: pd.concat([pd.read_csv("results/metrics.csv"), pd.read_csv("results/metrics_boosting.csv")])
```

- Các cột chỉ số là `MAE, RMSE, MAPE, sMAPE, R2, MaxAE, ME`. ME = dự báo − thực tế, âm nghĩa là dự báo thấp.
- Dự báo test: thêm cột `LightGBM`, `XGBoost` vào `results/predictions_test.parquet` bằng cách merge theo `time` (5.260 dòng), hoặc ghi `results/predictions_test_boosting.parquet`.
- Chẩn đoán dùng lại được: `error_by_group(time, y, pred, "hour"|"dayofweek"|"month")`, `plot_forecast(...)`, `plot_metric_bars(table, "MAE", path)`.

## Kết quả phần C để so sánh (test, MW)

| Mô hình | MAE | RMSE | MAPE | ME |
|---|---|---|---|---|
| TSO (tham chiếu) | 252,1 | 379,1 | 0,87% | −14,9 |
| Random Forest | 425,0 | 628,0 | 1,49% | −30,5 |
| Lasso | 665,1 | 885,6 | 2,38% | −77,7 |
| Ridge | 666,4 | 885,9 | 2,38% | −76,4 |
| Seasonal naive (t-168h) | 1.233,6 | 1.939,0 | 4,30% | +25,3 |

- Tham số Random Forest tốt nhất: `max_depth=None, min_samples_leaf=1, max_features=0.5`.
- So SHAP với `results/feature_importance_rf.csv`: permutation importance trong file đó tính trên val, với Random Forest fit trên train. Muốn so công bằng thì tính SHAP trên val với mô hình fit trên train.
- Một lần thử nhanh LightGBM theo đúng giao thức trên (với bộ 39 đặc trưng cũ, trước khi sửa phần B) cho MAE test khoảng 393. Đây chỉ là số tham khảo, chưa tinh chỉnh.

## Đã sửa ở phần B (nhánh này)

- Ngày lễ quốc gia tính cho 2014–2019, Thứ Sáu Tuần Thánh tính bằng công thức Phục Sinh; `cal_is_bridge` của 31/12/2018 đã đúng.
- Thêm cờ ngày nghỉ phụ (Thứ Năm Tuần Thánh, Thứ Hai Phục Sinh, 24/12, 31/12, Thứ Hai sau lễ rơi vào Chủ Nhật) và cờ ngày nghỉ của ngày trễ.
- Lag theo tuần lấy theo giờ đồng hồ địa phương, chỉ áp dụng khi lag lớn hơn horizon nên không bao giờ gần mục tiêu hơn horizon.

## Lưu ý đã biết (chưa sửa)

- **Giờ lỗi đo trong test:** 2018-06-05 10:00 UTC có tải 19.964 MW, trong khi giờ trước và sau khoảng 32.000 MW. Giờ này quyết định `MaxAE` và làm tăng `RMSE` của mọi mô hình, kể cả TSO (bỏ giờ này đi thì RMSE của TSO giảm từ 379,1 xuống 339,7). Tập train còn 4 gai tương tự. Đừng diễn giải `MaxAE`. Nhóm cần chốt có nội suy các giờ gai ở phần A hay không.
- **Thư viện chưa khai báo:** `lightgbm`, `xgboost`, `shap` chưa có trong `pyproject.toml` và `uv.lock`. Phần D thêm vào, ghim phiên bản, và commit lại `uv.lock`.
- **Alpha chạm biên lưới:** alpha tối ưu của Ridge (0,01) và Lasso (0,1) nằm ở biên dưới. Đường cong gần như phẳng, nên Ridge gần như không có điều chuẩn.
- **Vài ngày giảm tải chưa có cờ:** 26/12/2017 (Sant Esteve), 27–28/12/2016, 24/06/2015 (lễ vùng). Ngày 02/11/2015 có cờ `cal_is_special` nhưng tải chỉ giảm khoảng 3%.
- **Báo cáo LaTeX** (`Report (For section B & C)`) vẫn còn số, hình và công thức cũ ($y_{t-168}$, `y.shift(lag)`) từ trước khi sửa dữ liệu và phần B.
