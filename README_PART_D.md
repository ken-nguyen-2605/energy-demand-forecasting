# Phần D — Gradient Boosting (LightGBM & XGBoost), Tinh chỉnh Siêu tham số & Giải thích TreeSHAP

Tài liệu hướng dẫn nhanh & bàn giao kỹ thuật cho **Phần D** (Đồ án môn học **CO3135 — Programming for AI & Data Science**, dự báo phụ tải điện theo giờ tại Tây Ban Nha giai đoạn 2015–2018, chân trời trước 1 ngày `horizon = 24h`).

> **Đã đồng bộ hoàn toàn lên nền `ff1758c` (`fix/data_alignment_and_baselines`)** và tuân thủ 100% các hướng dẫn trong [`docs/HANDOFF_PART_D.md`](docs/HANDOFF_PART_D.md).

---

## 1. Tuân thủ giao thức chung của nhóm (0 rò rỉ dữ liệu, 0 sửa code A/B/C)

- **Dữ liệu đã sửa lỗi đảo ngày/tháng**: Sử dụng nguyên vẹn 100% tập chia theo thời gian (`70 / 15 / 15`) xuất từ `src/data_preprocessing.py` (`fix_day_month_swap()`, đã khôi phục `12.672` giờ).
- **Bộ 42 đặc trưng Phần B**: Gọi trực tiếp `make_datasets(FeatureConfig(horizon=24))` từ [`src/features.py`](src/features.py) (`Calendar: 18`, `Lag / Rolling: 13`, `Weather: 11`), gồm 3 cột lịch mới (`cal_is_special`, `cal_lag1d_is_off`, `cal_lag7d_is_off`) và lag tuần theo giờ đồng hồ DST.
- **Đánh giá dùng chung**: Chấm đầy đủ **7 chỉ số** (`MAE, RMSE, MAPE, sMAPE, R2, MaxAE, ME`) qua `ResultsLog` trong [`src/evaluate.py`](src/evaluate.py), cố định `SEED = 0`.
- **Giao thức không rò rỉ dữ liệu (khớp tuyệt đối Phần C)**:
  - **Bước 1 & 2**: Huấn luyện trên `train` (`23.872` giờ), so sánh hàm mất mát và tinh chỉnh **Optuna TPE** (`40` trials LightGBM + `30` trials XGBoost) theo **Validation MAE** trên `val` (`5.260` giờ) — *khóa hoàn toàn tập `test`*.
  - **Bước 3**: Đóng băng cấu hình tốt nhất (bao gồm `best_iteration` lấy từ `val`), huấn luyện lại trên `train + val` (`29.132` giờ) và chấm điểm **1 lần duy nhất** trên `test` (`5.260` giờ).
  - **Bước 4**: Giải thích bằng **Exact TreeSHAP** trên cả **`val` (`train → val`)** để so sánh công bằng với RF Permutation (Phần C) lẫn **`test` (`train + val → test`)** cho mô hình cuối.

---

## 2. Tóm tắt kết quả chính (trên bộ dữ liệu sạch 42 đặc trưng)

### 2.1. Hiệu năng trên tập `test` (`5.260` giờ, day-ahead `h = 24h`)

| Mô hình | Hàm Loss | Test MAE (MW) | Test RMSE (MW) | Test MAPE (%) | Test $R^2$ | MaxAE* (MW) | ME (MW) | Thời gian fit (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive (`t - 24h`) *(Phần C)* | — | 1.804,2 | 2.803,3 | 6,328 | 0,603 | 12.282 | −1,7 | — |
| Seasonal naive (`t - 168h`) *(Phần C)* | — | 1.233,6 | 1.939,0 | 4,299 | 0,810 | 16.410 | +25,3 | — |
| Ridge *(Phần C)* | $L_2$ | 666,4 | 885,9 | 2,384 | 0,960 | 13.461 | −76,4 | 0,02 |
| Lasso *(Phần C)* | $L_2 + L_1$ | 665,1 | 885,6 | 2,380 | 0,960 | 13.426 | −77,7 | 5,22 |
| Random Forest *(Phần C)* | $L_2$ | 425,0 | 628,0 | 1,485 | 0,980 | 11.940 | −30,5 | 11,46 |
| LightGBM (mặc định) | $L_2$ | 448,0 | 639,8 | 1,578 | 0,979 | 11.958 | −107,3 | 2,64 |
| XGBoost (mặc định) | $L_2$ | 448,1 | 635,7 | 1,581 | 0,980 | 11.622 | −88,2 | 0,29 |
| **XGBoost (Tuned — Optuna)** | **$L_2$ (MSE)** | **371,3** | **540,0** | **1,298** | **0,985** | 11.904 | −78,3 | **9,37** |
| **LightGBM (Tuned — Optuna)** | **$L_2$ + `cat`** | **368,0** | **530,4** | **1,286** | **0,986** | **11.803** | −69,1 | **15,50** |
| *TSO forecast (tham chiếu)* | — | *252,1* | *379,1* | *0,873* | *0,993* | *12.207* | *−14,9* | — |

*\*Lưu ý theo `docs/HANDOFF_PART_D.md`: `MaxAE` (~11.800–12.200 MW) ở mọi mô hình kể cả TSO đều do 1 giờ gai cảm biến `2018-06-05 10:00 UTC` (`19.964 MW` giữa hai giờ ~`32.000 MW`).*

- **Cải thiện vượt trội so với Random Forest (Phần C)**:
  - **LightGBM (Tuned)** giảm **`-57,0 MW` (`-13,4%`) Test MAE**, giảm **`-97,7 MW` (`-15,6%`) Test RMSE**, nâng $R^2$ từ `0,980` lên **`0,986`** (khai báo `categorical_feature=["cal_hour", "cal_dayofweek", "cal_month"]` giúp giảm thêm `-8 MW` Val MAE đúng như gợi ý trong `HANDOFF_PART_D.md`).
  - **XGBoost (Tuned)** giảm **`-53,7 MW` (`-12,6%`) Test MAE**, giảm **`-88,0 MW` (`-14,0%`) Test RMSE**, đạt $R^2 = 0{,}985$.
  - **Ý nghĩa của việc tinh chỉnh**: Cả hai mô hình mặc định (`448,0` & `448,1 MW`) đều **thua Random Forest (`425,0 MW`)** trên `test`; sau khi tinh chỉnh bằng Optuna TPE, sai số giảm tới **`-80 MW`** và vượt RF rất sâu.
- **Kiểm định độ ổn định đa seed (`seeds = 0..4` trên `test`)**:
  - `LightGBM`: MAE = **`369,85 ± 3,48 MW`**, RMSE = `534,14 ± 5,01 MW`, MAPE = `1,293 ± 0,012%`, $R^2 = 0{,}986 \pm 0{,}000$.
  - `XGBoost`: MAE = **`370,61 ± 3,09 MW`**, RMSE = `538,32 ± 4,20 MW`, MAPE = `1,296 ± 0,011%`, $R^2 = 0{,}985 \pm 0{,}000$.
- **So sánh chân trời dự báo (`h = 1h` vs. `h = 24h` trên `val`)**:
  - Ở `h = 1h`: **LightGBM đạt `182,5 MW` (`MAPE = 0,623%`)** và **XGBoost đạt `195,6 MW` (`MAPE = 0,668%`)** — vượt xa Random Forest (`257,5 MW`, `-29,1%`) và **vượt cả dự báo tham chiếu TSO trên `val` (`309,4 MW`, `1,053%`)**.

---

## 3. Dành riêng cho Người E (Mục 5 — So sánh toàn cục & Kiểm định thống kê)

Toàn bộ kết quả Phần C và Phần D đã được **ghép sẵn trên cùng nền dữ liệu `ff1758c` và khớp tuyệt đối theo cột `time`** trong 5 file sau tại thư mục `results/`:

| File | Nội dung | Công dụng cho Mục 5 |
| :--- | :--- | :--- |
| [`results/metrics_all.csv`](results/metrics_all.csv) | Bảng đủ **7 chỉ số** (gồm cả `ME`) cho toàn bộ mô hình Phần C + Phần D trên `val` & `test` | Bảng tổng hợp chính thức Mục 5 |
| [`results/predictions_test_all.parquet`](results/predictions_test_all.parquet) | `5.260` dòng dự báo trên `test` (merge theo `time`, khớp 100% với `predictions_test.parquet` của Phần C) | Chạy kiểm định **Diebold–Mariano / Wilcoxon** giữa các mô hình |
| [`results/mae_by_hour_all.csv`](results/mae_by_hour_all.csv) | Test MAE theo 24 giờ địa phương (`Europe/Madrid`) | Phân tích sai số giờ cao điểm / thấp điểm |
| [`results/mae_by_dow_all.csv`](results/mae_by_dow_all.csv) | Test MAE theo 7 thứ trong tuần | Phân tích sai số ngày thường vs. cuối tuần |
| [`results/horizon_comparison_all.csv`](results/horizon_comparison_all.csv) | So sánh `h = 1h` vs. `h = 24h` trên `val` cho Naive, Ridge, RF, LightGBM & XGBoost | Bảng độ nhạy theo chân trời dự báo |

---

## 4. Tóm tắt phát hiện chính từ Exact TreeSHAP (Phục vụ Mục 6)

- **Kiểm chứng tiên đề cộng tính cục bộ (*Local Accuracy*)**:
  - LightGBM (`test`): $\max_i \lvert \hat{y}_i - (\phi_0 + \sum_j \phi_{i,j}) \rvert = \mathbf{3{,}20 \times 10^{-10}\text{ MW}}$ (khớp tuyệt đối mức `float64`, giá trị cơ sở $\phi_0 = 28.635{,}34\text{ MW}$).
  - LightGBM (`val`, fit trên `train`): **`2,69 × 10⁻¹⁰ MW`**.
- **So sánh công bằng trên `val` (`train → val`) với RF Permutation Importance (theo `HANDOFF_PART_D.md`)**:
  - `Spearman(LightGBM Val SHAP, RF Val Perm) = 0,8201` — **Top 4 đặc trưng giống nhau 100%** (`lag_same_hour_4w_mean`, `lag_24`, `lag_168`, `cal_is_offday`).
  - Khác biệt đáng chú ý: LightGBM khai thác tốt hơn rõ rệt nhóm nhiệt độ (`w_temp_roll24` hạng **#5**, `w_temp_c_sq` hạng **#8** so với hạng #18–#20 ở RF) và biến phân loại `cal_month` (hạng **#7** trên `test` nhờ `categorical_feature`); đặc trưng mới `cal_lag1d_is_off` của Phần B lọt ngay **Top 15**.
- **Cộng gộp theo 3 họ đặc trưng Phần B (42 biến)**:
  - **Lag / Rolling (13 biến)**: **`3.355,9 MW` (`73,73%`)** | **Calendar (18 biến)**: **`926,8 MW` (`20,36%`)** | **Weather (11 biến)**: **`268,7 MW` (`5,90%`)** — gần như trùng khớp hoàn toàn giữa LightGBM và XGBoost (`3.381,0 / 913,6 / 264,6 MW`).
- **Đường cong chữ U của nhiệt độ**: Cộng gộp SHAP 5 biến nhiệt độ (`w_temp_c + w_temp_c_sq + w_hdh + w_cdh + w_temp_roll24`) hiện lên đường cong chữ U cực kỳ sắc nét với đáy thoải mái đúng khoảng **`18°C – 22°C`**, tăng tới **`+1.200 MW`** khi lạnh (`<10°C`) và **`+1.900 MW`** khi nắng nóng đỉnh điểm (`>33°C`).
- **Phân tích 3 tình huống cục bộ (Local Waterfall) + ghi nhận giờ gai cảm biến**:
  - **Đỉnh hè (`2018-08-03 13:00 CEST`, Thứ Sáu)**: Thực tế `39.756 MW`, Dự báo **`39.457 MW`** (sai chỉ **`+299 MW` / `0,75%`**!) — cụm nhiệt độ nóng (`w_temp_roll24 = 29,5°C` đóng `+738 MW`, `w_temp_c_sq` đóng `+497 MW`) đóng góp trên **`+1.235 MW`**.
  - **Ngày Quốc khánh (`2018-10-12 12:00 CEST`, Thứ Sáu)**: Thực tế `26.606 MW`, Dự báo **`26.743 MW`** (sai chỉ **`-137 MW` / `0,5%`**!) — `cal_is_offday (-2.133 MW)` và `cal_is_holiday (-1.721 MW)` tự động kéo giảm `-3.854 MW` triệt tiêu đà Thứ Sáu thường.
  - **Giờ sai số thực lớn nhất (loại trừ gai cảm biến `05/06`) — Sáng Giáng Sinh `2018-12-25 09:00 CET`**: Thực tế `20.540 MW`, Dự báo `23.746 MW` (`-3.206 MW`) — dù `cal_is_offday + cal_is_holiday` đã trừ tới `-4.800 MW`, Giáng Sinh là ngày giảm tải sâu nhất năm trong khi cờ nhị phân `cal_is_holiday` coi mọi ngày lễ như nhau (gợi ý cải tiến rõ ràng cho hướng phát triển!).
  - **Gai cảm biến đã ghi nhận (`2018-06-05 12:00 CEST / 10:00 UTC`)**: Thực tế `19.964 MW`, Dự báo `31.767 MW` (`-11.803 MW`), khớp hoàn toàn cảnh báo của `docs/HANDOFF_PART_D.md`.

---

## 5. Hướng dẫn cài đặt & chạy kiểm chứng lại

Các thư viện Phần D (`lightgbm`, `xgboost`, `shap`, `optuna`, `scipy`) đã được khai báo vào [`pyproject.toml`](pyproject.toml). *(Lưu ý: nếu máy bạn dùng `uv`, chạy `uv lock && uv sync` để cập nhật `uv.lock` khớp với `pyproject.toml` mới).*

```bash
# 1. Cài dependencies (chọn 1 trong 2 cách)
uv sync
# (hoặc với pip: pip install lightgbm xgboost shap optuna scipy pytest)

# 2. Chạy toàn bộ unit tests của cả nhóm (Phần A + B + C + D)
pytest -v

# 3. Chạy từng bước của pipeline Phần D bằng CLI
python src/train_boosting.py --step 1   # Untuned + Learning Curve (train -> val)
python src/train_boosting.py --step 2   # Loss Comparison (2A) + Optuna TPE (2B)
python src/train_boosting.py --step 3   # Refit train+val -> test, 5-seed, h=1 vs h=24
python src/train_boosting.py --step 4   # Exact TreeSHAP (Test & Fair Val) + 6 biểu đồ

# 4. Chạy kiểm chứng sạch notebook tổng hợp (Restart & Run All ~40 giây)
jupyter nbconvert --to notebook --execute --inplace src/boosting_shap.ipynb
```

---

## 6. Danh mục tệp Phần D & Gợi ý dùng 14 biểu đồ (Slide vs. Report)

- **Mã nguồn, Notebook & Kiểm thử**: [`src/train_boosting.py`](src/train_boosting.py), [`src/boosting_shap.ipynb`](src/boosting_shap.ipynb), [`tests/test_boosting.py`](tests/test_boosting.py), [`README_PART_D.md`](README_PART_D.md), và bổ sung 5 gói vào [`pyproject.toml`](pyproject.toml).
- **15 tệp kết quả (`results/`)**: 5 file gộp cho Người E (`metrics_all.csv`, `predictions_test_all.parquet`, `mae_by_hour_all.csv`, `mae_by_dow_all.csv`, `horizon_comparison_all.csv`) + 10 file chi tiết (`metrics_boosting.csv`, `d_step1_untuned.csv`, `d_step2_loss_comparison.csv`, `d_step2_tuning_lgbm.csv`, `d_step2_tuning_xgb.csv`, `d_step2_best_params.json`, `d_step3_multiseed.csv`, `d_step4_shap_importance.csv`, `d_step4_shap_by_group.csv`, `d_step4_local_cases.csv`).
- **14 biểu đồ (`plots/d_step*.png`)**:

| Biểu đồ | Nội dung | Gợi ý dùng cho |
| :--- | :--- | :--- |
| `plots/d_step1_learning_curve.png` | Đường cong học tập `1 → 600` cây ($L_2$, `lr=0.1`) | Report (Mục 4.3) / Backup slide |
| `plots/d_step2_tuning_curves.png` | Hội tụ Optuna TPE (LightGBM 40 trials, XGBoost 30 trials) | **Slide chính** + Report (Mục 4.3) |
| `plots/d_step3_mae_test_all.png` | So sánh Test MAE toàn bộ mô hình (LightGBM `368 MW` dẫn đầu) | **Slide chính** + Report (Mục 4.3 / 5) |
| `plots/d_step3_mape_test_all.png` | So sánh Test MAPE toàn bộ mô hình | Phụ lục / Người E |
| `plots/d_step3_forecast_2weeks.png` | Dự báo 2 tuần trên `test` (RF vs. LightGBM vs. XGBoost) | Report (Mục 5 — Người E) / Backup |
| `plots/d_step3_forecast_holiday.png` | Tuần lễ Quốc khánh `12/10/2018` (Boosting ôm sát cú giảm lễ) | Report (Mục 4.3 / 5) / Backup |
| `plots/d_step3_mae_by_hour_all.png` | Test MAE theo 24 khung giờ địa phương across model families | Report (Mục 5 — Người E) / Backup |
| `plots/d_step3_residuals.png` | Chẩn đoán phần dư hậu huấn luyện của LightGBM ($R^2 = 0{,}986$) | Report (Mục 4.3) / Backup |
| `plots/d_step4_shap_beeswarm.png` | TreeSHAP Beeswarm Top 15 trên `test` | **Slide chính** + Report (Mục 6) |
| `plots/d_step4_shap_vs_rf.png` | Đối chiếu công bằng trên `val`: LightGBM SHAP vs. RF Permutation (`r = 0,820`) | **Slide chính** + Report (Mục 6) |
| `plots/d_step4_shap_dependence.png` | Tương tác `lag_24 × cal_is_offday` & **đường cong chữ U** nhiệt độ `18°C–22°C` | **Slide chính** + Report (Mục 6) |
| `plots/d_step4_waterfall_peak.png` | Local Waterfall: đỉnh hè `2018-08-03 13:00` (sai chỉ `+299 MW`) | Report (Mục 6) / Backup |
| `plots/d_step4_waterfall_holiday.png` | Local Waterfall: Quốc khánh `2018-10-12 12:00` (sai chỉ `-137 MW`) | **Slide chính** + Report (Mục 6) |
| `plots/d_step4_waterfall_worst_error.png` | Local Waterfall: sáng Giáng Sinh `2018-12-25 09:00` (sai số thực lớn nhất) | **Slide chính** + Report (Mục 6) |
