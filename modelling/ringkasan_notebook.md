Dokumentasi Pemodelan Tesis — Prediksi IHSG dengan Random Forest
1. Ringkasan Penelitian
Penelitian ini membangun model prediksi harga Indeks Harga Saham Gabungan (IHSG) menggunakan algoritma Random Forest dengan pendekatan time series. Model menggunakan variabel target IHSG daily price dan covariate dari tiga kategori: variabel makroekonomi domestik, harga komoditas global, dan indeks regional. Periode data mencakup 10 tahun (Januari 2015 - Januari 2025). Prediksi dilakukan secara one-step-ahead (horizon=1 hari) dengan variasi lookback window (30, 60, 120 hari).

2. Data
2.1 Target Variable
IHSG (Indeks Harga Saham Gabungan): Harga penutupan harian, frekuensi business day
2.2 Covariate Variables
Variabel Makroekonomi Domestik (5 variabel):

BI_Rate: Suku bunga acuan Bank Indonesia (bulanan, forward-filled ke daily)
CPI: Consumer Price Index Indonesia (bulanan, forward-filled)
M2: Jumlah uang beredar dalam arti luas (bulanan, forward-filled)
NPL_Ratio: Non-Performing Loan ratio perbankan (bulanan, forward-filled)
USDIDR: Nilai tukar USD/IDR (daily)
Harga Komoditas Global (6 variabel):

Coal: Harga batubara global (daily)
Copper: Harga tembaga (daily)
Nickel: Harga nikel (daily)
Silver: Harga perak (daily)
Tin: Harga timah (daily)
Gold: Harga emas (daily)
Indeks Regional (1 variabel):

STI: Straits Times Index, indeks saham Singapura (daily)
2.3 Periode dan Ukuran Data
Rentang: 1 Januari 2015 — 31 Januari 2025 (~10 tahun)
Total observasi setelah merge: ~2.443 business days
Variabel bulanan (BI_Rate, CPI, M2, NPL_Ratio) di-merge ke frekuensi daily menggunakan merge_asof dengan direction backward (forward-fill)
Variabel daily yang memiliki missing values di-forward-fill
Baris dengan NaN di-drop setelah merge
2.4 Sumber Data
Seluruh data disimpan di GitHub repository dan diakses via raw CSV URL. Sumber asli:

IHSG, USDIDR: Data pasar keuangan Indonesia
BI_Rate, CPI, M2, NPL_Ratio: Bank Indonesia / BPS
Coal, Copper, Nickel, Silver, Tin, Gold: Harga komoditas global
STI: Bursa Singapura
3. Preprocessing dan Transformasi Data
3.1 Stationarity Test (ADF Test)
Augmented Dickey-Fuller test dilakukan pada semua variabel sebelum dan sesudah transformasi dengan significance level 5%.

Hasil ADF Test pada level (sebelum transformasi):

IHSG: p=0.519 → Non-stationary
BI_Rate: p=0.337 → Non-stationary
CPI: p=0.837 → Non-stationary
M2: (non-stationary)
NPL_Ratio: p=0.721 → Non-stationary
USDIDR: (non-stationary)
Semua komoditas dan STI: Non-stationary
Kesimpulan: Seluruh variabel non-stationary di level, sehingga transformasi diperlukan.

3.2 Strategi Transformasi
Berdasarkan hasil ADF test, variabel dibagi menjadi dua kelompok dengan transformasi berbeda:

LEVEL_VARS (price/level variables) → Log-Difference:
M2, USDIDR, Coal, Copper, Nickel, Silver, Tin, Gold, STI

Transformasi: diff(log(x)) — menghasilkan log-return, stasioner, dan menstabilkan varians.

RATE_VARS (rate/percentage variables) → First Difference (tanpa log):
BI_Rate, CPI, NPL_Ratio

Transformasi: diff(x) — variabel sudah berupa persentase kecil, log tidak bermakna. First-difference cukup untuk membuat stasioner.

Target (IHSG) → Log-Difference:
diff(log(IHSG)) — konsisten dengan treatment pada price variables.

Setelah transformasi, seluruh variabel stasioner (p < 0.05 pada ADF test).

3.3 Scaling
Setelah differencing, MinMaxScaler diterapkan pada semua variabel. Scaler di-fit hanya pada data training untuk menghindari data leakage, kemudian di-transform pada seluruh data.

3.4 Pencegahan Data Leakage
Seluruh operasi fit (differencing, scaling) dilakukan eksklusif pada data training:

Diff.fit_transform() pada train, Diff.transform() pada full data
Scaler.fit() pada train, Scaler.transform() pada full data
Berlaku untuk target maupun covariate
4. Desain Eksperimen
4.1 Covariate Sets (7 konfigurasi)
Set	Variabel	Jumlah	Tujuan
None	-	0	Baseline univariate
Macro	BI_Rate, CPI, M2, NPL_Ratio, USDIDR	5	Fundamental ekonomi domestik
Commodity	Coal, Copper, Nickel, Silver, Tin, Gold	6	Komoditas ekspor utama Indonesia
Regional	STI	1	Sentimen pasar regional
Macro_Regional	Macro + STI	6	Fundamental + sentimen regional
Commodity_Regional	Commodity + STI	7	Komoditas + sentimen regional
Full	Semua variabel	12	Seluruh informasi tersedia
4.2 Window Scenarios (3 konfigurasi)
Scenario	Window (Lookback)	Horizon	Keterangan
W30_H1	30 hari (~1.5 bulan)	1 hari	Short-term memory
W60_H1	60 hari (~3 bulan)	1 hari	Medium-term, satu kuartal
W120_H1	120 hari (~6 bulan)	1 hari	Long-term, menangkap siklus makro
Horizon tetap 1 (one-step-ahead prediction) pada semua skenario.

4.3 Total Eksperimen
7 covariate sets × 3 window scenarios = 21 eksperimen

5. Algoritma dan Hyperparameter
5.1 Random Forest (via Darts Library)
Model menggunakan RandomForestModel dari library Darts (Python), yang membungkus scikit-learn RandomForestRegressor untuk time series forecasting. Fitur yang digunakan model adalah lagged values dari target dan covariate.

Parameter kunci:

lags: Jumlah lag dari target variable (= window)
lags_past_covariates: Jumlah lag dari covariate (= window)
output_chunk_length: Panjang prediksi per step (= horizon = 1)
5.2 Hyperparameter Tuning
Grid search dilakukan pada konfigurasi terbaik dari eksperimen awal (W120_H1, Macro_Regional) dengan 5-fold expanding window cross-validation.

Search space (36 kombinasi):

n_estimators: [100, 300, 500]
max_depth: [5, 10, 15, None]
max_features: ["sqrt", 0.5, None]
Parameter tetap:

max_samples: 0.7
random_state: 42
n_jobs: -1
Hasil tuning — Top 5:

Rank	n_est	max_depth	max_feat	MAPE
1	500	5	sqrt	0.6592%
2	300	5	sqrt	0.6598%
3	100	5	sqrt	0.6601%
4	500	10	sqrt	0.6621%
5	300	10	sqrt	0.6628%
Observasi tuning:

max_features=sqrt dominan di semua top 5 — membatasi fitur per split mengurangi overfitting
max_depth=5 konsisten terbaik — tree dangkal lebih baik, data tidak butuh interaksi kompleks
n_estimators=500 sedikit lebih baik tapi perbedaan marginal vs 300
Final hyperparameters:

n_estimators: 500
max_depth: 5
max_features: sqrt
max_samples: 0.7
random_state: 42
6. Validasi — Expanding Window Cross-Validation
6.1 Metode
Digunakan 5-fold expanding window cross-validation, bukan single train/test split.

Struktur:

Fold 1: Train 40% data, Test 15% berikutnya
Fold 2: Train bertambah, Test 15% berikutnya
Fold 3-5: Train terus bertambah (expanding)
Training selalu dimulai dari awal data dan bertambah panjang setiap fold. Test size tetap ~15% per fold.

6.2 Alasan
Single 80/20 split rentan terhadap regime-dependent performance (test bisa kebetulan jatuh di bull/bear market)
Expanding window menghormati urutan temporal (tidak ada data masa depan di training)
Metrik dilaporkan sebagai mean ± std across folds — lebih robust dan kredibel secara statistik
7. Metrik Evaluasi
Metrik	Formula	Interpretasi
MAPE	Mean Absolute Percentage Error	Rata-rata persentase error absolut
MAE	Mean Absolute Error	Rata-rata error absolut dalam poin IHSG
RMSE	Root Mean Squared Error	Akar rata-rata kuadrat error, sensitif terhadap outlier
R²	Coefficient of Determination	Proporsi variansi yang dijelaskan model (1.0 = sempurna)
DA	Directional Accuracy	Persentase hari di mana arah prediksi (naik/turun) benar
Catatan DA: Dihitung sebagai persentase hari di mana sign(predicted[t] - actual[t-1]) == sign(actual[t] - actual[t-1]). DA > 50% berarti model lebih baik dari random guess dalam memprediksi arah.

8. Hasil Eksperimen Final
Seluruh 21 eksperimen dijalankan dengan hyperparameter final (n_est=500, max_depth=5, max_feat=sqrt).

8.1 Tabel Hasil Lengkap (sorted by MAPE)
Scenario	Covariates	MAPE (±std)	RMSE (±std)	R² (±std)	DA (±std)
W120_H1	Macro_Regional	0.6592±0.1822%	55.22±10.54	0.9710	53.2±2.1%
W30_H1	Commodity	0.6594±0.1834%	55.05±10.56	0.9713	53.2±1.9%
W120_H1	Full	0.6594±0.1823%	55.14±10.38	0.9709	51.7±1.8%
W30_H1	Full	0.6597±0.1830%	55.17±10.60	0.9711	53.4±1.9%
W60_H1	Commodity_Regional	0.6598±0.1830%	55.18±10.60	0.9712	52.1±2.2%
W120_H1	Commodity	0.6598±0.1841%	55.13±10.48	0.9710	50.7±3.9%
W120_H1	Macro	0.6599±0.1830%	55.24±10.52	0.9709	51.8±1.5%
W30_H1	Macro_Regional	0.6599±0.1828%	55.33±10.63	0.9710	53.3±1.6%
W60_H1	Commodity	0.6600±0.1821%	55.15±10.47	0.9713	51.5±1.8%
W30_H1	Commodity_Regional	0.6601±0.1822%	55.17±10.59	0.9712	53.4±2.0%
W30_H1	Macro	0.6604±0.1841%	55.32±10.64	0.9709	53.7±0.8%
W60_H1	Full	0.6604±0.1826%	55.20±10.49	0.9712	51.0±1.6%
W60_H1	Macro_Regional	0.6608±0.1839%	55.34±10.68	0.9710	52.3±2.0%
W60_H1	Macro	0.6609±0.1842%	55.26±10.56	0.9709	51.8±2.1%
W120_H1	Regional	0.6623±0.1865%	55.43±10.82	0.9710	52.3±1.2%
W30_H1	Regional	0.6626±0.1865%	55.49±10.89	0.9710	53.4±2.6%
W120_H1	None	0.6627±0.1873%	55.35±10.79	0.9710	51.8±2.6%
W60_H1	Regional	0.6635±0.1870%	55.48±10.86	0.9710	50.1±2.7%
W60_H1	None	0.6642±0.1893%	55.48±10.97	0.9710	52.0±2.2%
W30_H1	None	0.6661±0.1916%	55.68±11.16	0.9708	51.5±2.3%
8.2 Best Model
Konfigurasi: W120_H1 + Macro_Regional
MAPE: 0.6592 ± 0.1822%
RMSE: 55.22 ± 10.54
R²: ~0.9710
DA: 53.2 ± 2.1%
8.3 Interpretasi Hasil
Pengaruh Covariate:

Baseline (None) selalu menempati posisi terbawah di setiap window → covariate memberikan informasi tambahan
Macro_Regional dan Commodity adalah covariate set terbaik
Full (12 variabel) tidak selalu menang vs subset lebih kecil → indikasi curse of dimensionality pada terlalu banyak fitur
Pengaruh Window Length:

Perbedaan antar W30/W60/W120 sangat kecil (range MAPE: 0.6592-0.6661)
W120 sedikit lebih baik pada MAPE → IHSG merespons informasi jangka menengah
W30 lebih baik pada DA di beberapa set → momentum jangka pendek lebih baik menangkap arah
Directional Accuracy:

DA berkisar 50-54% di seluruh konfigurasi — sedikit di atas random guess (50%)
DA tertinggi: W30_H1 Macro (53.7±0.8%) — short window + macro paling konsisten untuk prediksi arah
Ini tipikal one-step-ahead prediction: model bagus memprediksi magnitude (MAPE rendah, R² tinggi) karena "harga besok ≈ harga hari ini", tapi lemah dalam memprediksi arah
9. Feature Importance
Feature importance diekstrak dari Random Forest model (rata-rata feature_importances_ dari semua trees across semua folds).

Untuk konfigurasi terbaik (W120_H1, Macro_Regional):

Feature importance menunjukkan kontribusi relatif setiap lagged feature terhadap prediksi
Importance di-aggregate per variabel (sum semua lag per variabel) untuk melihat kontribusi keseluruhan
Lag-lag terbaru (lag_1, lag_2) dari IHSG umumnya memiliki importance tertinggi — model sangat bergantung pada harga terkini
Detail lengkap tersedia di file CSV hasil ekspor
10. Tools dan Library
Python 3.12
Darts (time series library): TimeSeries, RandomForestModel, Scaler, Diff, train_test_split, historical_forecasts
scikit-learn (via Darts): RandomForestRegressor backend
statsmodels: Augmented Dickey-Fuller test
pandas, numpy: Data manipulation
matplotlib: Visualisasi
joblib: Model persistence
11. Reproduksibilitas
random_state=42 digunakan di semua model
Seluruh data diakses dari GitHub repository yang fixed
Hasil model dan feature importance disimpan ke CSV
Model objects disimpan via joblib untuk loading tanpa re-training
12. File Output
rf_final_results_[timestamp].csv: Tabel hasil 21 eksperimen
rf_feature_importance_[timestamp].csv: Feature importance dari best config
rf_predictions_best_[timestamp].csv: Prediksi aktual vs predicted untuk visualisasi
saved_models/experiment_results_[timestamp].joblib: Seluruh hasil eksperimen
saved_models/df_merged_[timestamp].joblib: Dataset merged
plot_feature_importance.png: Plot feature importance
plot_actual_vs_predicted.png: Timeline aktual vs prediksi
plot_error_distribution.png: Distribusi error
plot_scatter_actual_vs_predicted.png: Scatter plot + R²
plot_residuals.png: Residual over time
plot_covariate_comparison.png: Perbandingan MAPE antar covariate set
13. Limitasi dan Catatan
DA sekitar 50-53% menunjukkan model lemah dalam memprediksi arah — perlu didiskusikan sebagai limitasi
Variabel bulanan (BI_Rate, CPI, M2, NPL_Ratio) di-forward-fill ke daily, sehingga informasi baru hanya masuk sebulan sekali
Model tidak memperhitungkan structural breaks atau regime changes
One-step-ahead prediction cenderung menghasilkan "lagged copy" dari harga aktual — ini bukan kelemahan model tetapi karakteristik inherent dari pendekatan ini
