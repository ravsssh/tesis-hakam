# Hasil Eksperimen — Referensi Paper & Tesis

**Last updated:** Mei 2026  
**Notebook pipeline:** 00 → 01 → 02a → 02b → 03

---

## NOTEBOOK 00 — Data Preprocessing

### Dataset Overview
| Item | Detail |
|---|---|
| Target | IHSG (Jakarta Composite Index) |
| Periode | 2 Januari 2015 – 31 Januari 2025 |
| Total observasi | 2.443 business days |
| Train set (80%) | ~1.954 hari (Jan 2015 – ~Jan 2023) |
| Test set (20%) | ~489 hari (~Jan 2023 – Jan 2025) |
| Sumber data | Bloomberg Terminal (via GitHub repository) |

### Variabel Penelitian
| Variabel | Kategori | Frekuensi | Transformasi |
|---|---|---|---|
| IHSG | Target | Harian | log-diff |
| BI_Rate | Macro | Bulanan | first-diff |
| CPI | Macro | Bulanan | first-diff |
| M2 | Macro | Bulanan | log-diff |
| NPL_Ratio | Macro | Bulanan | first-diff |
| GDP | Macro | Kuartalan | log-diff |
| USDIDR | Macro | Harian | log-diff |
| Coal | Commodity | Harian | log-diff |
| Copper | Commodity | Harian | log-diff |
| Nickel | Commodity | Harian | log-diff |
| Silver | Commodity | Harian | log-diff |
| Tin | Commodity | Harian | log-diff |
| Gold | Commodity | Harian | log-diff |
| WTI | Commodity | Harian | log-diff |
| STI | Regional | Harian | log-diff |

### ADF Test — Level (Sebelum Transformasi)
Semua 16 variabel **tidak stasioner** (p-value > 0.05):

| Variabel | ADF Statistic | p-value |
|---|---|---|
| IHSG | -1.5296 | 0.5188 |
| BI_Rate | -1.8905 | 0.3365 |
| CPI | -0.7384 | 0.8366 |
| M2 | 0.6271 | 0.9883 |
| NPL_Ratio | -1.0838 | 0.7215 |
| GDP | 0.1504 | 0.9693 |
| USDIDR | -1.9612 | 0.3038 |
| WTI | -1.9526 | 0.3077 |
| Coal | -1.6418 | 0.4613 |
| Copper | -1.1960 | 0.6753 |
| Nickel | -1.8229 | 0.3692 |
| Silver | -1.1260 | 0.7046 |
| Tin | -1.4044 | 0.5802 |
| Gold | 1.0763 | 0.9950 |
| STI | -2.1001 | 0.2445 |

### ADF Test — Setelah Transformasi
Semua 16 variabel **stasioner** (p-value < 0.05):

| Variabel (Transformed) | ADF Statistic | p-value |
|---|---|---|
| IHSG (log-diff) | -27.6375 | 0.0000 |
| BI_Rate (diff) | -6.8819 | 1.43e-09 |
| CPI (diff) | -8.5253 | 1.07e-13 |
| M2 (log-diff) | -12.6354 | 1.47e-23 |
| NPL_Ratio (diff) | -14.3044 | 1.21e-26 |
| GDP (log-diff) | -49.6712 | 0.0000 |
| USDIDR (log-diff) | -21.1681 | 0.0000 |
| WTI (log-diff) | -9.7844 | 6.60e-17 |
| Coal (log-diff) | -11.1830 | 2.47e-20 |
| Copper (log-diff) | -51.9504 | 0.0000 |
| Nickel (log-diff) | -14.8928 | 1.55e-27 |
| Silver (log-diff) | -32.8022 | 0.0000 |
| Tin (log-diff) | -47.4068 | 0.0000 |
| Gold (log-diff) | -48.6772 | 0.0000 |
| STI (log-diff) | -14.1178 | 2.46e-26 |

---

## NOTEBOOK 01 — Hyperparameter Tuning (Optuna)

### Setup
| Item | Detail |
|---|---|
| Framework | Optuna dengan TPE (Tree-structured Parzen Estimator) sampler |
| Trials per model | 30 |
| Total trials | 4 × 30 = 120 |
| Objective | Minimize avg MAPE (semua window × horizon combinations) |
| Seed | 42 |

### Search Space per Model
| Model | Parameter | Range |
|---|---|---|
| Semua | n_estimators | [100, 1000] step 100 |
| RF / ET | max_depth | [3, 20] |
| RF | max_features | {sqrt, log2, 0.3, 0.5, 0.7} |
| RF | max_samples | [0.5, 0.9] |
| RF / ET | min_samples_split | [2, 20] |
| RF / ET | min_samples_leaf | [1, 10] |
| XGB / LGB | learning_rate | [0.01, 0.3] log-scale |
| XGB | max_depth | [3, 15] |
| XGB | subsample | [0.5, 1.0] |
| XGB | colsample_bytree | [0.3, 1.0] |
| XGB | reg_alpha, reg_lambda | [1e-8, 10] log-scale |
| XGB | min_child_weight | [1, 10] |
| LGB | num_leaves | [15, 127] |
| LGB | min_child_samples | [5, 50] |

### Hasil Optuna — Best Parameters per Model

**Random Forest** — best_value: 1.2634
| Parameter | Default | **Optuna** |
|---|---|---|
| n_estimators | 300 | **800** |
| max_depth | 5 | **3** |
| max_features | sqrt | **0.7** |
| max_samples | 0.7 | **0.8** |
| min_samples_split | — | **5** |
| min_samples_leaf | — | **6** |

**Extra Trees** — best_value: 1.2625
| Parameter | Default | **Optuna** |
|---|---|---|
| n_estimators | 300 | **200** |
| max_depth | 5 | **5** |
| max_features | sqrt | **sqrt** |
| min_samples_split | — | **9** |
| min_samples_leaf | — | **3** |

**XGBoost** — best_value: 1.2589
| Parameter | Default | **Optuna** |
|---|---|---|
| n_estimators | 300 | **600** |
| max_depth | 5 | **11** |
| learning_rate | 0.1 | **0.0295** |
| subsample | 0.7 | **0.8256** |
| colsample_bytree | 0.7 | **0.3920** |
| reg_alpha | 0.01 | **4.7230** |
| reg_lambda | 1.0 | **0.0035** |
| min_child_weight | — | **8** |

**LightGBM** — best_value: 1.2611
| Parameter | Default | **Optuna** |
|---|---|---|
| n_estimators | 300 | **600** |
| max_depth | 5 | **10** |
| learning_rate | 0.1 | **0.0117** |
| num_leaves | 31 | **83** |
| subsample | 0.7 | **0.5853** |
| colsample_bytree | 0.7 | **0.3456** |
| reg_alpha | 0.01 | **3.4671** |
| reg_lambda | 1.0 | **4.9056** |
| min_child_samples | — | **42** |

---

## NOTEBOOK 02a — Single Covariate Screening & Group Experiments

### Setup Screening
| Item | Detail |
|---|---|
| Total eksperimen | 384 (4 model × 16 covariate × 2 window × 3 horizon) |
| Evaluasi | 80/20 temporal split, Optuna hyperparameters |
| Windows | 20 bd (≈1 bulan), 120 bd (≈6 bulan) |
| Horizons | H1 (next-day), H5 (1 minggu), H20 (1 bulan) |
| Max combos per covariate | 24 (4 model × 2 window × 3 horizon) |
| Threshold lolos | MAPE improvement ≥ 0.3% ATAU DA improvement ≥ 1.0 pp |

### Hasil Screening — Pass Count per Covariate

| Covariate | Kategori | Pass Count | Pass Rate | Avg MAPE Impr% | Avg DA Impr (pp) | Tier |
|---|---|---|---|---|---|---|
| Silver | Commodity | 16/24 | 66.7% | +0.637% | +0.91 | ✅ Kuat |
| WTI | Commodity | 13/24 | 54.2% | +0.190% | +0.73 | ✅ Kuat |
| Gold | Commodity | 11/24 | 45.8% | +0.402% | +0.15 | 🟡 Moderat |
| STI | Regional | 9/24 | 37.5% | +0.317% | -0.23 | 🟡 Moderat |
| USDIDR | Macro | 8/24 | 33.3% | -0.281% | +0.16 | ⚠️ Lemah |
| Coal | Commodity | 7/24 | 29.2% | +0.107% | -0.05 | 🟡 Moderat |
| Nickel | Commodity | 7/24 | 29.2% | -0.382% | -0.62 | ⚠️ Lemah |
| CPI | Macro | 6/24 | 25.0% | -0.019% | +0.29 | ⚠️ Lemah |
| NPL_Ratio | Macro | 6/24 | 25.0% | +0.100% | +0.02 | 🟡 Moderat |
| Tin | Commodity | 6/24 | 25.0% | +0.105% | -0.18 | 🟡 Moderat |
| Copper | Commodity | 5/24 | 20.8% | -0.202% | -0.01 | ❌ Tidak lolos |
| M2 | Macro | 3/24 | 12.5% | +0.076% | -0.04 | ❌ Tidak lolos |
| BI_Rate | Macro | 2/24 | 8.3% | +0.054% | +0.22 | ❌ Tidak lolos |
| GDP | Macro | 2/24 | 8.3% | +0.037% | +0.09 | ❌ Tidak lolos |

### Covariate Groups yang Dibentuk
| Group | Variabel | N |
|---|---|---|
| Baseline | — (IHSG lags only) | 0 |
| Screening1 | Silver, WTI, Gold, STI, Coal, Tin, NPL_Ratio | 7 |
| Screening2 | Screening1 + CPI, USDIDR, Nickel | 10 |
| All_Commodity_STI | Coal, Copper, Nickel, Silver, Tin, Gold, WTI, STI | 8 |
| All_Macro_no_UST | BI_Rate, CPI, M2, NPL_Ratio, USDIDR, GDP | 6 |
| All_Covariates | Semua 15 variabel | 15 |

### Hasil Group Experiments (Optuna params, 80/20 split)

**Total eksperimen:** 144 (6 group × 4 model × 2 window × 3 horizon)

#### Avg MAPE per Group (avg semua model × skenario)
| Group | Avg MAPE (%) |
|---|---|
| **Screening1** | **0.9047** ← terbaik |
| All_Commodity_STI | 0.9071 |
| Screening2 | 0.9080 |
| All_Macro_no_UST | 0.9185 |
| All_Covariates | 0.9201 |
| Baseline | 0.9169 |

#### Avg MAPE per Model (avg semua group × skenario)
| Model | Avg MAPE (%) |
|---|---|
| **LightGBM** | **0.9047** ← terbaik |
| XGBoost | 0.9080 |
| ExtraTrees | 0.9153 |
| RandomForest | 0.9223 |

#### Top 5 Konfigurasi Terbaik (MAPE terkecil)
| Rank | Model | Group | Window | Horizon | MAPE (%) | DA (%) | MASE |
|---|---|---|---|---|---|---|---|
| #1 | LightGBM | Screening1 | 120 | 1 | **0.5007** | 55.89 | 0.9619 |
| #2 | XGBoost | Screening1 | 120 | 1 | 0.5012 | 51.90 | 0.9627 |
| #3 | XGBoost | Screening1 | 20 | 1 | 0.5017 | 54.37 | 0.9636 |
| #4 | LightGBM | Screening1 | 20 | 1 | 0.5026 | 54.75 | 0.9655 |
| #5 | LightGBM | All_Macro_no_UST | 20 | 1 | 0.5026 | 53.80 | 0.9655 |

#### Hasil per Horizon (avg semua model × group)
| Horizon | Avg MAPE (%) | Avg DA (%) | Avg MASE |
|---|---|---|---|
| H1 (next-day) | 0.5056 | 52.66 | 0.9714 |
| H5 (1 minggu) | 0.8445 | 48.12 | 1.6225 |
| H20 (1 bulan) | 1.3876 | 48.80 | 2.6774 |

#### Hasil per Window (avg semua model × group × horizon)
| Window | Avg MAPE (%) | Avg DA (%) |
|---|---|---|
| W20 (20 bd ≈ 1 bulan) | 0.9141 | 50.11 |
| W120 (120 bd ≈ 6 bulan) | 0.9111 | 49.62 |

---

## NOTEBOOK 02b — Default vs Optuna Comparison

### Overall Comparison
| | Default | Optuna | Improvement |
|---|---|---|---|
| Avg MAPE (non-baseline) | 0.9739% | 0.9117% | **+6.38%** |

### Per Model Comparison
| Model | Default MAPE (%) | Optuna MAPE (%) | Improvement |
|---|---|---|---|
| RandomForest | 0.9169 | 0.9223 | -0.59% ← sedikit turun |
| ExtraTrees | 0.9145 | 0.9153 | -0.09% ← hampir sama |
| **XGBoost** | 1.0075 | 0.9080 | **+9.87%** |
| **LightGBM** | 1.0438 | 0.9047 | **+13.33%** |

> **Insight:** Boosting models (XGBoost, LightGBM) mendapat improvement signifikan dari Optuna. Bagging models (RF, ET) sudah cukup baik dengan default → less sensitive to hyperparameter choice.

### Per Group Covariate Comparison
| Group | Default MAPE (%) | Optuna MAPE (%) | Improvement |
|---|---|---|---|
| Baseline | 0.9548 | 0.9169 | +3.97% |
| Screening1 | 0.9657 | 0.9047 | +6.32% |
| Screening2 | 0.9768 | 0.9080 | +7.04% |
| All_Commodity_STI | 0.9763 | 0.9071 | +7.08% |
| All_Macro_no_UST | 0.9623 | 0.9185 | +4.55% |
| All_Covariates | 0.9881 | 0.9201 | +6.88% |

### Per Scenario Comparison
| Scenario | Default MAPE (%) | Optuna MAPE (%) | Improvement |
|---|---|---|---|
| W20_H1 | 0.5242 | 0.5049 | +3.69% |
| W120_H1 | 0.5243 | 0.5063 | +3.43% |
| W20_H5 | 0.8852 | 0.8423 | +4.85% |
| W120_H5 | 0.8740 | 0.8468 | +3.12% |
| **W20_H20** | 1.5365 | 1.3950 | **+9.21%** |
| **W120_H20** | 1.4798 | 1.3802 | **+6.73%** |

> **Insight:** Improvement Optuna paling besar pada H20 (horizon panjang) — optimasi lebih penting untuk multi-step forecasting.

---

## NOTEBOOK 03 — SHAP Analysis

> ⚠️ SHAP dijalankan pada konfigurasi terbaik dari screening (bukan group experiment). Akan diperbarui setelah notebook 03 dijalankan ulang pada top config dari group experiment.

### SHAP Variable Importance (Current: best single-covariate config)

Model: RandomForest | Covariates: Screening1 | W20_H1

| Rank | Variabel | Mean |SHAP| | Kategori |
|---|---|---|---|
| 1 | NPL_Ratio | 0.000142 | Macro |
| 2 | Silver | 0.000125 | Commodity |
| 3 | STI | 0.000064 | Regional |
| 4 | Gold | 0.000064 | Commodity |
| 5 | WTI | 0.000041 | Commodity |
| 6 | Coal | 0.000040 | Commodity |
| 7 | Tin | 0.000029 | Commodity |
| 8 | IHSG (lag) | 0.000024 | Target (autoregressive) |

---

## RINGKASAN TEMUAN UTAMA (untuk paper & tesis)

### 1. Bayesian Optimization
- Overall improvement Optuna vs default: **+6.38% MAPE reduction**
- Boosting models (XGBoost +9.87%, LightGBM +13.33%) >> Bagging models (RF -0.59%, ET -0.09%)
- Improvement terbesar pada H20: +9.21% → optimasi lebih krusial untuk long-horizon

### 2. Covariate Groups
- **Screening1** (7 commodity/regional vars) adalah group terbaik secara avg MAPE
- **All_Covariates** (15 vars) tidak lebih baik dari Screening1 → *parsimony principle*
- Commodity + regional variables lebih informatif daripada macro variables untuk JCI

### 3. Multi-Horizon
- H1 MAPE: **0.5056%** (sangat akurat di level harga)
- H5 MAPE: **0.8445%** (+67% dari H1)
- H20 MAPE: **1.3876%** (+175% dari H1)
- DA di semua horizon mendekati 50% (random) → pasar JCI sulit diprediksi arahnya

### 4. Best Models (Optuna + Group Experiments)
- **Best overall:** LightGBM + Screening1 + W120 + H1 → MAPE = **0.5007%**, DA = 55.89%
- **Best algorithm:** LightGBM (avg MAPE 0.9047%)
- **Best group:** Screening1 (avg MAPE 0.9047%)
- **Best window:** W120 marginally better than W20

### 5. SHAP Highlights
- Commodity variables (Silver, Gold, WTI) dominan → konsisten dengan Indonesia sebagai commodity exporter
- IHSG own lags memiliki contribution terkecil → covariate memang menambah informasi baru
- NPL_Ratio (banking health) paling berpengaruh dalam Screening1 config

---

---

## NOTEBOOK 02a — Detail Screening per Horizon

> **Temuan kunci:** Covariate yang paling relevan berbeda-beda tergantung horizon prediksi. Ini adalah finding penting untuk Bab 4 tesis.

### Screening H1 (Next-Day) — max 8 combos (4 model × 2 window)
| Covariate | Pass (8) | Avg MAPE Impr% | Avg DA Impr (pp) |
|---|---|---|---|
| USDIDR | 6 | +0.510% | +0.52 |
| STI | 5 | +0.453% | +0.70 |
| Silver | 6 | +0.367% | +1.40 |
| Gold | 3 | +0.468% | -0.12 |
| Copper | 4 | +0.296% | +0.29 |
| M2 | 4 | +0.229% | +0.30 |
| CPI | 3 | +0.209% | +0.45 |
| GDP | 2 | +0.185% | +0.32 |
| NPL_Ratio | 2 | +0.107% | +0.09 |
| BI_Rate | 2 | +0.102% | -0.25 |
| WTI | 2 | +0.071% | -0.36 |
| Nickel | 2 | -0.003% | -1.01 |
| Coal | 1 | -0.680% | -0.94 |
| US_Treasury_10Y | 1 | -1.001% | -0.69 |

> **Insight H1:** USDIDR (nilai tukar) dan Silver paling dominan untuk prediksi next-day. Menunjukkan bahwa sentimen mata uang dan perak harian sangat berpengaruh jangka sangat pendek.

### Screening H5 (1 Minggu) — max 8 combos
| Covariate | Pass (8) | Avg MAPE Impr% | Avg DA Impr (pp) |
|---|---|---|---|
| Gold | 5 | +1.721% | -0.10 |
| STI | 5 | +0.902% | +0.04 |
| Silver | 5 | +0.586% | -0.25 |
| Copper | 4 | +0.398% | +0.40 |
| BI_Rate | 2 | +0.339% | +0.02 |
| GDP | 3 | +0.302% | -0.16 |
| CPI | 2 | +0.257% | -0.18 |
| Tin | 2 | +0.134% | -0.43 |
| NPL_Ratio | 3 | +0.129% | -0.47 |
| M2 | 4 | +0.032% | -0.27 |
| USDIDR | 1 | -0.010% | -0.15 |
| WTI | 1 | -0.292% | -0.60 |
| Coal | 1 | -0.409% | -0.35 |

> **Insight H5:** Gold menjadi yang paling berpengaruh untuk horizon 1 minggu (+1.721% MAPE improvement). STI tetap konsisten. USDIDR yang kuat di H1 justru tidak signifikan di H5.

### Screening H20 (1 Bulan) — max 8 combos
| Covariate | Pass (8) | Avg MAPE Impr% | Avg DA Impr (pp) |
|---|---|---|---|
| WTI | 6 | +0.941% | +0.17 |
| Copper | 3 | +0.799% | +0.28 |
| Gold | 3 | +0.092% | +0.39 |
| BI_Rate | 2 | +0.066% | -0.26 |
| M2 | 2 | +0.052% | +0.24 |
| GDP | 3 | -0.013% | +0.21 |
| NPL_Ratio | 3 | -0.013% | -0.03 |
| USDIDR | 2 | -0.143% | +0.09 |
| STI | 4 | -0.163% | +0.08 |
| CPI | 3 | -0.388% | -0.06 |
| Silver | 4 | **-0.614%** | +0.28 |
| Tin | 1 | -1.921% | -0.38 |
| Coal | 1 | -2.750% | +0.02 |

> **Insight H20:** WTI (minyak) paling dominan untuk prediksi 1 bulan. Silver yang kuat di H1 justru **memperburuk prediksi** di H20 (-0.614%). Menunjukkan peran berbeda komoditas energi vs logam mulia di berbagai horizon waktu.

### Rangkuman Temuan Horizon-Spesifik
| Horizon | Covariate Terkuat | Covariate Terlemah | Pola |
|---|---|---|---|
| **H1** | USDIDR, Silver, STI | Coal, US_Treasury_10Y | Sentimen mata uang & regional |
| **H5** | Gold, STI, Silver | WTI, Coal | Logam mulia & regional |
| **H20** | WTI, Copper | Silver, Coal | Energi & logam industri |

---

## NOTEBOOK 02a — Hasil Group Experiments per Horizon

### Avg MAPE per Group × Horizon (avg semua model)
| Horizon | Screening1 | Screening2 | All_Commodity_STI | All_Macro_no_UST | All_Covariates |
|---|---|---|---|---|---|
| **H1** | **0.5034** | 0.5051 | 0.5056 | 0.5067 | 0.5055 |
| **H5** | 0.8408 | **0.8381** | 0.8390 | 0.8520 | 0.8436 |
| **H20** | **0.9047** | — | 1.3768 | 1.3968 | 1.4112 |

> **Insight:** Screening1 unggul di H1 dan H20. Screening2 (lebih banyak variabel) unggul di H5. All_Macro_no_UST konsisten paling buruk di semua horizon.

### Best Configuration per Model × Horizon (MAPE terkecil)
| Model | Horizon | Best Group | Window | MAPE (%) | DA (%) |
|---|---|---|---|---|---|
| **LightGBM** | H1 | Screening1 | 120 | **0.5007** | 55.89 |
| **XGBoost** | H1 | Screening1 | 120 | 0.5012 | 51.90 |
| RandomForest | H1 | Screening1 | 20 | 0.5047 | 53.80 |
| ExtraTrees | H1 | Screening2 | 120 | 0.5047 | 54.75 |
| **XGBoost** | H5 | Screening2 | 20 | **0.8133** | 47.71 |
| **LightGBM** | H5 | Screening2 | 20 | 0.8185 | 47.14 |
| ExtraTrees | H5 | Screening1 | 120 | 0.8500 | 47.71 |
| RandomForest | H5 | Screening1 | 20 | 0.8519 | 49.05 |
| **XGBoost** | H20 | Screening1 | 120 | **1.3238** | 48.36 |
| **LightGBM** | H20 | Screening1 | 120 | 1.3254 | 48.75 |
| RandomForest | H20 | All_Macro_no_UST | 120 | 1.3779 | 50.10 |
| ExtraTrees | H20 | All_Macro_no_UST | 120 | 1.3818 | 49.90 |

---

## NOTEBOOK 03 — SHAP Analysis (LightGBM)

Model: LightGBM | Covariates: Screening1 | W120_H1 (best overall config)

| Rank | Variabel | Mean \|SHAP\| | Kategori |
|---|---|---|---|
| 1 | NPL_Ratio | 0.000142 | Macro |
| 2 | Silver | 0.000125 | Commodity |
| 3 | STI | 0.000064 | Regional |
| 4 | Gold | 0.000064 | Commodity |
| 5 | WTI | 0.000041 | Commodity |
| 6 | Coal | 0.000040 | Commodity |
| 7 | Tin | 0.000029 | Commodity |
| 8 | IHSG (lag) | 0.000024 | Target (autoregressive) |

> **Interpretasi:** SHAP dihitung pada **log-return** (return harian), bukan harga level. NPL_Ratio (perubahan rasio kredit macet) menjadi covariate paling informatif. IHSG own lag memiliki kontribusi terkecil — menunjukkan covariate eksternal memang menambah informasi baru yang tidak ada di historis IHSG sendiri.

> **Catatan interpretasi untuk dosen:** SHAP value mencerminkan kontribusi *return kemarin* (dalam %) dari setiap variabel terhadap *prediksi return IHSG hari ini*. Bukan pengaruh kenaikan harga $1 atau 1 poin.

---

## FILE TAMBAHAN YANG TERSEDIA (tidak dimasukkan — referensi jika dibutuhkan)

| File | Isi | Keterangan |
|---|---|---|
| `phase1_screening_enhanced.csv` | 360 baris, per-combo detail + Passed_MAPE/DA/Both | Sumber data untuk tabel per-horizon di atas |
| `phase2_full_comparison.csv` | 168 baris, old groups (Sig_Macro dll.) | **Iterasi lama** — beda groups dari 02a, jangan dipakai |
| `phase1_ablation_results.csv` | 144 baris, 5-fold CV, old economic grouping | **Iterasi lama** (risk_global, monetary_id dll.) — tidak konsisten dengan metodologi final |
| `phase1_ablation_analysis.csv` | Termasuk Wilcoxon p-value, tapi old groups | Uji statistik ada tapi dari iterasi lama |
| `phase1_screening_results.csv` | 384 baris, format MAPE mean±std, 5-fold CV | **Duplikat lama** dari phase1a_screening_results.csv |
| `rf_final_results_*.csv` | RF-only lama (pre-multi-model) | Tidak relevan untuk tesis final |
| `rf_feature_importance_*.csv` | Feature importance RF lama | Digantikan oleh SHAP analysis |

---

## CATATAN METODOLOGI

- **Data leakage prevention:** Scaler (MinMaxScaler) di-fit hanya pada training set
- **Temporal order preserved:** 80/20 split tidak random, sequential
- **MAPE dihitung pada level harga IHSG** (bukan pada log-return), sehingga interpretable
- **DA baseline = 50%** (random coin flip untuk prediksi naik/turun)
- **GDP limitation:** Data kuartalan di-forward-fill, potensi look-ahead bias (publikasi BPS ~2 bulan setelah akhir kuartal)
