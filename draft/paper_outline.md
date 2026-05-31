# Paper Writing Guide — Claude Project Instruction

**Title:** Multi-Horizon Jakarta Composite Index Forecasting Using Tree-Based Ensemble and Bayesian Hyperparameter Optimization

**Target:** Jurnal ML/Computational Finance (Scopus Q2/Q3)  
**Length:** ~6.000–8.000 words  
**Language:** English (Academic)

---

## FOCUS & SCOPE

**Yang DIANGKAT:**
- Multi-horizon prediction (H1=next-day, H5=1 week, H20=1 month)
- 4 tree-based ensemble models: Random Forest, Extra Trees, XGBoost, LightGBM
- 6 covariate group configurations (Baseline, Screening1, Screening2, All_Commodity_STI, All_Macro_no_UST, All_Covariates)
- Bayesian hyperparameter optimization (Optuna/TPE) vs default parameters
- Evaluation: MAPE, DA (Directional Accuracy), MASE

**Yang TIDAK DIANGKAT:**
- Detail proses covariate screening (cukup sebutkan bahwa groups dipilih berdasarkan systematic screening)
- SHAP analysis
- Penjelasan teknis Darts library secara mendalam

---

## KONTEKS EKSPERIMEN (untuk referensi saat menulis)

### Data
- Target: IHSG (Jakarta Composite Index) daily, Januari 2015 – Januari 2025
- 2.443 business days
- 15 covariate variables: 7 macro (BI Rate, CPI, M2, NPL Ratio, USD/IDR, GDP), 7 commodity (Coal, Copper, Nickel, Silver, Tin, Gold, WTI), 1 regional (STI)
- Transformasi: log-difference untuk level variables, first-difference untuk rate variables
- Split: 80% training (Jan 2015 – ~Jan 2023), 20% test (~Jan 2023 – Jan 2025)

### Models
| Model | Type | Key Parameters (Optuna) |
|---|---|---|
| Random Forest | Bagging | n_est=800, max_depth=3, max_features=0.7, max_samples=0.8 |
| Extra Trees | Bagging (random splits) | n_est=200, max_depth=5, max_features=sqrt |
| XGBoost | Gradient Boosting | n_est=600, max_depth=11, lr=0.029, subsample=0.826 |
| LightGBM | Light Gradient Boosting | n_est=600, max_depth=10, lr=0.012, num_leaves=83 |

### Covariate Groups
| Group | Variables | N |
|---|---|---|
| Baseline | — (IHSG lags only) | 0 |
| Screening1 | Silver, WTI, Gold, STI, Coal, Tin, NPL_Ratio | 7 |
| Screening2 | Screening1 + CPI, USDIDR, Nickel | 10 |
| All_Commodity_STI | All 7 commodities + STI | 8 |
| All_Macro | BI Rate, CPI, M2, NPL Ratio, USD/IDR, GDP | 6 |
| All_Covariates | All 14 variables | 14 |

### Optimization
- Optuna with TPE sampler, 30 trials per model
- Objective: minimize avg MAPE across all window × horizon combinations
- Windows: 20 bd (≈1 month), 120 bd (≈6 months)
- Horizons: H1, H5, H20

### Key Results (Update setelah semua eksperimen selesai)
- Overall Optuna improvement vs default: **+6.4% MAPE reduction**
- Best model per algorithm (fill setelah 02b selesai): ...
- Best covariate group overall: ...
- MAPE range: H1 ~0.5%, H5 ~0.85%, H20 ~1.4%

---

## 1. INTRODUCTION

**Tujuan section:** Establish why this paper matters. 3–4 paragraf.

**Paragraph 1 — Motivasi:**
- JCI (Jakarta Composite Index) adalah indikator utama ekonomi Indonesia, pasar berkembang terbesar di Asia Tenggara
- Prediksi harga saham penting untuk investor, fund manager, regulator
- Tantangan: volatile, non-linear, dipengaruhi banyak faktor makro dan komoditas

**Paragraph 2 — Gap dalam literatur:**
- Sebagian besar studi hanya fokus pada next-day prediction (single horizon)
- Studi di pasar Indonesia masih terbatas dibanding AS/Eropa
- Belum banyak yang membandingkan pengaruh berbagai kelompok covariate secara sistematis
- Hyperparameter tuning sering diabaikan atau menggunakan grid search yang tidak efisien

**Paragraph 3 — Kontribusi paper:**
1. Multi-horizon evaluation framework (H1, H5, H20) untuk JCI
2. Systematic comparison of 6 covariate group configurations across 4 tree-based models
3. Bayesian hyperparameter optimization (Optuna/TPE) vs default parameters — quantified improvement
4. Comprehensive evaluation: MAPE, Directional Accuracy, MASE

**Paragraph 4 — Paper structure:**
Section 2: Literature Review. Section 3: Methodology. Section 4: Results and Discussion. Section 5: Conclusion.

---

## 2. LITERATURE REVIEW

**Tujuan section:** Posisikan paper ini dalam konteks penelitian yang ada. 3 subsection.

### 2.1 Machine Learning for Stock Market Prediction
- Review singkat ML approaches: linear models → SVM → ensemble → deep learning
- Mengapa tree-based ensemble populer untuk financial time series:
  - Menangani non-linearity dan interaksi variabel
  - Robust terhadap outlier
  - Tidak memerlukan feature scaling yang ketat
  - Interpretable (feature importance)
- Referensi studi serupa: RF/XGB/LGBM untuk prediksi saham Asia

### 2.2 Covariate Variables in Stock Market Prediction
- Macro variables (interest rate, inflation, money supply) → fundamental analysis
- Commodity prices → khusus relevan untuk Indonesia sebagai commodity exporter
- Regional index (STI) → spillover effect dari pasar tetangga
- Pentingnya memilih covariate yang tepat: curse of dimensionality, overfitting risk

### 2.3 Hyperparameter Optimization
- Grid Search: exhaustive, computationally expensive
- Random Search: lebih efisien, tidak guaranteed optimal
- Bayesian Optimization (TPE/Optuna): sequential, builds probabilistic model → lebih efisien
- Review studi yang menggunakan Bayesian optimization untuk financial ML
- Gap: belum banyak yang quantify improvement Bayesian vs default di multi-horizon setting

---

## 3. METHODOLOGY

**Tujuan section:** Explain the experimental setup clearly dan reproducibly. 5 subsection.

### 3.1 Data Description
- JCI daily price data: period, source (Bloomberg), n observations
- 15 covariate variables: deskripsi singkat tiap kategori (macro/commodity/regional)
- Tabel: Variable, Category, Frequency, Transformation

**Tabel yang dibutuhkan:**
| Variable | Category | Freq | Transformation |
|---|---|---|---|
| IHSG | Target | Daily | log-diff |
| BI Rate | Macro | Monthly | first-diff |
| CPI | Macro | Monthly | first-diff |
| ... | | | |

- Stationarity: ADF test results → all variables non-stationary at level, stationary after transformation
- Train/test split: 80/20 temporal, no random shuffling (preserve time order)

### 3.2 Model Specifications
- Brief description of each model (1–2 kalimat per model)
- **Random Forest:** bagging ensemble, parallel training
- **Extra Trees:** bagging with random threshold splits (bootstrap=False)
- **XGBoost:** sequential gradient boosting, regularization L1/L2
- **LightGBM:** leaf-wise gradient boosting, faster than XGBoost
- Lag features: window W ∈ {20, 120} business days → each model uses W lags of target + W lags of each covariate
- Forecast horizons: H ∈ {1, 5, 20} business days

### 3.3 Covariate Group Configurations
- Explain 6 groups (tabel di atas)
- Sebutkan bahwa groups dipilih berdasarkan preliminary screening analysis (tidak perlu detail)
- Total experiments: 6 groups × 4 models × 2 windows × 3 horizons = **144 experiments**

### 3.4 Bayesian Hyperparameter Optimization
- Framework: Optuna dengan TPE (Tree-structured Parzen Estimator) sampler
- Objective function: minimize average MAPE across all window × horizon combinations
- 30 trials per model (4 × 30 = 120 total trials)
- **Tabel search space:**

| Model | Parameter | Search Space |
|---|---|---|
| All | n_estimators | [100, 1000] step 100 |
| RF/ET | max_depth | [3, 20] |
| RF | max_features | {sqrt, log2, 0.3, 0.5, 0.7} |
| RF | max_samples | [0.5, 0.9] |
| RF/ET | min_samples_split | [2, 20] |
| RF/ET | min_samples_leaf | [1, 10] |
| XGB/LGB | learning_rate | [0.01, 0.3] log scale |
| XGB | max_depth | [3, 15] |
| XGB | subsample | [0.5, 1.0] |
| XGB | colsample_bytree | [0.3, 1.0] |
| XGB | reg_alpha, reg_lambda | [1e-8, 10] log scale |
| LGB | num_leaves | [15, 127] |
| LGB | min_child_samples | [5, 50] |

- Bandingkan dengan default: tabel default vs Optuna result (dari notebook 02b)

### 3.5 Evaluation Metrics
- **MAPE** (Mean Absolute Percentage Error): primary metric, dihitung pada level harga JCI
  - Formula: MAPE = (1/n) Σ |y_t - ŷ_t| / y_t × 100%
- **DA** (Directional Accuracy): % prediksi yang benar arah naik/turun
  - Formula: DA = (1/n) Σ 1[sign(ŷ_t - y_{t-1}) = sign(y_t - y_{t-1})] × 100%
  - Baseline: 50% (random guess)
- **MASE** (Mean Absolute Scaled Error): scale-free metric vs naive forecast
  - Formula: MASE = MAE / MAE_naive, dimana MAE_naive = mean(|y_t - y_{t-1}|) on training set

---

## 4. RESULTS AND DISCUSSION

**Tujuan section:** Present findings clearly, discuss implications. 4 subsection.

### 4.1 Hyperparameter Optimization Results

**Tabel: Default vs Optuna Parameters per Model** (dari notebook 02b, cell parameter comparison)

**Tabel: MAPE Improvement Optuna vs Default per Model**
| Model | Default MAPE (%) | Optuna MAPE (%) | Improvement (%) |
|---|---|---|---|
| Random Forest | ... | ... | ... |
| Extra Trees | ... | ... | ... |
| XGBoost | ... | ... | +9.87% |
| LightGBM | ... | ... | +13.33% |
| **Overall** | **0.9739** | **0.9117** | **+6.38%** |

**Discussion points:**
- XGBoost dan LightGBM mendapat improvement terbesar dari optimization → model ini lebih sensitif terhadap hyperparameter
- RF dan ET sudah cukup baik dengan default → robust to hyperparameter choice
- Bayesian optimization efektif: 30 trials menghasilkan significant improvement

**Tabel: Per-Scenario MAPE Comparison** (Default vs Optuna, per W × H)

### 4.2 Impact of Covariate Groups

**Tabel: Avg MAPE per Group (avg semua model × skenario)**

**Tabel: Avg MAPE per Group × Model** (pivot table, dari notebook 02a/02b)

**Discussion points:**
- Kelompok mana yang terbaik dan mengapa? (commodity-based groups)
- Apakah lebih banyak covariate selalu lebih baik? (All_Covariates vs Screening1)
- Implikasi ekonomi: commodity prices (Silver, WTI, Gold) lebih informatif untuk JCI
  - Indonesia sebagai net commodity exporter → harga komoditas mempengaruhi earnings perusahaan
- Macro variables: All_Macro_no_UST kurang efektif → macro sudah ter-reflect dalam price?

### 4.3 Multi-Horizon Analysis

**Tabel: MAPE dan DA per Horizon (avg semua model × group)**
| Horizon | MAPE (%) | DA (%) | MASE |
|---|---|---|---|
| H1 (1 day) | ~0.50 | ~52–53 | ~0.97 |
| H5 (1 week) | ~0.85 | ~48–49 | ~1.58 |
| H20 (1 month) | ~1.40 | ~49–50 | ~2.60 |

**Discussion points:**
- MAPE meningkat seiring horizon → wajar, uncertainty akumulasi
- DA mendekati 50% (random) di semua horizon → pasar JCI mendekati semi-strong efficiency untuk directional prediction
- DA H1 sedikit di atas 50% → ada predictable pattern jangka sangat pendek
- Window W20 vs W120: mana yang lebih baik per horizon? (W20 untuk H1, W120 untuk H20?)

**Tabel: Best Configuration per Scenario (Window × Horizon)**
Top 3 dari tiap skenario (dari notebook 02a bagian bawah)

### 4.4 Model Comparison

**Tabel: Best MAPE per Model (best config)**

**Discussion points:**
- XGBoost dan LightGBM cenderung unggul setelah optimization
- Extra Trees competitif meski tuning lebih sedikit parameter
- Ensemble diversity: kenapa model dengan tipe berbeda (bagging vs boosting) menunjukkan perbedaan?

---

## 5. CONCLUSION

**Tujuan section:** Summarize, answer research questions, discuss implications. 1 halaman.

**Paragraph 1 — Main findings:**
1. Bayesian optimization (Optuna) menghasilkan improvement MAPE rata-rata 6.4% dibanding default, terutama signifikan untuk boosting models (XGBoost +9.9%, LightGBM +13.3%)
2. Covariate groups berbasis komoditas (All_Commodity_STI, Screening1) konsisten mengungguli baseline di semua horizon
3. Akurasi prediksi menurun seiring horizon (H1 MAPE ~0.5% vs H20 MAPE ~1.4%), mencerminkan sifat random walk pasar jangka panjang

**Paragraph 2 — Practical implications:**
- Untuk short-term trading (H1): model dengan Screening1/All_Commodity_STI + W20 paling efektif
- Untuk portfolio rebalancing (H20): model masih memberikan MAPE kompetitif meski DA mendekati random
- Commodity prices (Silver, WTI, Gold) terbukti sebagai leading indicators untuk JCI

**Paragraph 3 — Limitations:**
- Single market study: hasil mungkin tidak generalizable ke pasar lain
- GDP potential look-ahead bias (quarterly release lag tidak di-offset)
- DA ~50% menunjukkan keterbatasan model untuk directional prediction

**Paragraph 4 — Future work:**
- Deep learning comparison (LSTM, Transformer)
- Market regime detection (bull/bear) sebagai additional feature
- Extend ke saham individual (not just index)
- Walk-forward validation dengan lebih banyak folds

---

## FIGURES & TABLES CHECKLIST

### Figures
- [ ] Figure 1: Research framework flowchart
- [ ] Figure 2: MAPE per horizon (bar/line chart)
- [ ] Figure 3: Default vs Optuna MAPE per model (grouped bar chart)
- [ ] Figure 4: MAPE per covariate group × model (heatmap dari notebook 02a)
- [ ] Figure 5: Actual vs predicted JCI price — top 3 model (line plot dari notebook 04)

### Tables
- [ ] Table 1: Variable description (name, category, frequency, transformation)
- [ ] Table 2: Hyperparameter search space
- [ ] Table 3: Default vs Optuna parameters + MAPE comparison
- [ ] Table 4: Avg MAPE per covariate group (avg all models × scenarios)
- [ ] Table 5: Best MAPE per Window × Horizon scenario
- [ ] Table 6: Model comparison — best config per algorithm

---

## WRITING NOTES FOR CLAUDE

Saat membantu menulis section tertentu:
1. **Gunakan data dari tabel di atas** — jangan karang angka
2. **Tone:** academic, objective, tidak over-claim
3. **Tenses:** Present untuk methodology, Past untuk results
4. **Avoid:** "this paper proves", gunakan "the results suggest/indicate"
5. **Untuk formula:** gunakan LaTeX notation
6. **Setiap claim harus didukung** angka dari tabel experiment
7. **JCI atau Jakarta Composite Index** (bukan IHSG dalam text English)
