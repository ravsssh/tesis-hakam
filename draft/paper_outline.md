# Paper Brainstorm & Outline — Prediksi IHSG

---

## Brainstorm Judul

Tiga opsi dengan penekanan berbeda:

**A — Fokus metodologi (paling kuat untuk paper ML):**
> *"Multi-Horizon IHSG Forecasting Using Tree-Based Ensemble with Systematic Covariate Screening and Bayesian Hyperparameter Optimization"*

**B — Fokus ekonomi (lebih menarik untuk jurnal ekonomi/keuangan):**
> *"The Role of Commodity Prices in Predicting Indonesian Stock Market: A Machine Learning Approach with Explainability Analysis"*

**C — Gabungan (balance antara ML dan ekonomi):**
> *"Predicting Indonesian Composite Stock Index: Covariate Selection, Ensemble Learning, and Hyperparameter Optimization"*

Rekomendasi: **A** kalau target jurnal ML/computational finance, **B** kalau target jurnal ekonomi Indonesia.

---

## Result yang Layak Diangkat

### 1. Covariate Screening (kontribusi metodologi)
- Silver, WTI, Gold paling konsisten membantu prediksi (pass rate >45%) — komoditas ekspor utama Indonesia
- US_Treasury_10Y, Nickel, M2 tidak signifikan — menarik untuk diskusi ekonomi
- Screening1 (7 vars) performanya sangat dekat dengan All_Covariates (15 vars) → *parsimony principle*

### 2. Hyperparameter Optimization (kontribusi utama paper)
- Optuna (Bayesian) vs default: improvement MAPE dari hasil notebook 02b
- Perbandingan parameter yang berubah dan magnitude perubahannya
- Ini yang diminta dosen secara eksplisit

### 3. Multi-Horizon Analysis (kontribusi unik)
- H1 (next-day): MAPE ~0.5%, DA ~52%
- H5 (1 minggu): MAPE ~0.85%
- H20 (1 bulan): MAPE ~1.4%
- Insight: akurasi arah (DA) tidak meningkat meski MAPE membesar → pasar semakin random untuk horizon panjang

### 4. Model Comparison
- XGBoost/LightGBM cenderung menang setelah optimasi
- ExtraTrees kompetitif tanpa perlu banyak tuning

### 5. SHAP — Economic Interpretability
- Nickel dan WTI lag-1 paling berpengaruh → Indonesia sebagai net exporter commodity
- Menjawab pertanyaan dosen: "bagaimana pengaruh tiap variabel terhadap IHSG"

---

## Outline Paper

# [JUDUL PAPER]

---

## 1. Introduction
- Background: pentingnya prediksi pasar saham, IHSG sebagai indikator ekonomi Indonesia
- Gap: kebanyakan studi pakai satu model/satu horizon, belum systematic covariate selection
- Contribution:
  1. Systematic single-covariate screening sebelum group experiments
  2. Bayesian hyperparameter optimization (Optuna) vs default
  3. Multi-horizon evaluation (H1, H5, H20)
  4. SHAP untuk economic interpretability
- Research questions:
  - Covariate mana yang signifikan untuk prediksi IHSG?
  - Seberapa besar improvement Optuna vs default?
  - Model mana yang terbaik di tiap horizon?

---

## 2. Literature Review

### 2.1 Stock Market Prediction with Machine Learning
- Kenapa tree-based ensemble (RF, ET, XGB, LGB) populer untuk finansial
- Referensi studi serupa di pasar Asia/Indonesia

### 2.2 Covariate Selection untuk Time Series
- Pentingnya variable selection sebelum modeling
- Metode screening yang umum dipakai

### 2.3 Hyperparameter Optimization
- Grid search vs Random search vs Bayesian (Optuna/TPE)
- Keunggulan Bayesian optimization untuk model kompleks

### 2.4 Model Interpretability (XAI) di Finansial
- SHAP values dan aplikasinya di prediksi finansial
- Relevansi untuk investor dan regulator

---

## 3. Methodology

### 3.1 Data
- IHSG daily, periode 2015–2025
- 15 covariate: 7 macro, 7 commodity, 1 regional
- Transformasi: log-diff (level vars), first-diff (rate vars)
- Train/test split: 80/20 temporal

### 3.2 Model Architecture
- 4 tree-based models: RF, ET, XGB, LGB
- Darts library: lag features (window 20 dan 120 business days)
- Output: multi-horizon (H1, H5, H20)

### 3.3 Phase 1 — Covariate Screening
- 384 eksperimen single covariate (4 model × 16 covariate × 2 window × 3 horizon)
- Screening criteria: MAPE improvement ≥ 0.3% OR DA improvement ≥ 1pp
- Hasil: Screening1 (7 vars) dan Screening2 (10 vars) groups

### 3.4 Phase 2 — Hyperparameter Optimization
- Optuna dengan TPE sampler, 30 trials per model
- Objective: minimize avg MAPE (semua window × horizon)
- Parameter yang dioptimasi: tabel perbandingan default vs Optuna

### 3.5 Phase 3 — Group Experiments
- 6 covariate groups × 4 models × 2 windows × 3 horizons = 144 eksperimen
- Evaluasi: MAPE, DA, MASE, Hit Large

### 3.6 Phase 4 — SHAP Analysis
- TreeExplainer pada test set (out-of-sample)
- Agregasi per variabel dan per lag

---

## 4. Results and Discussion

### 4.1 Covariate Screening Results
- Tabel: pass rate tiap covariate, Tier 1/2/3
- Temuan: dominasi komoditas (Silver, WTI, Gold)
- Diskusi: mengapa variabel makro kurang signifikan?

### 4.2 Hyperparameter Optimization: Default vs Optuna
- Tabel perbandingan parameter (dari notebook 02b)
- Tabel improvement MAPE per model dan per skenario
- Diskusi: model mana yang paling terpengaruh oleh tuning?

### 4.3 Group Experiment Results
- Top configurations per horizon
- Perbandingan Screening1 vs All_Covariates
- Tradeoff: jumlah variabel vs akurasi (parsimony)

### 4.4 Multi-Horizon Analysis
- Tabel MAPE dan DA per H1/H5/H20
- Insight: degradasi akurasi seiring horizon membesar

### 4.5 SHAP: Economic Interpretability
- Bar chart variable importance
- Dependence plots top 3 variabel
- Implikasi ekonomi: komoditas ekspor Indonesia paling berpengaruh

---

## 5. Conclusion
- Ringkasan temuan utama (3–4 poin)
- Jawaban atas research questions
- Limitation: single market, look-ahead GDP, DA rendah (~52%)
- Future work: deep learning comparison, macro regime detection

---

## References
[...]

---

## Appendix
- A. Full parameter search space Optuna
- B. Full screening results table
- C. Additional SHAP plots
