# [JUDUL TESIS SEMENTARA]
### Prediksi Multi-Horizon Indeks Harga Saham Gabungan (IHSG) Menggunakan Tree-Based Ensemble Learning, Optimasi Hyperparameter Bayesian, dan SHAP

**Penulis:** Hakam
**Tahun:** 2026

> **Catatan penggunaan dokumen ini:** Struktur bab mengikuti format yang kamu tentukan, disamakan penomoran sub-bab-nya dengan referensi tesis `tesis-reference/TESIS_Teuku.docx` (Program Master of Digital Economy, BINUS/UNPAD) supaya konsisten dengan ekspektasi pembimbing/penguji. Bagian yang sudah representatif dari hasil riset aktual (`README.md`, `RINGKASAN_REVISI_METODOLOGI.md`, notebook `00`–`04`) ditulis lebih detail. Bagian yang sifatnya naratif/opini akademis (Latar Belakang, Perumusan Masalah, landasan teori naratif) sengaja **hanya berisi poin-poin besar** yang perlu kamu kembangkan sendiri jadi paragraf, sesuai permintaan — supaya tulisan akhir tetap dalam suara kamu sendiri.
>
> Tiga diagram baru sudah dibuat di `draft/figures/` (skrip pembuatnya: `draft/figures/make_thesis_diagrams.py`):
> - `gambar_2_kerangka_pemikiran.png` — Gambar 2.x, Kerangka Pemikiran
> - `gambar_3_1_tahapan_penelitian.png` — Gambar 3.1, Diagram Alur Penelitian
> - `gambar_3_2_desain_walkforward_cv.png` — Gambar 3.2, ilustrasi walk-forward CV dengan tanggal fold **asli** (dihitung langsung dari `cv_lib.py` + data hasil fix, bukan ilustrasi manual)

---

## DAFTAR ISI (rencana)

```
BAB I PENDAHULUAN
  1.1 Latar Belakang
  1.2 Perumusan Masalah
  1.3 Tujuan Penelitian
  1.4 Manfaat Penelitian

BAB II TINJAUAN PUSTAKA
  2.1 Pasar Modal dan Indeks Harga Saham Gabungan
  2.2 Variabel Makroekonomi dan Komoditas
  2.3 Konsep Stasioneritas dan Uji Augmented Dickey-Fuller
  2.4 Look-Ahead Bias dan Lag Publikasi Data pada Time Series Finansial
  2.5 Model Tree-Based Ensemble Learning
      2.5.1 Random Forest
      2.5.2 Extra Trees (Extremely Randomized Trees)
      2.5.3 Extreme Gradient Boosting (XGBoost)
      2.5.4 Light Gradient Boosting Machine (LightGBM)
  2.6 Walk-Forward Cross-Validation untuk Data Time Series
  2.7 Optimasi Hyperparameter Bayesian (Optuna / TPE)
  2.8 SHAP (SHapley Additive exPlanations)
  2.9 Metrik Evaluasi Model (MAPE, Directional Accuracy, MASE)
  2.10 Penelitian Terdahulu
  2.11 Kerangka Pemikiran

BAB III METODOLOGI PENELITIAN
  3.1 Tahapan Penelitian
      3.1.1 Pengumpulan Data
      3.1.2 Pra-Pemrosesan Data
      3.1.3 Desain Validasi Walk-Forward CV
      3.1.4 Tahap A: Seleksi Model dan Konfigurasi (Default Hyperparameter)
      3.1.5 Tahap B: Optimasi Hyperparameter (Nested-CV Optuna)
      3.1.6 Interpretasi Model dengan SHAP
      3.1.7 Visualisasi dan Pembahasan Hasil
  3.2 Ruang Lingkup Penelitian
  3.3 Data dan Sumber Data

BAB IV HASIL DAN PEMBAHASAN
  4.1 Deskripsi Data dan Hasil Uji Stasioneritas
  4.2 Hasil Tahap A: Seleksi Model dan Konfigurasi
  4.3 Hasil Tahap B: Optimasi Hyperparameter (Nested-CV Tuning)
  4.4 Analisis Prediksi per Fold
  4.5 Interpretasi SHAP
  4.6 Pembahasan

BAB V KESIMPULAN DAN SARAN
  5.1 Kesimpulan
  5.2 Saran

DAFTAR PUSTAKA
```

---

# BAB I PENDAHULUAN

## 1.1 Latar Belakang

**Poin-poin yang perlu dijelaskan (bukan draft final, kembangkan jadi paragraf):**

1. **Konteks & urgensi** — IHSG sebagai barometer ekonomi Indonesia (kutip Bimenyimana et al., 2025 dan Mankiw soal EMH, sudah ada draft narasinya di `reference/tinjauan_pustaka_bab2 (1).docx` §2.1). Kenapa prediksi IHSG penting: investor, fund manager, regulator, dan riset akademik keuangan Indonesia yang secara relatif masih lebih sedikit dibanding pasar AS/Eropa/Tiongkok.
2. **Tantangan prediksi IHSG** — non-linear, volatile, dipengaruhi banyak faktor sekaligus (makro domestik, harga komoditas global, sentimen regional). Efficient Market Hypothesis (EMH) tidak menghalangi upaya menangkap sinyal dari variabel fundamental (poin ini sudah ada di draft bab 2 kamu, tinggal disambungkan ke pendahuluan).
3. **Gap metodologis di literatur & pipeline versi awal kamu sendiri** — ini yang membuat penelitian ini punya kontribusi jelas, jadi tonjolkan:
   - Banyak studi prediksi IHSG memakai **single train/test split** (biasanya 80/20), rentan terhadap *look-ahead bias* dan tidak merepresentasikan performa across banyak kondisi pasar (COVID crash, siklus kenaikan suku bunga, dst).
   - Variabel makroekonomi bulanan/kuartalan (GDP, M2, BI Rate, dst) sering di-*merge* ke data harian memakai tanggal *label periode* (mis. GDP diberi tanggal awal kuartal), bukan tanggal **rilis publikasi** resminya — celah ini secara diam-diam memberi model informasi masa depan (kasus nyata: GDP Q1 dilabeli 1 Januari padahal BPS baru merilis ±5 Mei,±124 hari kemudian).
   - Skema optimasi hyperparameter yang lazim menyamaratakan (rata-rata MAPE lintas skenario/window/horizon) sebelum memilih model terbaik, bukan menyeleksi dulu baru optimasi khusus konfigurasi terpilih.
   - Ketiga isu ini **ditemukan sendiri lewat proses review naskah/pembimbingan** pada versi awal riset ini (lihat `modelling/legacy_pre_revision/` dan `RINGKASAN_REVISI_METODOLOGI.md` §1) — jadi bukan sekadar isu teoritis dari literatur, tapi pelajaran langsung dari proses riset yang lantas memotivasi desain revisi (walk-forward CV + koreksi lag publikasi + skema dua tahap). Ini bagian yang membuat Latar Belakang kamu punya "cerita" yang otentik, bukan generik.
4. **Posisi penelitian ini** — membangun 4 model tree-ensemble (Random Forest, Extra Trees, XGBoost, LightGBM), diseleksi & dituning dengan skema dua tahap yang menghindari tiga celah di atas, lalu diinterpretasi dengan SHAP supaya hasilnya tidak *black-box* bagi pemangku kepentingan ekonomi.

## 1.2 Perumusan Masalah

**Format ikuti gaya referensi (poin a/b/c dalam bentuk pertanyaan).** Draf pertanyaan (silakan revisi bahasanya):

a. Bagaimana performa keempat model tree-based ensemble (Random Forest, Extra Trees, XGBoost, LightGBM) dalam memprediksi IHSG pada berbagai horizon (H1, H5, H20) ketika dievaluasi dengan walk-forward cross-validation yang bebas dari look-ahead bias?

b. Kombinasi model, kelompok variabel kovariat (makro/komoditas/regional), window lookback, dan horizon manakah yang memberikan akurasi prediksi (MAPE) terbaik?

c. Sejauh mana optimasi hyperparameter Bayesian (Optuna/TPE) meningkatkan akurasi model dan konfigurasi terpilih dibandingkan hyperparameter default, dan apakah peningkatannya konsisten di berbagai kondisi pasar (termasuk periode krisis)?

d. Variabel apa yang paling berpengaruh terhadap prediksi IHSG berdasarkan interpretasi SHAP, dan apakah temuan ini sejalan dengan teori makroekonomi/keuangan yang relevan?

## 1.3 Tujuan Penelitian

Sesuaikan satu-satu dengan rumusan masalah di atas (pola a→a, b→b, ...):

a. Mengevaluasi dan membandingkan performa empat model tree-based ensemble dalam memprediksi IHSG multi-horizon menggunakan desain walk-forward CV.
b. Mengidentifikasi kombinasi model, kelompok kovariat, window, dan horizon dengan akurasi prediksi terbaik.
c. Mengevaluasi peningkatan akurasi dari optimasi hyperparameter Bayesian (Optuna) dibandingkan hyperparameter default, termasuk konsistensinya lintas kondisi pasar.
d. Mengidentifikasi variabel-variabel makroekonomi, komoditas, dan regional yang paling berpengaruh terhadap prediksi IHSG melalui interpretasi SHAP.

## 1.4 Manfaat Penelitian

**Poin besar untuk dikembangkan:**
- **Akademis** — kontribusi metodologis: desain walk-forward CV + koreksi lag publikasi untuk riset prediksi pasar saham Indonesia (sering diabaikan di studi sejenis); memperkaya literatur ML utk keuangan di pasar berkembang (emerging market) dibanding studi yang dominan di pasar maju.
- **Praktis (investor/fund manager)** — pemahaman variabel apa yang benar-benar layak dipantau (mis. jika NPL_Ratio dominan secara SHAP, ini sinyal ke arah kesehatan sektor perbankan sebagai leading indicator).
- **Regulator/pembuat kebijakan (BI/OJK)** — insight soal sensitivitas IHSG terhadap variabel makro domestik vs komoditas vs global.
- **Metodologis untuk peneliti lain** — pipeline & kode (`cv_lib.py`) yang reusable untuk riset time-series finansial lain, termasuk cara mendeteksi & mengoreksi look-ahead bias.

---

# BAB II TINJAUAN PUSTAKA

> Instruksi kamu: sebelum ke dua sub-bab wajib (Penelitian Terdahulu, Kerangka Pemikiran), harus membahas **semua landasan teori** yang dipakai riset ini. Berikut daftar lengkapnya berdasarkan apa yang benar-benar dipakai di `cv_lib.py` dan notebook `00`–`04` — pastikan tidak ada yang kelewatan saat kamu tulis penuh.

## 2.1 Pasar Modal dan Indeks Harga Saham Gabungan

**Sudah ada draft lengkap dan bagus** di `reference/tinjauan_pustaka_bab2 (1).docx` (definisi pasar modal, fungsi pasar modal menurut Lubis et al. 2024, sejarah IHSG menurut Kung et al. 2010, Efficient Market Hypothesis menurut Mankiw, dampak COVID-19 menurut Sugandi 2022, faktor perilaku investor menurut Marciano et al. 2025). **Yang perlu disesuaikan sebelum dipakai di tesis ini:**
- Draft itu ditulis untuk desain lama "Model 1 (makro bulanan) / Model 2 (komoditas harian)" — riset saat ini pakai **satu model terpadu dengan 15 kovariat** (bukan dua model terpisah per frekuensi data), jadi kalimat yang menyebut "Model 1"/"Model 2" perlu direvisi mengikuti desain terkini (lihat README §Dataset & §Models).
- Bagian EMH & random walk penting untuk membingkai kenapa Directional Accuracy hasil riset kamu (~50-54%, lihat Bab IV) mendekati level koin/random walk — itu justru **konsisten** dengan EMH, bukan kegagalan model.

## 2.2 Variabel Makroekonomi dan Komoditas

**Juga sudah ada draft lengkap** di file yang sama (§2.2, sub 2.2.1 Variabel Makroekonomi Domestik, 2.2.2 Harga Komoditas Global) — mencakup inflasi/CPI, suku bunga/BI Rate, nilai tukar/USDIDR, money supply/M2, **NPL Ratio** (penting — ini variabel dengan SHAP tertinggi di hasil riset, lihat Bab IV.5, jadi landasan teorinya krusial untuk dikuatkan), harga komoditas (Creti et al. 2012 soal financialization of commodity markets, emas sebagai safe-haven, batu bara & nikel menurut Rahmah et al. 2024), dan STI sebagai proxy sentimen regional.

**Yang perlu ditambahkan:** teori singkat soal **US Treasury 10Y** (belum dibahas di draft lama) sebagai proxy kondisi likuiditas global/US monetary policy yang mempengaruhi capital flow ke emerging market termasuk Indonesia — relevan karena US_Treasury_10Y adalah salah satu dari 15 kovariat penelitian ini.

## 2.3 Konsep Stasioneritas dan Uji Augmented Dickey-Fuller

**Belum ada draft, perlu ditulis baru.** Poin besar:
- Kenapa model time-series/ML butuh data stasioner (mean & varians konstan sepanjang waktu) — data harga level (IHSG, harga komoditas) umumnya non-stasioner (random walk).
- Konsep unit root & uji ADF (Dickey & Fuller, 1979) — hipotesis nol = ada unit root (non-stasioner).
- Transformasi yang dipakai penelitian ini: **log-difference** (return) untuk variabel level (IHSG dan `LEVEL_VARS`: M2, USDIDR, komoditas, GDP, STI), **first-difference** untuk variabel rate (`RATE_VARS`: BI_Rate, CPI, NPL_Ratio, US_Treasury_10Y) — jelaskan kenapa keduanya beda perlakuan (rate sudah dalam bentuk persentase, tidak perlu log).
- Sitasi standar: Dickey & Fuller (1979), atau textbook time series (Enders, Box-Jenkins).

## 2.4 Look-Ahead Bias dan Lag Publikasi Data pada Time Series Finansial

**Belum ada draft — ini bagian teori yang cukup unik/orisinal untuk tesis kamu, worth ditulis dengan detail** karena jadi salah satu kontribusi metodologis utama:
- Definisi look-ahead bias dalam backtesting/prediksi finansial (model "melihat" informasi yang belum tersedia pada waktu prediksi).
- Kenapa data makro (GDP, M2, dll) rawan kasus ini: sumber data biasanya melabeli nilai dengan tanggal **periode referensi** (mis. GDP Q1 dilabeli 1 Januari), padahal nilai itu baru **dipublikasikan** oleh BPS/BI berminggu-minggu atau berbulan-bulan kemudian.
- Cari 1-2 sitasi literatur soal *data revision* / *publication lag* dalam nowcasting ekonomi (bidang ini punya nama: "real-time data" vs "vintage data" dalam makroekonomi — bisa cari istilah "real-time data revision GDP" atau "ragged-edge data" untuk sitasi akademis yang pas).
- Jelaskan solusi penelitian ini: menggeser tanggal setiap seri makro sebesar lag publikasi riilnya sebelum di-*merge* (`merge_asof(direction="backward")`), dengan lag diverifikasi ke kalender rilis BPS/BI (detail di Bab III/README): GDP +124 hari, M2 & NPL_Ratio +51 hari, CPI & BI_Rate +0 hari.

## 2.5 Model Tree-Based Ensemble Learning

Landasan umum: konsep *decision tree*, lalu *ensemble learning* — dua aliran utama **bagging** (paralel, tiap pohon independen, mengurangi varians) vs **boosting** (sekuensial, tiap pohon mengoreksi error pohon sebelumnya, mengurangi bias). Referensi bisa pakai gaya penjelasan Gambar 2.1/2.2 di `TESIS_Teuku.docx` sebagai contoh ilustrasi (tapi buat ilustrasi versi kamu sendiri, jangan salin).

### 2.5.1 Random Forest
Breiman (2001) — bagging + random feature subsampling per split. Parameter kunci yang dipakai penelitian ini (lihat `cv_lib.py`): `n_estimators`, `max_depth`, `max_features`, `max_samples`, `min_samples_split`, `min_samples_leaf`.

### 2.5.2 Extra Trees (Extremely Randomized Trees)
Geurts, Ernst & Wehenkel (2006) — mirip Random Forest tapi split threshold dipilih **acak** (bukan dicari yang optimal), sedikit lebih banyak randomisasi → varians lebih rendah, bias sedikit lebih tinggi. **Ini model pemenang resmi penelitian ini** (lihat Bab IV) — jadi bagian ini layak dijelaskan agak lebih detail dibanding model lain.

### 2.5.3 Extreme Gradient Boosting (XGBoost)
Chen & Guestrin (2016) — gradient boosting dengan regularisasi eksplisit (L1/L2). Parameter: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_weight`.

### 2.5.4 Light Gradient Boosting Machine (LightGBM)
Ke et al. (2017) — gradient boosting dengan *histogram-based split* dan *leaf-wise growth* (bukan level-wise seperti XGBoost default), lebih cepat untuk dataset besar. Parameter: sama seperti XGBoost ditambah `num_leaves`.

## 2.6 Walk-Forward Cross-Validation untuk Data Time Series

**Belum ada draft — penting, ini fondasi metodologi utama Bab III & IV:**
- Kenapa **k-fold random split biasa tidak valid** untuk data time series (mengacak data melanggar urutan waktu, menyebabkan model dilatih dengan data "masa depan" untuk memprediksi "masa lalu").
- Konsep **expanding window**: training set membesar bertahap, tidak pernah mengecil, tidak pernah melihat data setelah titik evaluasi.
- Konsep **embargo gap**: jeda antara akhir data training dan awal data uji, untuk mencegah kebocoran akibat lag/autokorelasi dekat batas potongan (Lopez de Prado, 2018, *Advances in Financial Machine Learning* — sitasi yang relevan & banyak dipakai untuk konsep ini).
- Konsep **nested cross-validation**: pencarian hyperparameter (inner loop) dan evaluasi performa (outer loop) dipisah, supaya proses tuning tidak pernah "mengintip" data yang dipakai untuk melaporkan hasil akhir.
- Rujuk Gambar 3.2 (`gambar_3_2_desain_walkforward_cv.png`) untuk ilustrasi konkret desain ini di penelitian kamu.

## 2.7 Optimasi Hyperparameter Bayesian (Optuna / TPE)

**Belum ada draft.** Poin besar:
- Kenapa Grid Search/Random Search kurang efisien untuk ruang pencarian besar (referensi: Bergstra & Bengio, 2012 — juga dikutip di `TESIS_Teuku.docx` soal risiko overfitting ke partisi data tertentu saat tuning terlalu ekstensif, poin bagus untuk connect ke temuan Bab IV soal RandomForest yang sedikit dirugikan oleh tuning).
- Konsep Bayesian Optimization: membangun model probabilistik dari hasil trial sebelumnya untuk memilih titik pencarian berikutnya yang lebih menjanjikan.
- Tree-structured Parzen Estimator (TPE) — algoritma spesifik yang dipakai Optuna (Bergstra et al., 2011; Akiba et al., 2019 untuk paper Optuna itu sendiri).

## 2.8 SHAP (SHapley Additive exPlanations)

Lundberg & Lee (2017). Bisa mengikuti struktur penjelasan `TESIS_Teuku.docx` §2.5 (asal dari *cooperative game theory*/Shapley value, TreeExplainer sebagai varian efisien untuk model tree-based) tapi sesuaikan konteks: penelitian ini regresi (memprediksi nilai return IHSG), bukan klasifikasi rating seperti referensi — jadi framing "SHAP value = kontribusi tiap fitur terhadap **nilai prediksi**" bukan "terhadap probabilitas kelas".

## 2.9 Metrik Evaluasi Model (MAPE, Directional Accuracy, MASE)

**Belum ada draft — beda dari referensi (yang pakai metrik klasifikasi: accuracy/precision/recall/F1/confusion matrix), karena penelitian ini regresi.** Poin per metrik:
- **MAPE** (Mean Absolute Percentage Error) — metrik utama penelitian ini, mudah diinterpretasi sebagai persentase error rata-rata.
- **Directional Accuracy (DA)** — persentase prediksi yang benar arah naik/turunnya (bukan besarannya) — penting karena untuk keputusan investasi, arah pergerakan sering lebih relevan daripada presisi nilai.
- **MASE** (Mean Absolute Scaled Error) — membandingkan error model terhadap error naive forecast (Hyndman & Koehler, 2006), berguna untuk menilai apakah model benar-benar "lebih baik dari menebak nilai kemarin."

## 2.10 Penelitian Terdahulu — *wajib*

Format: gunakan **tabel ringkasan** (Peneliti, Tahun, Metode, Temuan Utama) seperti kelaziman tesis, didahului 2-3 paragraf naratif yang mengalir kronologis/tematis (contoh gaya ada di `TESIS_Teuku.docx` baris 604-780).

**Struktur naratif yang disarankan (3 kelompok):**
1. **Metode statistik/ekonometrik tradisional untuk prediksi pasar saham** — ARIMA, GARCH, regresi linear (bisa sitasi Endri et al. 2020 yang sudah ada di draft bab 2 kamu — pakai GARCH(1,1) untuk hubungan makro-IHSG).
2. **Machine learning untuk prediksi indeks saham** — cari & baca beberapa paper yang paling relevan di `reference/` (nama file mengindikasikan relevansi tinggi, prioritaskan baca ini dulu):
   - `9.+(402-414)+-+Short+term+IHSG+Closinv+Price+Prediction.pdf` — langsung tentang prediksi IHSG
   - `Determinants and Prediction of the Stock Market during COVID-19_ Evidence from Indonesia.pdf`
   - `IMFI_2022_03_Salim.pdf`, `IMFI_2024_01_Xu.pdf`
   - `fischer2018.pdf` (kemungkinan LSTM utk prediksi saham — cek relevansi horizon/metode)
   - `garg2021.pdf`, `heo2020.pdf`
   - `A_Hybrid_Relational_Approach_Toward_Stock_Price_Prediction_and_Profitability.pdf`
   - `Using_Machine_Learning_on_Macr.pdf` — kemungkinan langsung soal ML + variabel makro
3. **Studi yang membandingkan tree-ensemble & pakai SHAP untuk keuangan** — `BAREKENG_Implementation of IML for Credit rating_rev.pdf`, dan tesis referensi Teuku sendiri (kredit rating, bukan prediksi indeks, tapi metodologi model+SHAP-nya relevan untuk dibandingkan).

**Yang perlu ditegaskan di akhir sub-bab ini (gap statement):** studi-studi di atas umumnya (a) pakai single train/test split, bukan walk-forward CV; (b) jarang mengoreksi lag publikasi data makro; (c) jarang membandingkan banyak kombinasi window×horizon×kovariat secara sistematis sebelum tuning. Ini yang membedakan penelitian ini.

> **Catatan proses:** ada ~50 PDF di `reference/` — saya tidak membaca semua isinya secara mendalam untuk ringkasan ini (di luar scope permintaan kamu saat ini). Kalau mau, saya bisa bantu baca & ekstrak temuan kunci dari daftar file di atas satu per satu untuk diisi ke tabel Penelitian Terdahulu.

## 2.11 Kerangka Pemikiran — *wajib*

Gunakan **Gambar 2.x** (`draft/figures/gambar_2_kerangka_pemikiran.png`, sudah dibuat) sebagai ilustrasi alur: Masalah → Data & Pra-pemrosesan → Tahap A (seleksi) → Tahap B (tuning) → Interpretasi SHAP → Hasil, ditutup dua hipotesis:

- **Hipotesis 1:** Model tree-ensemble dengan kovariat terpilih memberikan MAPE lebih rendah dibanding baseline autoregresif (tanpa kovariat).
- **Hipotesis 2:** SHAP dapat mengidentifikasi variabel makro/komoditas/regional yang paling berpengaruh terhadap prediksi IHSG.

Tulis 1-2 paragraf naratif yang menjelaskan diagram ini dengan kalimat (jangan cuma taruh gambar tanpa narasi), lihat contoh gaya di `TESIS_Teuku.docx` baris 937-968.

---

# BAB III METODOLOGI PENELITIAN

## 3.1 Tahapan Penelitian

Gunakan **Gambar 3.1** (`draft/figures/gambar_3_1_tahapan_penelitian.png`) sebagai diagram alur utama. Detail tiap tahap (ini sudah bisa ditulis cukup lengkap karena mengikuti persis apa yang ada di kode):

### 3.1.1 Pengumpulan Data
Data sekunder diunduh dari GitHub repository penelitian (`dataset/`): IHSG harian + 15 variabel kovariat, periode 2 Januari 2015 – 31 Januari 2025 (lihat Tabel 3.1 di §3.3).

### 3.1.2 Pra-Pemrosesan Data
- Parsing & merge tiap sumber data (format tanggal & desimal berbeda-beda per sumber, ditangani di `00_data_preprocessing.ipynb`).
- **Koreksi lag publikasi** untuk variabel makro bulanan/kuartalan sebelum `merge_asof(direction="backward")`: GDP +124 hari, M2 & NPL_Ratio +51 hari, CPI & BI_Rate +0 hari (dijelaskan landasan teorinya di §2.4).
- Variabel harian (USDIDR, komoditas, STI, US_Treasury_10Y) di-*forward fill* untuk mengisi akhir pekan/libur.
- Uji ADF pada level (sebelum transformasi) dan setelah transformasi log-diff/first-diff (landasan teori di §2.3), memastikan seluruh variabel stasioner sebelum dipakai model.
- Baris dengan nilai kosong dibuang (akibat lag publikasi & transformasi diff) → total baris turun dari 2.443 menjadi 2.408.

### 3.1.3 Desain Validasi Walk-Forward CV
5-fold **expanding window** dengan **embargo** = horizon prediksi (landasan teori §2.6), diimplementasikan di `cv_lib.expanding_window_folds()`. Lihat **Gambar 3.2** untuk ilustrasi konkret dengan tanggal fold asli.

### 3.1.4 Tahap A: Seleksi Model dan Konfigurasi (Default Hyperparameter)
Grid penuh: 4 model × 21 skenario kovariat (1 baseline + 15 kovariat tunggal + 5 kelompok) × 2 window (20, 120 hari bursa) × 3 horizon (H1, H5, H20) = 504 kombinasi × 5 fold = **2.520 fit**, seluruhnya memakai hyperparameter default (apples-to-apples, belum ada yang dituning) — `02c_model_selection_cv.ipynb`. Konfigurasi dengan mean CV MAPE terendah dipilih sebagai pemenang.

### 3.1.5 Tahap B: Optimasi Hyperparameter (Nested-CV Optuna)
Optuna dengan sampler TPE (seed=42) dijalankan **hanya** pada konfigurasi pemenang Tahap A, dengan nested cross-validation: setiap outer fold (5 fold yang sama dengan Tahap A) punya inner search 3-fold yang dibatasi ketat pada rentang training outer fold-nya sendiri (diverifikasi lewat `assert` di kode, bukan sekadar diasumsikan) — `02d_nested_tuning_cv.ipynb`. 40 trial per outer fold + 50 trial untuk tahap produksi final.

### 3.1.6 Interpretasi Model dengan SHAP
`TreeExplainer` dijalankan pada model final (hyperparameter produksi) dan dievaluasi pada **fold uji out-of-sample terakhir** (bukan data training) — `03_shap_analysis.ipynb`. Global importance dihitung sebagai mean |SHAP value| per variabel.

### 3.1.7 Visualisasi dan Pembahasan Hasil
`02e_visualization_cv.ipynb` (heatmap Tahap A/B, perbandingan window/DA) dan `04_visualization.ipynb` (prediksi aktual-vs-prediksi per outer fold, figur publikasi).

Seluruh proses memakai Python dengan pustaka Darts (wrapper model time series), scikit-learn, XGBoost, LightGBM, Optuna, SHAP, Pandas, NumPy, Matplotlib, statsmodels, dan Jupyter Notebook.

## 3.2 Ruang Lingkup Penelitian

- **Objek penelitian:** Indeks Harga Saham Gabungan (IHSG), Bursa Efek Indonesia.
- **Periode data:** 2 Januari 2015 – 31 Januari 2025 (~2.408 hari bursa setelah pra-pemrosesan).
- **Variabel penelitian:** 15 kovariat — 7 makro (BI Rate, CPI, M2, NPL Ratio, GDP, USD/IDR, US Treasury 10Y), 7 komoditas (Batu Bara, Tembaga, Nikel, Perak, Timah, Emas, WTI), 1 regional (Straits Times Index/STI).
- **Model penelitian:** 4 model tree-based ensemble (Random Forest, Extra Trees, XGBoost, LightGBM) dibungkus lewat Darts, dituning dengan Optuna, diinterpretasi dengan SHAP.
- **Window & horizon:** lookback window 20 & 120 hari bursa; horizon prediksi H1 (besok), H5 (1 minggu), H20 (1 bulan).
- **Batasan yang perlu ditulis eksplisit** (supaya jujur secara akademis dan mengantisipasi pertanyaan penguji):
  - Lag publikasi NPL_Ratio (51 hari) masih **placeholder** mengikuti pola M2, belum diverifikasi ke kalender rilis resmi OJK — sebutkan ini sebagai keterbatasan, bukan disembunyikan (juga masuk ke Bab V Saran).
  - Optimasi hyperparameter (Tahap B) hanya dijalankan penuh untuk **satu konfigurasi pemenang** (bukan keempat model) — meski ada hasil eksplorasi tambahan untuk 3 model lain (lihat §4.3), itu di luar rancangan resmi Tahap A/B.
  - Window test antar fold **saling tumpang tindih** (~1 tahun overlap antar fold berurutan) — bukan 5 sampel independen, perlu disebutkan sebagai batasan desain CV di bagian ini atau di Bab V.

## 3.3 Data dan Sumber Data

**Tabel 3.1 — Variabel Penelitian** (isi tabel ini persis, datanya sudah final dari `README.md`):

| Kelompok | Variabel | Frekuensi Asli | Transformasi | Sumber |
|---|---|---|---|---|
| Target | IHSG (harga penutupan harian) | Harian | Log-difference | Bursa Efek Indonesia |
| Makro | BI_Rate | Bulanan → ffill harian | First-difference | Bank Indonesia |
| Makro | CPI | Bulanan → ffill harian | First-difference | BPS |
| Makro | M2 (jumlah uang beredar) | Bulanan → ffill harian | Log-difference | Bank Indonesia |
| Makro | NPL_Ratio | Bulanan → ffill harian | First-difference | OJK |
| Makro | GDP | Kuartalan → ffill harian | Log-difference | BPS |
| Makro | USD/IDR | Harian | Log-difference | — |
| Makro (global) | US Treasury 10Y | Harian | First-difference | — |
| Komoditas | Batu Bara, Tembaga, Nikel, Perak, Timah, Emas | Harian | Log-difference | — |
| Komoditas (energi) | WTI Crude Oil | Harian | Log-difference | — |
| Regional | Straits Times Index (STI) | Harian | Log-difference | Bursa Efek Singapura |

Sumber data mentah: `dataset/` (CSV per variabel) + `dataset/publikasi bi/` (bulletin resmi Bank Indonesia untuk verifikasi tanggal rilis). Seluruh 15 variabel terbukti non-stasioner pada level (uji ADF p > 0.05) dan stasioner setelah transformasi (p < 0.05) — detail tabel lengkap di Bab IV.1.

---

# BAB IV HASIL DAN PEMBAHASAN

> Ini bab yang paling detail sesuai permintaan kamu — semua angka di bawah adalah hasil riil dari `stage_a_summary.csv`, `stage_a_fold_results.csv`, `nested_cv_outer_fold_results.csv`, `nested_cv_default_vs_tuned_summary.csv`, dan `shap_variable_importance_ExtraTrees_20260713_0425.csv`, bukan contoh/placeholder.

## 4.1 Deskripsi Data dan Hasil Uji Stasioneritas

Tulis ringkasan: 2.408 baris data (2015-02-23 s/d 2025-01-31) setelah koreksi lag publikasi + pembuangan baris kosong. Sajikan tabel ADF level vs transformed (`saved_models/adf_level_20260712_1323.csv`, `saved_models/adf_transformed_20260712_1323.csv`) — semua p-value turun di bawah 0.05 setelah transformasi.

**Item yang perlu diperbaiki sebelum ditulis final:** `descriptive_stats_transformed.csv` (statistik deskriptif: mean, std, skewness, kurtosis per variabel) saat ini **belum di-generate ulang setelah fix lag publikasi** — filenya masih menunjukkan N=2.442 (angka dari dataset lama, 2.443 baris sebelum fix), bukan 2.408. Jalankan ulang sel yang menghasilkan file ini di `00_data_preprocessing.ipynb` sebelum dipakai di tesis final, supaya konsisten dengan angka lain di seluruh Bab IV.

## 4.2 Hasil Tahap A: Seleksi Model dan Konfigurasi

**Tabel 4.1 — Mean MAPE per Model, per Window** (dari `stage_a_summary.csv`, hyperparameter default):

| Model | W20 Mean MAPE (%) | W120 Mean MAPE (%) |
|---|---|---|
| Extra Trees | 1.2314 | 1.2328 |
| Random Forest | 1.2354 | 1.2388 |
| LightGBM | 1.3741 | 1.3582 |
| XGBoost | 1.3685 | 1.3682 |

→ **Model bagging (Random Forest, Extra Trees) unggul dibanding model boosting (XGBoost, LightGBM) pada hyperparameter default** — kebalikan dari hasil setelah tuning (lihat §4.3). Ini poin diskusi penting: boosting butuh tuning untuk kompetitif, bagging sudah kuat sejak default.

**Tabel 4.2 — Mean MAPE dan Directional Accuracy per Horizon** (across seluruh 504 kombinasi grid — ini yang menunjukkan sifat **multi-horizon** penelitian, sesuai judul):

| Horizon | Mean MAPE (%) | Mean Directional Accuracy (%) |
|---|---|---|
| H1 (besok) | 0.66 | 51.42 |
| H5 (1 minggu) | 1.10 | 50.85 |
| H20 (1 bulan) | 2.14 | 49.72 |

→ **Poin pembahasan penting:** MAPE meningkat ~3x dari H1 ke H20 (wajar, ketidakpastian bertambah seiring horizon), sementara Directional Accuracy justru **menurun mendekati 50%** (level tebak-koin) — mengindikasikan model kehilangan kemampuan prediksi arah pada horizon panjang, konsisten dengan Efficient Market Hypothesis (lebih sulit memprediksi arah jauh ke depan). Ini nuansa penting untuk didiskusikan karena judul tesis menekankan "multi-horizon" — jangan hanya laporkan H1 tanpa membahas H5/H20.

**Tabel 4.3 — Top 10 Konfigurasi (mean CV MAPE terendah, tidak termasuk Baseline):**

| Model | Kovariat | Window | Horizon | Mean MAPE (%) | Std MAPE | Mean DA (%) |
|---|---|---|---|---|---|---|
| Extra Trees | Screening1 | 120 | 1 | 0.6360 | 0.150 | 52.42 |
| Extra Trees | All_Covariates | 20 | 1 | 0.6362 | 0.151 | 52.53 |
| Extra Trees | All_Covariates | 120 | 1 | 0.6362 | 0.150 | 52.32 |
| Extra Trees | All_Commodity_STI | 120 | 1 | 0.6366 | 0.152 | 51.44 |
| Extra Trees | Copper (tunggal) | 20 | 1 | 0.6367 | 0.150 | 52.78 |
| Extra Trees | All_Macro_no_UST | 120 | 1 | 0.6368 | 0.150 | 52.27 |
| Extra Trees | Silver (tunggal) | 120 | 1 | 0.6370 | 0.151 | 53.25 |
| Random Forest | All_Covariates | 20 | 1 | 0.6372 | 0.151 | 53.56 |
| Extra Trees | Screening2 | 120 | 1 | 0.6373 | 0.151 | 52.68 |
| Extra Trees | NPL_Ratio (tunggal) | 120 | 1 | 0.6373 | 0.152 | 51.65 |

→ **Konfigurasi pemenang resmi: Extra Trees · Screening1 (Silver, WTI, Gold, STI, Coal, Tin, NPL_Ratio) · Window 120 · Horizon 1**, MAPE 0,636%.

→ **Poin pembahasan penting & jujur secara akademis:** perhatikan bahwa 10 konfigurasi teratas semuanya sangat berdekatan (0,636%–0,637%), dan baseline Extra Trees **tanpa kovariat sama sekali** pada W120/H1 mendapat MAPE 0,639% — hanya beda tipis dari yang terbaik. Artinya **penambahan kovariat memberikan perbaikan marginal** dibanding model autoregresif murni untuk kombinasi model/window/horizon ini, meskipun begitu Screening1 tetap konsisten sebagai kelompok kovariat terbaik. Ini poin nuansa penting untuk Bab IV.6 (Pembahasan) — jangan berlebihan mengklaim kovariat "sangat meningkatkan" akurasi dibanding baseline pada level agregat MAPE, meski SHAP (§4.5) tetap menunjukkan kovariat seperti NPL_Ratio berkontribusi signifikan **relatif terhadap kovariat lain yang disertakan**.

## 4.3 Hasil Tahap B: Optimasi Hyperparameter (Nested-CV Tuning)

**Tabel 4.4 — Default vs Tuned per Fold, Konfigurasi Pemenang** (dari `nested_cv_outer_fold_results.csv`):

| Fold | Periode Uji | Default MAPE (%) | Tuned MAPE (%) | Default DA (%) | Tuned DA (%) | Rezim |
|---|---|---|---|---|---|---|
| 0 | 2019-02-15 – 2020-08-12 | 0,8084 | 0,8077 | 49,48 | 53,09 | COVID crash |
| 1 | 2020-03-30 – 2021-09-23 | 0,7826 | 0,7788 | 50,52 | 54,90 | COVID crash |
| 2 | 2021-05-11 – 2022-11-04 | 0,5697 | 0,5711 | 52,58 | 51,80 | Rate-hike 2022 |
| 3 | 2022-06-22 – 2023-12-18 | 0,4734 | 0,4749 | 55,67 | 51,80 | Rate-hike 2022 |
| 4 | 2023-08-03 – 2025-01-28 | 0,5459 | 0,5441 | 53,87 | 52,58 | — |

**Rata-rata: 0,636% → 0,635% (improvement +0,07%)** — sangat kecil, konsisten dengan sifat model bagging yang sudah dekat titik optimal sejak default (landasan: §2.5.1/2.5.2).

**Tabel 4.5 — Perbandingan Efek Tuning Antar Model** (dari `RINGKASAN_REVISI_METODOLOGI.md` §4, hasil eksplorasi tambahan di luar Tahap B resmi — jelaskan di teks bahwa ini eksplorasi pelengkap, bukan bagian resmi seleksi Tahap A/B):

| Model | Kovariat | Window/Horizon | Default MAPE | Tuned MAPE | Improvement |
|---|---|---|---|---|---|
| Extra Trees *(resmi)* | Screening1 | 120/H1 | 0,636% | 0,635% | +0,07% |
| Random Forest | All_Covariates | 20/H1 | 0,637% | 0,638% | −0,33% |
| XGBoost | Screening2 | 20/H1 | 0,677% | 0,636% | **+6,18%** |
| LightGBM | Screening1 | 20/H1 | 0,679% | **0,632%** | **+7,32%** |

→ **Poin pembahasan penting:** model boosting (XGBoost, LightGBM) mendapat perbaikan besar dari tuning, sedangkan bagging (Random Forest, Extra Trees) nyaris tidak berubah (Random Forest bahkan sedikit lebih buruk). **Setelah tuning, LightGBM (0,632%) sedikit mengungguli Extra Trees (0,635%)** — artinya metodologi "pilih dulu berdasar default, baru tuning" (sesuai arahan pembimbing, lihat `RINGKASAN_REVISI_METODOLOGI.md` §1) punya trade-off yang jujur untuk didiskusikan di sini: pemenang Tahap A belum tentu pemenang jika semua model dituning terlebih dahulu. Catat ini secara eksplisit sebagai temuan, bukan disembunyikan — ini justru memperkuat kredibilitas metodologis tesis kamu.

## 4.4 Analisis Prediksi per Fold

Sajikan **Gambar 4.x** dari `modelling/plot_winning_config_per_fold.png` (prediksi vs aktual, satu panel per outer fold — fold yang overlap rezim ekstrem diberi latar warna berbeda) dan `modelling/plot_pub_winning_config.png` (figur publikasi, fold terakhir). Bahas pola visual: prediksi mengikuti tren tapi melebar saat volatilitas tinggi (COVID crash), MAPE fold 0-1 (~0,78-0,81%) jauh lebih tinggi dari fold 2-4 (~0,47-0,57%) — pull-up rata-rata keseluruhan berasal dari periode krisis, bukan performa yang seragam sepanjang waktu.

**Batasan desain CV yang perlu disebut di sini juga:** window uji antar fold saling tumpang-tindih ~1 tahun (lihat Gambar 3.2) — jadi std MAPE antar fold kemungkinan **meremehkan** ketidakpastian sebenarnya karena fold-fold ini bukan sampel independen.

## 4.5 Interpretasi SHAP

**Tabel 4.6 — Ranking SHAP Global Importance** (Extra Trees, Screening1, W120/H1, dihitung pada fold uji out-of-sample terakhir):

| Rank | Variabel | Mean \|SHAP\| | Kategori |
|---|---|---|---|
| 1 | NPL_Ratio | 0,0001613 | Macro/Rate |
| 2 | Tin | 0,0000701 | Commodity |
| 3 | Silver | 0,0000644 | Commodity |
| 4 | STI | 0,0000563 | Regional |
| 5 | IHSG (lag sendiri) | 0,0000367 | Autoregressive |
| 6 | Gold | 0,0000364 | Commodity |
| 7 | WTI | 0,0000253 | Commodity |
| 8 | Coal | 0,0000231 | Commodity |

Sajikan juga Gambar 4.x dari `plot_shap_bar_ExtraTrees_120.png`, `plot_shap_beeswarm_ExtraTrees_120.png`, `plot_shap_dependence_ExtraTrees_120.png`, `plot_shap_waterfall_ExtraTrees_obs0.png`.

→ **Poin pembahasan penting:**
- **NPL_Ratio mendominasi** — sambungkan ke landasan teori §2.2.1: sektor keuangan (BBCA, BBRI, BMRI) punya bobot terbesar di komposisi IHSG, jadi kesehatan perbankan (NPL) secara struktural relevan.
- **Komoditas (Tin, Silver, Gold, WTI, Coal) mendominasi sisanya** — Indonesia sebagai eksportir komoditas, sambungkan ke Creti et al. (2012) & Rahmah et al. (2024) di landasan teori §2.2.2.
- **IHSG lag sendiri (autoregresif) hanya rank #5** — kovariat eksternal membawa informasi yang tidak sepenuhnya ada di histori IHSG sendiri, ini temuan yang mendukung Hipotesis 2.
- **Temuan ini mereplikasi hasil sebelum revisi metodologi** (model & test window berbeda, tapi variabel top & ranking kasar sama) — argumen kuat bahwa temuan bukan artefak dari isu look-ahead bias/single-split yang sudah diperbaiki (lihat §2.4 & `RINGKASAN_REVISI_METODOLOGI.md`).
- Ingatkan pembaca: SHAP dihitung pada **log-return**, jadi mencerminkan kontribusi ke prediksi return harian, bukan ke level harga IHSG secara langsung.

## 4.6 Pembahasan

Ikat kembali ke rumusan masalah §1.2 satu-per-satu:
- (a) Performa 4 model → Tabel 4.1, model bagging unggul di default; Tabel 4.5, model boosting menyusul setelah tuning.
- (b) Konfigurasi terbaik → Extra Trees/Screening1/W120/H1 (Tabel 4.3), tapi catat marginal improvement kovariat vs baseline.
- (c) Efek tuning → Tabel 4.4/4.5, minimal untuk bagging, signifikan untuk boosting, tidak konsisten antar fold (membaik di fold krisis, memburuk di fold rate-hike — diskusikan kenapa: kemungkinan overfitting hyperparameter ke pola volatilitas tertentu).
- (d) Variabel paling berpengaruh → Tabel 4.6, NPL_Ratio dominan, dikonfirmasi juga oleh temuan versi pra-revisi (replikasi independen).

Diskusikan juga **konsistensi dengan teori** — draft §2.1/§2.2 (Endri et al. 2020, dll) menemukan hubungan arah (positif/negatif) antar variabel makro dan IHSG; sementara SHAP di penelitian ini mengukur **besarnya kontribusi**, bukan arahnya secara eksplisit — kalau mau memperkuat diskusi, bisa tambahkan analisis arah dari SHAP dependence plot (`plot_shap_dependence_ExtraTrees_120.png`) untuk melihat apakah arah pengaruh NPL_Ratio/Tin/Silver konsisten dengan teori di Bab II.

---

# BAB V KESIMPULAN DAN SARAN

## 5.1 Kesimpulan

Format poin a/b/c/d mengikuti pola Tujuan Penelitian §1.3, masing-masing dengan angka konkret dari Bab IV (jangan generik). Draf kerangka:

a. [Tentang perbandingan 4 model] — Extra Trees dan Random Forest (bagging) unggul dibanding XGBoost dan LightGBM (boosting) pada hyperparameter default (MAPE ~1,23% vs ~1,37%), tetapi pola ini berbalik setelah optimasi hyperparameter Bayesian.
b. [Tentang konfigurasi terbaik] — Extra Trees dengan kelompok kovariat Screening1 (Silver, WTI, Gold, STI, Coal, Tin, NPL_Ratio), window 120 hari, horizon H1, mencapai MAPE 0,636% pada evaluasi walk-forward CV.
c. [Tentang efek tuning] — optimasi hyperparameter memberi manfaat signifikan untuk model boosting (+6-7% improvement MAPE) namun minimal/negatif untuk model bagging, dan efeknya tidak konsisten di seluruh kondisi pasar.
d. [Tentang SHAP] — NPL_Ratio adalah variabel paling berpengaruh, diikuti harga komoditas (Timah, Perak, Emas, WTI, Batu Bara) dan indeks regional STI; IHSG lag sendiri berkontribusi relatif kecil, mengindikasikan kovariat eksternal membawa informasi tambahan.

Tutup dengan kalimat soal kontribusi metodologis (koreksi lag publikasi + walk-forward CV) sebagai bagian dari kesimpulan, bukan cuma soal angka model.

## 5.2 Saran

Poin yang **sudah teridentifikasi dari proses riset sendiri** (kredibel karena berasal dari batasan nyata, bukan generik):

a. **Verifikasi lag publikasi NPL_Ratio** (saat ini 51 hari, placeholder mengikuti pola M2) terhadap kalender rilis resmi OJK sebelum hasil ini dipakai untuk keputusan yang lebih luas — mengingat NPL_Ratio adalah variabel dengan SHAP tertinggi, akurasi lag publikasinya penting.
b. **Desain ulang skema fold agar tidak tumpang tindih** (non-overlapping test windows) atau setidaknya laporkan interval kepercayaan yang memperhitungkan autokorelasi antar fold, supaya estimasi ketidakpastian (std MAPE) lebih akurat.
c. **Eksplorasi tuning penuh untuk keempat model** (bukan hanya konfigurasi pemenang Tahap A) sebagai studi lanjutan formal — hasil awal (Tabel 4.5) menunjukkan LightGBM berpotensi mengungguli Extra Trees setelah tuning, layak diverifikasi dengan rancangan Tahap A/B yang setara.
d. **Perluasan ke horizon menengah/panjang (H5, H20)** dengan tuning khusus per horizon — penelitian ini fokus tuning pada H1; §4.2 menunjukkan performa & directional accuracy menurun tajam di H20, area yang perlu model/fitur tambahan (mis. fitur teknikal, volatilitas terealisasi).
e. **Menambahkan variabel non-tradisional** (sentimen berita, data ESG, indikator teknikal) mengikuti tren riset terbaru, dan mengevaluasi kontribusinya lewat SHAP dengan kerangka yang sama.
f. **Uji ketahanan (robustness) desain walk-forward CV** pada pasar/indeks lain di kawasan ASEAN untuk menilai generalisasi temuan (khususnya dominasi NPL_Ratio dan komoditas).

---

# DAFTAR PUSTAKA

Kumpulkan dari:
- Kutipan yang sudah ada di `reference/tinjauan_pustaka_bab2 (1).docx` (Mankiw, Endri et al. 2020, Creti et al. 2012, Kung et al. 2010, Lubis et al. 2024, Sugandi 2022, Marciano et al. 2025, Rahmah et al. 2024, Bimenyimana et al. 2025, dll).
- Sitasi metode ML: Breiman (2001) Random Forest, Geurts/Ernst/Wehenkel (2006) Extra Trees, Chen & Guestrin (2016) XGBoost, Ke et al. (2017) LightGBM, Lundberg & Lee (2017) SHAP, Bergstra et al. (2011)/Akiba et al. (2019) Optuna/TPE, Lopez de Prado (2018) walk-forward CV & nested CV, Dickey & Fuller (1979) ADF test, Hyndman & Koehler (2006) MASE.
- Paper prediksi IHSG/ML-keuangan yang dipilih dari daftar `reference/` di §2.10.

---

## Ringkasan status dokumen ini

| Bagian | Status |
|---|---|
| Bab I (1.1–1.4) | Poin besar tersedia, perlu ditulis jadi paragraf oleh kamu |
| Bab II 2.1–2.2 | **Draft lengkap sudah ada** (`reference/tinjauan_pustaka_bab2 (1).docx`), perlu penyesuaian kecil (hapus framing "Model 1/Model 2") |
| Bab II 2.3–2.9 | Poin besar + sitasi yang disarankan, belum ada draft, perlu ditulis baru |
| Bab II 2.10 Penelitian Terdahulu | Struktur + daftar PDF relevan tersedia, isi/sitasi detail perlu dibaca & ditulis |
| Bab II 2.11 Kerangka Pemikiran | **Diagram sudah dibuat** (Gambar 2.x), narasi pendukung perlu ditulis |
| Bab III | Cukup lengkap, mengikuti kode aktual, siap dikembangkan jadi paragraf |
| Bab IV | **Paling detail** — semua tabel & angka sudah final dari hasil riset, tinggal dinarasikan + 1 item perbaikan data (`descriptive_stats_transformed.csv` perlu di-generate ulang) |
| Bab V | Poin kesimpulan & saran konkret tersedia, perlu dirapikan jadi kalimat akhir |
