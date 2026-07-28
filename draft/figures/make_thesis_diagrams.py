"""Generates the three conceptual diagrams referenced by tesis_outline.md
(Kerangka Pemikiran, Diagram Alur/Tahapan Penelitian, Desain Walk-Forward CV).

These are thesis-writing figures, not pipeline outputs -- kept separate from
modelling/*.png (which are the actual result plots the notebooks generate).
The walk-forward CV timeline (Gambar 3.2) uses the real fold boundaries from
cv_lib.expanding_window_folds() against the post-leakage-fix df_merged, so the
dates match modelling/nested_cv_outer_fold_results.csv exactly -- not a
hand-drawn illustration.

Run from modelling/ (needs cv_lib.py + saved_models/df_merged_*.joblib on path):
    cd modelling && python ../draft/figures/make_thesis_diagrams.py
"""
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

NAVY = "#1a3a5c"
BLUE = "#2E6F9E"
GREEN = "#2E7D32"
ORANGE = "#B45309"
GREY = "#5b6570"
LIGHT = "#eef2f6"


def box(ax, xy, w, h, text, fc=LIGHT, ec=NAVY, fontsize=10, fontweight="normal", textcolor="#16202b"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, fontweight=fontweight, color=textcolor,
             wrap=True, zorder=3)
    return p


def arrow(ax, start, end, color=GREY, connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                         linewidth=1.4, color=color, connectionstyle=connectionstyle, zorder=1)
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Gambar 2.x -- Kerangka Pemikiran
# ---------------------------------------------------------------------------

def make_kerangka_pemikiran():
    fig, ax = plt.subplots(figsize=(8.2, 11.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.6, 14.9)
    ax.axis("off")

    ax.text(5, 14.5, "Kerangka Pemikiran", ha="center", fontsize=15, fontweight="bold", color=NAVY)

    steps = [
        ("MASALAH", "IHSG bersifat volatil dan dipengaruhi banyak faktor\n"
                     "(makro, komoditas, regional). Desain evaluasi single-split\n"
                     "rentan terhadap look-ahead bias & bias lag publikasi data makro.",
         NAVY, "#ffffff", 1.7),
        ("DATA & PRA-PEMROSESAN", "IHSG + 15 variabel kovariat (7 makro, 7 komoditas, 1 regional)\n"
                                   "2015–2025 • koreksi lag publikasi • uji ADF • transformasi stasioner",
         BLUE, "#ffffff", 1.5),
        ("TAHAP A — SELEKSI MODEL & KONFIGURASI", "4 model tree-ensemble × 21 skenario kovariat ×\n"
                                                         "2 window × 3 horizon, hyperparameter default,\n"
                                                         "walk-forward CV + embargo → 1 konfigurasi terbaik",
         BLUE, "#ffffff", 1.7),
        ("TAHAP B — OPTIMASI HYPERPARAMETER", "Optuna (TPE) dengan nested cross-validation,\n"
                                                     "hanya pada konfigurasi pemenang Tahap A",
         GREEN, "#ffffff", 1.5),
        ("INTERPRETASI SHAP", "Mengukur kontribusi tiap variabel terhadap\n"
                               "prediksi pada data out-of-sample",
         GREEN, "#ffffff", 1.3),
        ("HASIL", "Model & konfigurasi terbaik untuk prediksi IHSG,\n"
                   "beserta variabel paling berpengaruh",
         ORANGE, "#ffffff", 1.3),
    ]

    y = 13.0
    centers = []
    for title, body, fc, tc, h in steps:
        box(ax, (1.0, y - h), 8.0, h, f"{title}\n\n{body}", fc=fc, ec=fc, fontsize=9.3,
            fontweight="bold", textcolor=tc)
        centers.append((5.0, y - h))
        y = y - h - 0.55

    for i in range(len(centers) - 1):
        arrow(ax, (5.0, centers[i][1]), (5.0, centers[i][1] + 0.55))

    hy = centers[-1][1] - 0.75
    hh = 1.35
    box(ax, (0.6, hy - hh), 4.1, hh,
        "Hipotesis 1\n\nModel tree-ensemble dengan kovariat terpilih\nmemberikan MAPE lebih rendah dibanding\nbaseline autoregresif (tanpa kovariat).",
        fc="#fdf2e3", ec=ORANGE, fontsize=8, textcolor="#3a2a10")
    box(ax, (5.3, hy - hh), 4.1, hh,
        "Hipotesis 2\n\nSHAP dapat mengidentifikasi variabel\nmakro/komoditas/regional yang paling\nberpengaruh terhadap prediksi IHSG.",
        fc="#fdf2e3", ec=ORANGE, fontsize=8, textcolor="#3a2a10")
    arrow(ax, (5.0, centers[-1][1]), (2.65, hy), connectionstyle="arc3,rad=-0.15")
    arrow(ax, (5.0, centers[-1][1]), (7.35, hy), connectionstyle="arc3,rad=0.15")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gambar_2_kerangka_pemikiran.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Gambar 3.1 -- Diagram Alur / Tahapan Penelitian
# ---------------------------------------------------------------------------

def make_tahapan_penelitian():
    fig, ax = plt.subplots(figsize=(8.6, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16.5)
    ax.axis("off")
    ax.text(5, 16.1, "Diagram Alur Penelitian (Tahapan Penelitian)",
             ha="center", fontsize=14, fontweight="bold", color=NAVY)

    stages = [
        "3.1.1 Pengumpulan Data\nIHSG + 15 variabel kovariat (2015–2025)",
        "3.1.2 Pra-Pemrosesan Data\nKoreksi lag publikasi • Uji ADF • Transformasi\n(log-diff / first-diff) → df_merged",
        "3.1.3 Desain Validasi Walk-Forward CV\nExpanding window, 5 fold, embargo = horizon",
        "3.1.4 Tahap A: Seleksi Model & Konfigurasi\n4 model × 21 kovariat × 2 window × 3 horizon\n(hyperparameter default)",
        "3.1.5 Tahap B: Optimasi Hyperparameter\nOptuna (TPE), nested CV pada\nkonfigurasi pemenang Tahap A",
        "3.1.6 Interpretasi SHAP\nTreeExplainer pada fold uji\nout-of-sample terakhir",
        "3.1.7 Visualisasi & Pembahasan Hasil\nPerbandingan default vs tuned, ranking SHAP,\nanalisis per fold & rezim pasar",
        "Kesimpulan & Saran",
    ]
    colors = [BLUE, BLUE, GREY, BLUE, GREEN, GREEN, ORANGE, NAVY]

    n = len(stages)
    top = 15.1
    h = 1.55
    gap = 0.35
    centers = []
    for i, (txt, c) in enumerate(zip(stages, colors)):
        y = top - i * (h + gap)
        box(ax, (1.0, y - h), 8.0, h, txt, fc=c, ec=c, fontsize=8.8, fontweight="bold", textcolor="#ffffff")
        centers.append(y - h)
    for i in range(n - 1):
        arrow(ax, (5.0, centers[i]), (5.0, centers[i] + gap))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gambar_3_1_tahapan_penelitian.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Gambar 3.2 -- Desain Walk-Forward CV (expanding window + embargo)
# ---------------------------------------------------------------------------

def make_walkforward_cv_diagram():
    import glob
    import joblib

    sys.path.insert(0, os.path.join(OUT_DIR, "..", "..", "modelling"))
    import cv_lib as cv

    modelling_dir = os.path.join(OUT_DIR, "..", "..", "modelling")
    df_path = sorted(glob.glob(os.path.join(modelling_dir, "saved_models", "df_merged_*.joblib")), reverse=True)[0]
    df = joblib.load(df_path)
    target_ts, _ = cv.to_series(df, "IHSG")
    n = len(target_ts)
    folds = cv.expanding_window_folds(n, n_folds=5, embargo=1)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    bar_h = 0.6
    for f in folds:
        y = f["fold"]
        train_start_d = target_ts[0].start_time()
        train_end_d = target_ts[f["train_end"] - 1].end_time()
        test_start_d = target_ts[f["test_start"]].start_time()
        test_end_d = target_ts[f["test_end"] - 1].end_time()

        ax.barh(y, (train_end_d - train_start_d).days, left=0, height=bar_h,
                color=BLUE, edgecolor=NAVY, linewidth=0.8, label="Training (expanding)" if y == 0 else None)
        ax.barh(y, (test_end_d - test_start_d).days,
                left=(test_start_d - train_start_d).days, height=bar_h,
                color=ORANGE, edgecolor="#7a3c05", linewidth=0.8, label="Test (out-of-sample)" if y == 0 else None)
        embargo_days = (test_start_d - train_end_d).days
        if embargo_days > 0:
            ax.barh(y, embargo_days, left=(train_end_d - train_start_d).days, height=bar_h,
                    color="#d9d9d9", edgecolor="#999999", linewidth=0.6,
                    label="Embargo" if y == 0 else None)

        ax.text(-40, y, f"Fold {y}", ha="right", va="center", fontsize=9, fontweight="bold")
        ax.text((test_end_d - train_start_d).days + 20, y,
                 f"{test_start_d.date()} – {test_end_d.date()}", ha="left", va="center", fontsize=8, color=GREY)

    ax.set_yticks([])
    ax.set_xlabel("Hari sejak awal data (2015-02-23)", fontsize=10)
    ax.set_title("Desain Validasi Walk-Forward CV — Expanding Window dengan Embargo\n"
                 "(5 fold, embargo = horizon, contoh: H1)", fontsize=11, fontweight="bold", color=NAVY)
    ax.set_xlim(-260, 4650)
    ax.invert_yaxis()
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.22), ncol=3, fontsize=8, framealpha=0.9)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gambar_3_2_desain_walkforward_cv.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    make_kerangka_pemikiran()
    make_tahapan_penelitian()
    make_walkforward_cv_diagram()
