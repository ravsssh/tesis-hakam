#!/bin/bash
# Setup environment baru untuk tesis-hakam
# Jalankan: bash setup_env.sh

ENV_NAME="tesis_hakam"

echo "======================================================"
echo "  Membuat conda environment: $ENV_NAME"
echo "======================================================"

# 1. Buat env baru
conda create -n $ENV_NAME python=3.11 -y

# 2. Activate dan install
conda run -n $ENV_NAME pip install "numpy==1.26.4"

echo ""
echo "[1/6] numpy 1.26.4 terinstall"

conda run -n $ENV_NAME pip install \
    pandas \
    scipy \
    scikit-learn \
    statsmodels \
    matplotlib \
    joblib \
    tqdm

echo "[2/6] Core scientific stack terinstall"

# darts tanpa torch -- darts 0.30+ torch adalah opsional
conda run -n $ENV_NAME pip install "darts"

# Hapus torch/pytorch-lightning jika ikut terinstall (tidak dibutuhkan)
conda run -n $ENV_NAME pip uninstall pytorch-lightning torch torchvision torchaudio \
    pytorch-forecast-lightning -y 2>/dev/null || true

echo "[3/6] darts terinstall (tanpa torch)"

conda run -n $ENV_NAME pip install xgboost lightgbm

echo "[4/6] xgboost & lightgbm terinstall"

conda run -n $ENV_NAME pip install optuna shap

echo "[5/6] optuna & shap terinstall"

conda run -n $ENV_NAME pip install \
    jupyter \
    ipykernel \
    notebook \
    nbconvert \
    nbformat \
    python-docx

# Daftarkan kernel ke Jupyter
conda run -n $ENV_NAME python -m ipykernel install \
    --user \
    --name $ENV_NAME \
    --display-name "Python (tesis_hakam)"

echo "[6/6] Jupyter kernel terdaftar"

echo ""
echo "======================================================"
echo "  SELESAI!"
echo ""
echo "  Cara pakai:"
echo "  1. Di terminal  : conda activate $ENV_NAME"
echo "  2. Di notebook  : pilih kernel 'Python (tesis_hakam)'"
echo "======================================================"
