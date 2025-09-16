# src-backend/run.py
import os

# ✅ Đặt ENV TRƯỚC khi import torch/numpy/faiss
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
# Nếu vẫn dính libiomp5md.dll khi DEV, mở tạm dòng dưới (không khuyến nghị cho prod):
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from waitress import serve
from app import app  # import SAU khi ENV đã set

serve(app, host="0.0.0.0", port=5000, threads=1)
