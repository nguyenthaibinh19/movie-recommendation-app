# src-backend/run.py
import os, time

# ✅ Đặt ENV TRƯỚC khi import torch/numpy/faiss để tránh oversubscription & xung đột OpenMP
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
# Nếu còn lỗi libiomp5md.dll ở môi trường dev, có thể bật dòng dưới (không khuyến nghị cho prod)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

t0 = time.perf_counter()
print("[run] starting…")

# Import waitress trước để đo mốc thời gian
t1 = time.perf_counter()
from waitress import serve
print(f"[run] import waitress: {time.perf_counter() - t1:.2f}s")

# Import Flask app SAU khi ENV đã set
t2 = time.perf_counter()
from app import app   # app.py sẽ lazy-load predict_torch/similar khi cần
print(f"[run] import app: {time.perf_counter() - t2:.2f}s")

print(f"[run] total import time: {time.perf_counter() - t0:.2f}s")
print("[run] serving with waitress on http://0.0.0.0:5000  (threads=1)")

# 🧰 Nếu chỉ demo local, bạn có thể đổi host='127.0.0.1'
serve(app, host="0.0.0.0", port=5000, threads=1)
