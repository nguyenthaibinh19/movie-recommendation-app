# Data/build_popular.py
import os, json, math
from collections import Counter, defaultdict

PROJ_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(PROJ_ROOT, "Data", "Dataset")
ART_DIR     = os.path.join(PROJ_ROOT, "Data", "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

RATINGS_DAT = os.path.join(DATASET_DIR, "ratings.dat")  # format: UserID::MovieID::Rating::Timestamp
OUT_JSON    = os.path.join(ART_DIR, "popular_items.json")

cnt = Counter()
sum_rate = defaultdict(float)
n_rate = defaultdict(int)

with open(RATINGS_DAT, "r", encoding="latin-1") as f:
    for line in f:
        parts = line.strip().split("::")
        if len(parts) >= 3:
            mid = int(parts[1])
            r   = float(parts[2])
            cnt[mid] += 1
            sum_rate[mid] += r
            n_rate[mid]  += 1

# điểm đơn giản: count (có thể xếp theo mean như bonus nhẹ)
scored = []
for mid, c in cnt.items():
    mean = sum_rate[mid] / max(1, n_rate[mid])
    score = c + 0.1 * (mean - 3.0)  # bonus rất nhỏ theo chất lượng
    scored.append((score, mid))

scored.sort(reverse=True)
top_ids = [mid for _, mid in scored[:2000]]  # lấy nhiều chút để đủ exclude

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(top_ids, f)

print(f"[OK] Wrote {len(top_ids)} popular item ids -> {OUT_JSON}")
