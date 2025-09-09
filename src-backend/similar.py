# src-backend/similar.py
import os, json
import numpy as np
import faiss
import pandas as pd
import os, json, numpy as np, faiss, pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.normpath(os.path.join(ROOT_DIR, ".."))
DATA_DIR = os.path.join(PROJ_ROOT, "Data")
ART_DIR  = os.path.join(DATA_DIR, "artifacts")

ITEM_EMB_PATH = os.path.join(ART_DIR, "item_emb.npy")
ID2IDX_PATH   = os.path.join(ART_DIR, "item_id_to_index.json")
IDX2ID_PATH   = os.path.join(ART_DIR, "index_to_item_id.json")
MOVIES_DAT    = os.path.join(DATA_DIR, "Dataset", "movies.dat")

ITEM_EMB = None
INDEX = None
ITEM_ID_TO_INDEX = None
INDEX_TO_ITEM_ID = None
TITLE_BY_ID = {}

def _load_titles_from_movies_dat():
    global TITLE_BY_ID
    if not os.path.exists(MOVIES_DAT): return
    rows = []
    with open(MOVIES_DAT, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
    if rows:
        df = pd.DataFrame(rows, columns=["movieId","title","genres"])
        TITLE_BY_ID = dict(zip(df["movieId"].astype(str), df["title"]))

def init_similar():
    global ITEM_EMB, INDEX, ITEM_ID_TO_INDEX, INDEX_TO_ITEM_ID
    ITEM_EMB = np.load(ITEM_EMB_PATH).astype("float32")
    with open(ID2IDX_PATH, "r", encoding="utf-8") as f:
        ITEM_ID_TO_INDEX = {str(k): int(v) for k, v in json.load(f).items()}
    with open(IDX2ID_PATH, "r", encoding="utf-8") as f:
        INDEX_TO_ITEM_ID = {str(k): str(v) for k, v in json.load(f).items()}

    em = ITEM_EMB.copy()
    faiss.normalize_L2(em)
    index = faiss.IndexFlatIP(em.shape[1])
    index.add(em)
    INDEX = index

    _load_titles_from_movies_dat()
    return True

def similar_items_by_item_id(item_id: int | str, k: int = 10):
    raw = str(item_id)
    if raw not in ITEM_ID_TO_INDEX:
        raise ValueError(f"Unknown item_id: {item_id}")
    q_idx = ITEM_ID_TO_INDEX[raw]
    q = ITEM_EMB[q_idx:q_idx+1].copy()
    faiss.normalize_L2(q)
    D, I = INDEX.search(q, k + 1)
    out = []
    for d, i in zip(D[0], I[0]):
        if int(i) == int(q_idx): continue
        iid = INDEX_TO_ITEM_ID.get(str(int(i)), None)
        if iid is None: continue
        title = TITLE_BY_ID.get(str(iid), str(iid))
        out.append({"item_id": int(iid), "title": title, "score": float(d)})
        if len(out) == k: break
    return out

SESSION_U = {}  # user_id(str) -> np.ndarray(d,)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def online_update_u(user_id: str, item_id: int, y: int, w: float = 1.0,
                    lr: float = 0.2, reg: float = 1e-3, steps: int = 3):
    """
    Cập nhật vector u từ một tương tác:
      y=1 (positive), y=0 (negative)
      w: trọng số tín hiệu (finish > like > click > dismiss)
    """
    d = ITEM_EMB.shape[1]
    u = SESSION_U.get(user_id, np.zeros(d, dtype="float32"))
    raw = str(item_id)
    idx = ITEM_ID_TO_INDEX.get(raw)
    if idx is None:
        # item không có trong mapping -> giữ nguyên u
        return u
    e = ITEM_EMB[idx]
    for _ in range(steps):
        p = _sigmoid(float(np.dot(u, e)))
        grad = (p - y) * e + 2.0 * reg * u
        u = u - lr * w * grad
    u = u.astype("float32")
    SESSION_U[user_id] = u
    return u

# Tiện ích: recommend bằng u (retrieve nhanh)
# Dùng cosine: chuẩn hoá cả u và item embedding trước khi dot
_ITEMS_NORMED = None
def _ensure_items_normed():
    global _ITEMS_NORMED
    if _ITEMS_NORMED is None:
        norms = np.linalg.norm(ITEM_EMB, axis=1, keepdims=True) + 1e-12
        _ITEMS_NORMED = ITEM_EMB / norms
    return _ITEMS_NORMED

def recommend_by_u(u_vec: np.ndarray, top_k: int = 10, exclude: list[int] | None = None):
    E = _ensure_items_normed()
    uv = u_vec / (np.linalg.norm(u_vec) + 1e-12)
    scores = E @ uv  # cosine

    if exclude:
        for iid in exclude:
            idx = ITEM_ID_TO_INDEX.get(str(iid))
            if idx is not None:
                scores[idx] = -1e9

    k = min(top_k, scores.shape[0] - 1 if exclude else top_k)
    idxs = np.argpartition(-scores, k)[:k]
    idxs = idxs[np.argsort(-scores[idxs])]
    out = []
    for i in idxs:
        iid = INDEX_TO_ITEM_ID.get(str(int(i)), str(int(i)))
        title = TITLE_BY_ID.get(str(iid), str(iid))
        out.append({"item_id": int(iid), "title": title, "score": float(scores[i])})
    return out
