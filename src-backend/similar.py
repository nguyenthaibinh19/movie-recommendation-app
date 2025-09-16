# src-backend/similar.py
import os, json
import numpy as np
import faiss
import pandas as pd
from collections import defaultdict, deque
import time

USER_STATE = {}  # user_id -> dict(u, history, genre_counts, last_ts)
GENRES_BY_ID = {}
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


def _parse_genres_of(iid_str: str):
    # Đọc từ MOVIES_DAT đã load (TITLE_BY_ID hiện có; ta cần thêm GENRES_BY_ID)
    # Hãy tạo GENRES_BY_ID khi _load_titles_from_movies_dat() chạy:
    return GENRES_BY_ID.get(iid_str, "")

def update_user_profile(user_id: str, item_id: int, ev: str):
    """
    Ghi lại lịch sử & cập nhật đếm thể loại để tóm tắt gu.
    Không thay thế online_update_u (u-vector), mà bổ sung metadata cho Lantern.
    """
    st = USER_STATE.get(user_id)
    if st is None:
        st = {
            "history": deque(maxlen=200),          # (ts, item_id, ev, title)
            "genre_counts": defaultdict(int),      # genre_name -> count
            "last_ts": 0
        }
        USER_STATE[user_id] = st

    ts = int(time.time())
    iid = str(item_id)
    title = TITLE_BY_ID.get(iid, iid)
    st["history"].append((ts, int(item_id), ev, title))
    st["last_ts"] = ts

    # Với sự kiện tích cực mới đếm thể loại (like/finish/click)
    if ev in ("like", "finish", "click"):
        genres = _parse_genres_of(iid)  # ví dụ: "Action|Adventure|Sci-Fi"
        if isinstance(genres, str) and genres:
            for g in genres.split("|"):
                st["genre_counts"][g] += 1

def summarize_user(user_id: str, top_k_genres: int = 5, top_k_recent: int = 5):
    st = USER_STATE.get(user_id, {"history":[], "genre_counts":{}})
    # Top genres
    gc = st.get("genre_counts", {})
    top_genres = sorted(gc.items(), key=lambda x: -x[1])[:top_k_genres]

    # Recent positives (like/finish trước, rồi click)
    hist = list(st.get("history", []))
    pos = [h for h in reversed(hist) if h[2] in ("finish", "like")]  # ưu tiên finish/like
    if len(pos) < top_k_recent:
        pos += [h for h in reversed(hist) if h[2] == "click"]
    pos = pos[:top_k_recent]

    # Persona gợi ý đơn giản = top-1 genre (hoặc top-2)
    persona = " · ".join([g for g,_ in top_genres[:2]]) if top_genres else "Unknown"

    recent_titles = [{"ts": h[0], "item_id": h[1], "event": h[2], "title": h[3]} for h in pos]
    return {"persona": persona, "top_genres": top_genres, "recent": recent_titles}

def _load_titles_from_movies_dat():
    global TITLE_BY_ID, GENRES_BY_ID
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
        GENRES_BY_ID = dict(zip(df["movieId"].astype(str), df["genres"]))

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
