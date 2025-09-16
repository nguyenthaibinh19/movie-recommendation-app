# src-backend/predict_torch.py  (refactor: lazy-load & cache)
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time

# ===== Paths =====
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.normpath(os.path.join(ROOT_DIR, ".."))
DATA_DIR = os.path.join(PROJ_ROOT, "Data")
ART_DIR  = os.path.join(DATA_DIR, "artifacts")
POPULAR_JSON = os.path.join(ART_DIR, "popular_items.json")
_POPULAR_ITEMS = None  # list[int]

ITEM_ID_TO_INDEX_JSON = os.path.join(ART_DIR, "item_id_to_index.json")
INDEX_TO_ITEM_ID_JSON = os.path.join(ART_DIR, "index_to_item_id.json")
ITEM_EMB_NPY          = os.path.join(ART_DIR, "item_emb.npy")
PTH_PATH              = os.path.join(ART_DIR, "deepfm_pytorch.pth")
MOVIES_DAT            = os.path.join(DATA_DIR, "Dataset", "movies.dat")
ITEM_MAP_CSV          = os.path.join(DATA_DIR, "item_id_mapping.csv")

# Giới hạn threads để tránh khởi động chậm do oversubscription (đặc biệt trên Windows)
try:
    torch.set_num_threads(2)
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# ===== Hyperparams (khớp lúc train) =====
EMBED_DIM = 16
MLP_DIMS  = [128, 64]
DROPOUT   = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== DeepFM (giữ nguyên kiến trúc) =====
class DeepFM(nn.Module):
    def __init__(self, field_dims: dict, embed_dim=16, mlp_dims=[128,64], dropout=0.2):
        super().__init__()
        self.fields = list(field_dims.keys())
        self.emb = nn.ModuleDict({k: nn.Embedding(field_dims[k], embed_dim) for k in self.fields})
        self.lin = nn.ModuleDict({k: nn.Embedding(field_dims[k], 1) for k in self.fields})

        in_dim = embed_dim * len(self.fields)
        layers, d = [], in_dim
        for h in mlp_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.dnn = nn.Sequential(*layers)
        self.dnn_out = nn.Linear(d, 1)

        self._init_weights()

    def _init_weights(self):
        for emb in self.emb.values():
            nn.init.xavier_uniform_(emb.weight.data)
        for l in self.lin.values():
            nn.init.zeros_(l.weight.data)

    def forward(self, x: dict):
        lin_terms = [self.lin[k](x[k]) for k in self.fields]   # [B,1] list
        lin = torch.stack(lin_terms, dim=1).sum(dim=1)         # [B,1]

        embs = [self.emb[k](x[k]) for k in self.fields]        # list [B,d]
        E = torch.stack(embs, dim=1)                           # [B,F,d]
        sum_emb = E.sum(dim=1)                                 # [B,d]
        fm = 0.5 * (sum_emb.pow(2) - (E.pow(2)).sum(dim=1))    # [B,d]
        fm_logit = fm.sum(dim=1, keepdim=True)                 # [B,1]

        dnn_in = torch.cat(embs, dim=1)                        # [B, F*d]
        dnn_hidden = self.dnn(dnn_in)                          # [B,H]
        dnn_logit = self.dnn_out(dnn_hidden)                   # [B,1]

        return lin + fm_logit + dnn_logit                      # [B,1]

# ===== Lazy caches =====
_ITEM_ID_TO_INDEX = None
_INDEX_TO_ITEM_ID = None
_MOVIES_DF        = None          # pandas DataFrame: item_id, item_idx, genre_enc, title
_FIELD_DIMS       = None
_MODEL            = None
_ITEM_IDX_T       = None          # torch.Tensor on DEVICE
_GENRE_T          = None          # torch.Tensor on DEVICE

def _load_popular_once():
    global _POPULAR_ITEMS
    if _POPULAR_ITEMS is not None:
        return
    if os.path.exists(POPULAR_JSON):
        with open(POPULAR_JSON, "r", encoding="utf-8") as f:
            _POPULAR_ITEMS = json.load(f)  # [item_id, ...]
    else:
        # fallback siêu đơn giản nếu chưa có file
        _build_movies_df_once()
        _POPULAR_ITEMS = _MOVIES_DF["item_id"].tolist()

def recommend_popular(top_n: int = 10, exclude=None):
    _build_movies_df_once()
    _load_popular_once()
    exclude = set(exclude or [])
    selected = []
    for iid in _POPULAR_ITEMS:
        if iid in exclude: 
            continue
        selected.append(iid)
        if len(selected) >= top_n:
            break
    df = _MOVIES_DF[_MOVIES_DF["item_id"].isin(selected)].copy()
    order = {iid:i for i, iid in enumerate(selected)}
    df["__o"] = df["item_id"].map(order)
    df["score"] = 1.0
    df = df.sort_values("__o").drop(columns="__o")
    return df[["item_id","title","score"]].values.tolist()

def _load_mappings_once():
    """Lazy: load JSON mapping item_id <-> index."""
    global _ITEM_ID_TO_INDEX, _INDEX_TO_ITEM_ID
    if _ITEM_ID_TO_INDEX is None or _INDEX_TO_ITEM_ID is None:
        with open(ITEM_ID_TO_INDEX_JSON, "r", encoding="utf-8") as f:
            _ITEM_ID_TO_INDEX = {str(k): int(v) for k, v in json.load(f).items()}
        with open(INDEX_TO_ITEM_ID_JSON, "r", encoding="utf-8") as f:
            _INDEX_TO_ITEM_ID = {int(k): int(v) for k, v in json.load(f).items()}

def _build_movies_df_once():
    """Lazy: đọc movies.dat → build DataFrame + encode genre_first + map item_idx."""
    global _MOVIES_DF, _FIELD_DIMS, _ITEM_IDX_T, _GENRE_T
    if _MOVIES_DF is not None:
        return

    _load_mappings_once()

    rows = []
    with open(MOVIES_DAT, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) >= 3:
                rows.append((int(parts[0]), parts[1], parts[2]))
    df = pd.DataFrame(rows, columns=["item_id","title","genres"])

    def first_genre(s):
        return s.split("|")[0] if isinstance(s, str) and "|" in s else s

    df["genre_first"] = df["genres"].apply(first_genre)
    genre_to_index = {g:i for i, g in enumerate(sorted(df["genre_first"].unique()))}
    df["genre_enc"] = df["genre_first"].map(genre_to_index).astype(int)

    # Chỉ giữ các item có trong mapping
    df = df[df["item_id"].astype(str).isin(_ITEM_ID_TO_INDEX.keys())].copy()
    df["item_idx"] = df["item_id"].astype(str).map(_ITEM_ID_TO_INDEX).astype(int)
    df = df[["item_id","item_idx","genre_enc","title"]].drop_duplicates()

    _MOVIES_DF = df

    # Chuẩn bị FIELD_DIMS theo mapping đã có
    NUM_ITEMS  = len(_ITEM_ID_TO_INDEX)
    NUM_GENRES = len(genre_to_index)
    _FIELD_DIMS = {
        "gender": 2,
        "age": 7,
        "occupation": 21,
        "item_id": NUM_ITEMS,
        "genre": NUM_GENRES
    }

    # Cache tensor item/genre trên DEVICE (dùng lại cho mọi request)
    _ITEM_IDX_T = torch.tensor(_MOVIES_DF["item_idx"].values, dtype=torch.long, device=DEVICE)
    _GENRE_T    = torch.tensor(_MOVIES_DF["genre_enc"].values, dtype=torch.long, device=DEVICE)

def _get_model():
    """Lazy: khởi tạo model + load weights lần đầu gọi."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    _build_movies_df_once()  # đảm bảo FIELD_DIMS sẵn
    model = DeepFM(_FIELD_DIMS, embed_dim=EMBED_DIM, mlp_dims=MLP_DIMS, dropout=DROPOUT).to(DEVICE)

    t0 = time.perf_counter()
    # An toàn: thử weights_only=True (PyTorch >= 2.4); nếu không, fallback
    try:
        state = torch.load(PTH_PATH, map_location=DEVICE, weights_only=True)  # may raise if older torch
    except TypeError:
        state = torch.load(PTH_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    dt = time.perf_counter() - t0
    print(f"[predict_torch] Model loaded in {dt:.2f}s on {DEVICE}")

    _MODEL = model
    return _MODEL

@torch.inference_mode()
def recommend_movies(user_profile: dict, top_n: int = 10):
    """
    user_profile = { 'gender': int(0/1), 'age': int(0..6), 'occupation': int(0..20) }
    Return: list [[item_id, title, score], ...]
    """
    # Bảo đảm lazy-caches đã sẵn sàng
    _build_movies_df_once()
    model = _get_model()

    n = _MOVIES_DF.shape[0]

    # Broadcast user features cho toàn bộ n items (dùng full để rõ ràng)
    g = torch.full((n,), int(user_profile["gender"]), dtype=torch.long, device=DEVICE)
    a = torch.full((n,), int(user_profile["age"]), dtype=torch.long, device=DEVICE)
    o = torch.full((n,), int(user_profile["occupation"]), dtype=torch.long, device=DEVICE)

    batch = {
        "gender": g,
        "age": a,
        "occupation": o,
        "item_id": _ITEM_IDX_T,   # cached
        "genre": _GENRE_T         # cached
    }

    logits = model(batch).squeeze(1)          # [n]
    probs  = torch.sigmoid(logits).detach().cpu().numpy()

    df = _MOVIES_DF.copy()
    df["score"] = probs
    top = df.nlargest(top_n, "score")

    # Trả [[item_id, title, score], ...]
    return top[["item_id","title","score"]].values.tolist()
