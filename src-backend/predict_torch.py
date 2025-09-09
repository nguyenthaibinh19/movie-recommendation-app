# src-backend/predict_torch.py
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ===== Paths (robust by file location) =====
import os

# Thư mục chứa file predict_torch.py
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root = src-backend/..  (lên 1 cấp)
PROJ_ROOT = os.path.normpath(os.path.join(ROOT_DIR, ".."))

# Data/
DATA_DIR = os.path.join(PROJ_ROOT, "Data")

# Data/artifacts/
ART_DIR = os.path.join(DATA_DIR, "artifacts")

ITEM_ID_TO_INDEX_JSON = os.path.join(ART_DIR, "item_id_to_index.json")
INDEX_TO_ITEM_ID_JSON = os.path.join(ART_DIR, "index_to_item_id.json")
ITEM_EMB_NPY          = os.path.join(ART_DIR, "item_emb.npy")
PTH_PATH              = os.path.join(ART_DIR, "deepfm_pytorch.pth")

# Data/Dataset/movies.dat
MOVIES_DAT            = os.path.join(DATA_DIR, "Dataset", "movies.dat")

# Data/item_id_mapping.csv (không bắt buộc cho PyTorch nhưng nên giữ)
ITEM_MAP_CSV          = os.path.join(DATA_DIR, "item_id_mapping.csv")

# Debug nhanh để chắc đã trỏ đúng
print("[PATH] ART_DIR =", ART_DIR)
print("[PATH] ITEM_ID_TO_INDEX_JSON exists:", os.path.exists(ITEM_ID_TO_INDEX_JSON))
print("[PATH] INDEX_TO_ITEM_ID_JSON exists:", os.path.exists(INDEX_TO_ITEM_ID_JSON))
print("[PATH] ITEM_EMB_NPY exists:", os.path.exists(ITEM_EMB_NPY))
print("[PATH] PTH_PATH exists:", os.path.exists(PTH_PATH))
print("[PATH] MOVIES_DAT exists:", os.path.exists(MOVIES_DAT))

# ===== Model hyperparams (sửa cho đúng như lúc train Notebook) =====
EMBED_DIM = 16
MLP_DIMS  = [128, 64]
DROPOUT   = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== DeepFM định nghĩa phải KHỚP lúc train =====
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
        # Linear
        lin_terms = [self.lin[k](x[k]) for k in self.fields]  # [B,1] list
        lin = torch.stack(lin_terms, dim=1).sum(dim=1)        # [B,1]
        # Embeddings
        embs = [self.emb[k](x[k]) for k in self.fields]       # list [B,d]
        E = torch.stack(embs, dim=1)                          # [B,F,d]
        # FM
        sum_emb = E.sum(dim=1)                                # [B,d]
        fm = 0.5 * (sum_emb.pow(2) - (E.pow(2)).sum(dim=1))   # [B,d]
        fm_logit = fm.sum(dim=1, keepdim=True)                # [B,1]
        # DNN
        dnn_in = torch.cat(embs, dim=1)                       # [B, F*d]
        dnn_hidden = self.dnn(dnn_in)                         # [B,H]
        dnn_logit = self.dnn_out(dnn_hidden)                  # [B,1]
        return lin + fm_logit + dnn_logit                     # logits [B,1]

# ===== Load mappings =====
with open(ITEM_ID_TO_INDEX_JSON, "r", encoding="utf-8") as f:
    ITEM_ID_TO_INDEX = {str(k): int(v) for k, v in json.load(f).items()}
with open(INDEX_TO_ITEM_ID_JSON, "r", encoding="utf-8") as f:
    INDEX_TO_ITEM_ID = {int(k): int(v) for k, v in json.load(f).items()}

# ===== Load movies & encode genre_first =====
rows = []
with open(MOVIES_DAT, "r", encoding="latin-1") as f:
    for line in f:
        parts = line.strip().split("::")
        if len(parts) >= 3:
            rows.append((int(parts[0]), parts[1], parts[2]))
movies_df = pd.DataFrame(rows, columns=["item_id","title","genres"])

def first_genre(s):
    return s.split("|")[0] if isinstance(s, str) and "|" in s else s
movies_df["genre_first"] = movies_df["genres"].apply(first_genre)
genre_to_index = {g:i for i, g in enumerate(sorted(movies_df["genre_first"].unique()))}
movies_df["genre_enc"] = movies_df["genre_first"].map(genre_to_index).astype(int)

# Map item_id → index (embedding index)
movies_df = movies_df[movies_df["item_id"].astype(str).isin(ITEM_ID_TO_INDEX.keys())].copy()
movies_df["item_idx"] = movies_df["item_id"].astype(str).map(ITEM_ID_TO_INDEX).astype(int)
movies_df = movies_df[["item_id","item_idx","genre_enc","title"]].drop_duplicates()

NUM_ITEMS  = len(ITEM_ID_TO_INDEX)
NUM_GENRES = len(genre_to_index)

# ===== Build model & load weights =====
FIELD_DIMS = {
    "gender": 2,
    "age": 7,
    "occupation": 21,
    "item_id": NUM_ITEMS,
    "genre": NUM_GENRES
}
_model = DeepFM(FIELD_DIMS, embed_dim=EMBED_DIM, mlp_dims=MLP_DIMS, dropout=DROPOUT).to(DEVICE)
_state = torch.load(PTH_PATH, map_location=DEVICE)
_model.load_state_dict(_state)
_model.eval()

@torch.no_grad()
def recommend_movies(user_profile: dict, top_n: int = 10):
    """
    user_profile = { 'gender':int(0/1), 'age':int(0..6), 'occupation':int(0..20) }
    Trả: list [[item_id, title, score], ...]
    """
    # Chuẩn bị batch tất cả items
    n = len(movies_df)
    g = torch.full((n,), int(user_profile["gender"]), dtype=torch.long, device=DEVICE)
    a = torch.full((n,), int(user_profile["age"]), dtype=torch.long, device=DEVICE)
    o = torch.full((n,), int(user_profile["occupation"]), dtype=torch.long, device=DEVICE)
    item_idx = torch.tensor(movies_df["item_idx"].values, dtype=torch.long, device=DEVICE)
    genre    = torch.tensor(movies_df["genre_enc"].values, dtype=torch.long, device=DEVICE)

    batch = {"gender": g, "age": a, "occupation": o, "item_id": item_idx, "genre": genre}
    logits = _model(batch).squeeze(1)      # [n]
    probs  = torch.sigmoid(logits).cpu().numpy()

    df = movies_df.copy()
    df["score"] = probs
    top = df.sort_values("score", ascending=False).head(top_n)

    # ——> QUAN TRỌNG: trả kèm item_id để FE dùng /similar_items
    return top[["item_id","title","score"]].values.tolist()
