# export_item_embeddings.py
import tensorflow as tf, numpy as np, pandas as pd, json, os

MODEL_DIR = "Data/deepfm_model"   # giữ nguyên path bạn đang load trong predict.py
MAP_CSV   = "Data/item_id_mapping.csv"
OUT_DIR   = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

print("[*] Loading model...")
model = tf.keras.models.load_model(MODEL_DIR)

print("[*] Loading item_id mapping...")
map_df = pd.read_csv(MAP_CSV)
item_id_to_index = dict(zip(map_df["item_id"], map_df["index"]))
num_items = max(item_id_to_index.values()) + 1

# Tìm lớp Embedding có shape[0] == num_items
emb_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Embedding)]
item_emb_layer = None
for l in emb_layers:
    w = l.get_weights()[0]
    if w.shape[0] >= num_items:
        item_emb_layer = l
        break
assert item_emb_layer is not None, "Không tìm thấy Embedding cho item_id!"

W = item_emb_layer.get_weights()[0][:num_items].astype("float32")  # [num_items, d]
np.save(f"{OUT_DIR}/item_emb.npy", W)

# Lưu mapping id<->index để dùng ở server
with open(f"{OUT_DIR}/item_id_to_index.json", "w") as f:
    json.dump({int(k): int(v) for k, v in item_id_to_index.items()}, f)

print("[✓] Saved artifacts to artifacts/item_emb.npy and item_id_to_index.json")
