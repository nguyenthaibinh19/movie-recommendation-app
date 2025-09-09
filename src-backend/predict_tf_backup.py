import tensorflow as tf
import numpy as np
import pandas as pd

# Load model đã huấn luyện
model = tf.keras.models.load_model('Data/deepfm_model')
print("[🧠 Model Input Names]:", model.input_names)

# Load mapping item_id → index
item_map_df = pd.read_csv('Data/item_id_mapping.csv')
item_id_to_index = dict(zip(item_map_df["item_id"], item_map_df["index"]))

# Load danh sách phim
movies_df = pd.read_csv('Data/Dataset/movies.dat', sep='::', engine='python',
                        names=["item_id", "title", "genres"], encoding='latin-1')
movies_df["genre"] = movies_df["genres"].apply(lambda x: x.split("|")[0])
genre_to_index = {genre: idx for idx, genre in enumerate(sorted(movies_df["genre"].unique()))}
movies_df["genre"] = movies_df["genre"].map(genre_to_index)

# Ánh xạ item_id sang chỉ mục đúng như lúc train
movies_df = movies_df[movies_df["item_id"].isin(item_id_to_index)]
movies_df["item_id"] = movies_df["item_id"].map(item_id_to_index)

movies_df = movies_df[["item_id", "genre", "title"]].drop_duplicates()
# Tạo batch input
def recommend_movies(user_profile, top_n=10):
    n_items = len(movies_df)
    user_batch = {
        "gender":     np.array([user_profile["gender"]] * n_items),
        "age":        np.array([user_profile["age"]] * n_items),
        "occupation": np.array([user_profile["occupation"]] * n_items),
        "item_id":    movies_df["item_id"].values,
        "genre":      movies_df["genre"].values,
    }
# Dự đoán
    preds = model.predict(user_batch, verbose=0)
    movies_df["score"] = preds.flatten()
#Trả kết quả
    top_movies = movies_df.sort_values("score", ascending=False).head(top_n)
    return top_movies[["title", "score"]].values.tolist()
