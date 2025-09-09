# src-backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS

from preprocess import encode_user_profile
from predict_torch import recommend_movies  # <— dùng PyTorch thay TF

# (Sau khi chạy xong Similar ở bước 4–5, bạn import thêm 2 dòng dưới)
from similar import init_similar, similar_items_by_item_id, online_update_u, recommend_by_u

app = Flask(__name__)
CORS(app)

# (Sau bước 5 mới bật init Similar)
try:
    init_similar()
    print("[✅ Similar] FAISS index ready")
except Exception as e:
    print("[⚠️ Similar init failed]", e)

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        raw_data = request.get_json()
        user_profile = encode_user_profile(raw_data)
        print("[🐾 Encoded user profile]:", user_profile)
        recommendations = recommend_movies(user_profile)  # [[item_id, title, score], ...]
        print("[🎬 Top recommendations]:", recommendations[:3])
        return jsonify({"recommendations": recommendations}), 200
    except Exception as e:
        print("[❌ Backend error]:", e)
        return jsonify({"error": str(e)}), 400

# (Bật 2 endpoint sau khi tạo similar.py)
@app.route("/similar_items", methods=["GET"])
def similar_items():
    try:
        item_id = request.args.get("item_id", type=int)
        k = request.args.get("k", default=10, type=int)
        sims = similar_items_by_item_id(item_id, k=k)
        return jsonify({"item_id": item_id, "similar": sims}), 200
    except Exception as e:
        print("[❌ /similar_items]", e)
        return jsonify({"error": str(e)}), 400


@app.route("/event", methods=["POST"])
def event():
    try:
        data = request.get_json()
        user_id = str(data.get("user_id", "guest"))
        item_id = int(data["item_id"])
        ev = data.get("event", "click")

        # mapping event -> (label, weight)
        y, w = (1, 0.5)
        if ev == "like":    y, w = 1, 1.0
        if ev == "finish":  y, w = 1, 1.2
        if ev == "dismiss": y, w = 0, 0.3

        u = online_update_u(user_id, item_id, y=y, w=w)
        recs = recommend_by_u(u, top_k=10, exclude=[item_id])
        return jsonify({"ok": True, "recommendations": recs}), 200
    except Exception as e:
        print("[❌ /event]", e)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
