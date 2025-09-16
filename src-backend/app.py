# src-backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np  # <-- THÊM DÒNG NÀY

from preprocess import encode_user_profile
from predict_torch import recommend_movies, recommend_popular

app = Flask(__name__)
CORS(app)

# ---- Lazy import FAISS-based 'similar' only when needed ----
_sim = None
def _sim_mod():
    global _sim
    if _sim is None:
        import similar as _S  # import sau khi ENV đã được set từ run.py
        try:
            _S.init_similar()
            print("[✅ Similar] index ready")
        except Exception as e:
            print("[⚠️ Similar init failed]", e)
        _sim = _S
    return _sim
# ------------------------------------------------------------

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        raw_data = request.get_json()
        user_profile = encode_user_profile(raw_data)

        # Nếu có UNKNOWN ở bất kỳ field -> dùng popular
        if user_profile.get("use_popular", False):
            recs = recommend_popular(top_n=10)
            return jsonify({"recommendations": recs, "source": "popular"}), 200

        recommendations = recommend_movies(user_profile)  # DeepFM
        return jsonify({"recommendations": recommendations, "source": "model"}), 200
    except Exception as e:
        print("[❌ /recommend]", e)
        return jsonify({"error": str(e)}), 400


@app.route("/similar_items", methods=["GET"])
def similar_items():
    try:
        item_id = request.args.get("item_id", type=int)
        k = request.args.get("k", default=10, type=int)
        sim = _sim_mod()
        sims = sim.similar_items_by_item_id(item_id, k=k)
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

        y, w = (1, 0.5)
        if ev == "like":    y, w = 1, 1.0
        if ev == "finish":  y, w = 1, 1.2
        if ev == "dismiss": y, w = 0, 0.3

        sim = _sim_mod()
        sim.update_user_profile(user_id, item_id, ev)
        # cập nhật u-vector nhanh (đã có sẵn)
        u = sim.online_update_u(user_id, item_id, y=y, w=w)
        recs = sim.recommend_by_u(u, top_k=10, exclude=[item_id])
        return jsonify({"ok": True, "recommendations": recs}), 200
    except Exception as e:
        print("[❌ /event]", e)
        return jsonify({"error": str(e)}), 400

@app.route("/lantern/profile", methods=["GET"])
def lantern_profile():
    try:
        user_id = request.args.get("user_id", default="guest", type=str)
        sim = _sim_mod()
        summary = sim.summarize_user(user_id, top_k_genres=5, top_k_recent=6)
        # (tùy chọn) trả thêm norm(u) để biết đã học đủ chưa
        u = sim.SESSION_U.get(user_id)
        u_norm = float(np.linalg.norm(u)) if u is not None else 0.0
        return jsonify({"user_id": user_id, "u_norm": u_norm, **summary}), 200
    except Exception as e:
        print("[❌ /lantern/profile]", e)
        return jsonify({"error": str(e)}), 400

@app.route("/lantern/recommend", methods=["POST"])
def lantern_recommend():
    try:
        data = request.get_json(force=True) or {}
        user_id = str(data.get("user_id", "guest"))
        exclude = data.get("exclude", [])  # danh sách item_id muốn loại

        sim = _sim_mod()
        u = sim.SESSION_U.get(user_id)

        # Nếu chưa có u hoặc tín hiệu còn yếu -> popular
        TAU = 0.2
        if u is None or float(np.linalg.norm(u)) < TAU:
            recs = recommend_popular(top_n=10, exclude=exclude)
            return jsonify({"recommendations": recs, "note": "popular_fallback"}), 200

        # Tín hiệu đủ mạnh -> Lantern thuần
        recs = sim.recommend_by_u(u, top_k=10, exclude=exclude)
        return jsonify({"recommendations": recs}), 200
    except Exception as e:
        print("[❌ /lantern/recommend]", e)
        return jsonify({"error": str(e)}), 400

@app.route("/lantern/reset", methods=["POST"])
def lantern_reset():
    try:
        data = request.get_json(force=True) or {}
        user_id = str(data.get("user_id", "guest"))
        sim = _sim_mod()
        # reset u
        if user_id in sim.SESSION_U:
            del sim.SESSION_U[user_id]
        # reset state
        if user_id in sim.USER_STATE:
            del sim.USER_STATE[user_id]
        return jsonify({"ok": True}), 200
    except Exception as e:
        print("[❌ /lantern/reset]", e)
        return jsonify({"error": str(e)}), 400


# KHÔNG có if __name__ == "__main__" ở đây. Dùng run.py để chạy.
