import os
print(f"[📂 Using]: {__file__}")
print("[🚀 preprocess.py LOADED]")
# src-backend/preprocess.py
def bucketize_age(age_value: int) -> int:
    """
    Map tuổi thực (ví dụ 18,25,35,...) -> bucket 0..6.
    Tùy theo scheme bạn đang dùng; đây là ví dụ.
    """
    # Ví dụ scheme hiện tại trong FE:
    # 1 (Under 18), 18, 25, 35, 45, 50, 56
    if age_value < 0:  return 0
    if age_value == 1: return 0   # Under 18
    if age_value == 18: return 1
    if age_value == 25: return 2
    if age_value == 35: return 3
    if age_value == 45: return 4
    if age_value == 50: return 5
    return 6  # 56+

def encode_user_profile(raw: dict) -> dict:
    """
    Cho phép UNKNOWN -> dùng popular; không ép UNKNOWN thành Male/Female thật.
    """
    g   = raw.get('gender', 'UNKNOWN')
    age = int(raw.get('age', -1))
    occ = int(raw.get('occupation', -1))

    use_popular = (g == 'UNKNOWN') or (age == -1) or (occ == -1)

    # Giá trị sẽ bị bỏ qua nếu use_popular=True; vẫn điền để giữ schema
    gender = 0 if g == 'M' else (1 if g == 'F' else 0)
    age_bucket = 0 if age == -1 else bucketize_age(age)
    occupation = 0 if occ == -1 else int(occ)

    return {
        'gender': gender,
        'age': age_bucket,
        'occupation': occupation,
        'use_popular': use_popular
    }
