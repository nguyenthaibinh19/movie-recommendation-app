"""Encode profile fields exactly as in Data/Deep_FM_Pytorch.ipynb."""
AGE_CODES = (1, 18, 25, 35, 45, 50, 56)


def bucketize_age(age_value: int) -> int:
    if age_value not in AGE_CODES:
        raise ValueError('age must be a MovieLens age code: 1, 18, 25, 35, 45, 50, 56')
    return AGE_CODES.index(age_value)


def encode_user_profile(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError('Expected a JSON object')
    gender = raw.get('gender', 'UNKNOWN')
    age = int(raw.get('age', -1))
    occupation = int(raw.get('occupation', -1))
    if gender not in ('M', 'F', 'UNKNOWN'):
        raise ValueError('gender must be M, F or UNKNOWN')
    if occupation not in range(-1, 21):
        raise ValueError('occupation must be -1 or 0..20')
    age_bucket = 0 if age == -1 else bucketize_age(age)
    return {
        'gender': {'F': 0, 'M': 1, 'UNKNOWN': 0}[gender],
        'age': age_bucket,
        'occupation': max(occupation, 0),
        'use_popular': gender == 'UNKNOWN' or age == -1 or occupation == -1,
    }
