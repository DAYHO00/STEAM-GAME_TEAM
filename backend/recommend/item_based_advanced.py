# item_based_advanced.py
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from collections import Counter
from functools import lru_cache

# ============================================================
# 0. Load Data
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "processed" / "train_6cols.csv"
df = pd.read_csv(DATA_PATH)
print(f"item-based(advanced) 데이터 로드 완료. shape={df.shape}")

# ============================================================
# 1. Global Initialization
# ============================================================
USER_IDS = df["user_id"].unique()
GAME_TITLES = df["title"].unique()

USER2IDX = {u: i for i, u in enumerate(USER_IDS)}
GAME2IDX = {g: i for i, g in enumerate(GAME_TITLES)}
IDX2GAME = {i: t for t, i in GAME2IDX.items()}

values = df["is_recommended"].values
rows = df["user_id"].map(USER2IDX).values
cols = df["title"].map(GAME2IDX).values

R = csr_matrix((values, (rows, cols)),
               shape=(len(USER_IDS), len(GAME_TITLES)))

N_USERS, N_GAMES = R.shape

R_T = R.T.tocsr()
ITEM_USERS = [np.sort(R_T[i].indices) for i in range(N_GAMES)]
ITEM_USER_LEN = np.array([len(u) for u in ITEM_USERS])
SQRT_ITEM_USER_LEN = np.sqrt(ITEM_USER_LEN)

# Soft IDF
IDF = np.log(1 + N_USERS / (1 + ITEM_USER_LEN)) ** 0.15

# ============================================================
# 2. Hyper-parameters
# ============================================================
BETA = 1.5
MIN_INTERSECTION = 0.5
MAX_CANDIDATES = 100
MAX_RATED_ITEMS = 50
POPULARITY_CAP = 2000

# ============================================================
# 3. Two-pointer Intersection
# ============================================================
def fast_intersection_size(a, b):
    i = j = cnt = 0
    la, lb = len(a), len(b)
    while i < la and j < lb:
        if a[i] == b[j]:
            cnt += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return cnt

# ============================================================
# 4. item-item similarity (advanced)
# ============================================================
@lru_cache(maxsize=300_000)
def item_similarity_advanced(i_idx, j_idx):
    """
    원래 수식:
    sim(P_i, P_j) = min(|U_i ∩ U_j| / BETA, 1) *
                     ( |U_i^P ∩ U_j^P| * N_USERS ) / ( |U_i^P| * |U_j^P| + EPS )
    Where:
      - U_i : 게임 i를 평가한 유저 집합
      - U_i^P : 게임 i를 긍정적으로 평가한 유저 집합
      - N_USERS : 전체 유저 수
      - EPS : 분모 0 방지용 작은 숫자
    """
    users_i = ITEM_USERS[i_idx]
    users_j = ITEM_USERS[j_idx]

    inter_cnt = fast_intersection_size(users_i, users_j)
    if inter_cnt < MIN_INTERSECTION:
        return 0.0

    # positive-only term 완화
    pos_inter = inter_cnt  # 간단히 inter_cnt 기반 사용
    size_i_pos = len(users_i)
    size_j_pos = len(users_j)
    EPS = 1e-6
    pos_term = (pos_inter * N_USERS + 1) / (size_i_pos * size_j_pos + EPS)

    min_factor = min(inter_cnt / BETA, 1.0)
    sim = float(min_factor * pos_term * (IDF[i_idx] + IDF[j_idx]) / 2)  # IDF 가중치
    return sim

# ============================================================
# 5. 예측값 계산
# ============================================================
def predict_item_scores_advanced(user_idx, candidate_items, rated_items):
    """
    candidate_items: 추천 후보 아이템 리스트
    rated_items: 사용자가 평가한 아이템 리스트
    return: {item_idx: score}
    """
    scores = {}
    for item in candidate_items:
        s = 0.0
        for r in rated_items:
            sim = item_similarity_advanced(item, r)
            if sim > 0:
                s += sim  # 단순 합으로 변경 (정규화 제거)
        scores[item] = s
    return scores

# ============================================================
# 6. Top-K
# ============================================================
def top_k_recommend(scores_dict, k=5):
    sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:k]

# ============================================================
# 7. recommend_by_item_advanced
# ============================================================
def recommend_by_item_advanced(user_id: int):
    if user_id not in USER2IDX:
        return {"type": "item_based", "input_user_id": user_id, "result": [], "message": "사용자가 존재하지 않습니다."}

    u_idx = USER2IDX[user_id]
    rated_items = R[u_idx].indices
    rated_set = set(rated_items)
    if len(rated_items) > MAX_RATED_ITEMS:
        rated_items = rated_items[:MAX_RATED_ITEMS]
        rated_set = set(rated_items)

    # 후보 수집
    common_counter = Counter()
    for item in rated_items:
        users = ITEM_USERS[item]
        if len(users) > POPULARITY_CAP:
            users = users[:POPULARITY_CAP]
        for u in users:
            for g in R[u].indices:
                if ITEM_USER_LEN[g] == 0:
                    continue
                common_counter[g] += 1 / np.sqrt(1 + ITEM_USER_LEN[g])

    candidate_items = [g for g, _ in common_counter.most_common(MAX_CANDIDATES) if g not in rated_set]
    if not candidate_items:
        return {"type": "item_based", "input_user_id": user_id, "result": [], "message": "추천 후보가 없습니다."}

    # 점수 계산
    raw_scores = predict_item_scores_advanced(u_idx, candidate_items, rated_items)

    # Min-Max Scaling (fallback: raw_scores 그대로)
    vals = list(raw_scores.values())
    min_s, max_s = min(vals), max(vals)
    scaled_scores = {}
    for item, s in raw_scores.items():
        if max_s == min_s:
            scaled_scores[item] = s  # 모두 0이면 그대로
        else:
            scaled_scores[item] = (s - min_s) / (max_s - min_s)

    top_items = top_k_recommend(scaled_scores, k=5)

    scores_by_title = {IDX2GAME[i]: float(s) for i, s in scaled_scores.items()}

    return {
        "type": "item_based",
        "input_user_id": user_id,
        "scores": scores_by_title,  # 전체 후보 점수
        "result": [{"title": IDX2GAME[i], "sim": round(float(s), 5)} for i, s in top_items]
    }

# 호환성
recommend_by_item = recommend_by_item_advanced
