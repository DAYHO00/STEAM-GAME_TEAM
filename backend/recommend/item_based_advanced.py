import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from collections import Counter
from functools import lru_cache

# 0. Load Data
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "processed" / "train_6cols.csv"
df = pd.read_csv(DATA_PATH)
print(f"item-based(advanced) 데이터 로드 완료. shape={df.shape}")

# 1. Global Initialization
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
# sorted user lists per item for two-pointer intersection
ITEM_USERS = [np.sort(R_T[i].indices) for i in range(N_GAMES)]

ITEM_USER_LEN = np.array([len(u) for u in ITEM_USERS])
SQRT_ITEM_USER_LEN = np.sqrt(ITEM_USER_LEN)

IDF = np.log(1 + N_USERS / (1 + ITEM_USER_LEN))

# 2. Hyper-parameters
BETA = 5
MIN_INTERSECTION = 2
MAX_CANDIDATES = 50
MAX_RATED_ITEMS = 100
POPULARITY_CAP = 500   # 지나치게 인기 많은 게임 제외

# 3. Two-pointer Intersection
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

# 4. item-item similarity (advanced)
@lru_cache(maxsize=300_000)
def item_similarity_advanced(i_idx, j_idx):
    """
    아래의 식을 이용하여 유사도 계산함

        sim(P_i, P_j) = min(|U_i ∩ U_j| / BETA, 1) *
                      ( |U_i^P ∩ U_j^P| * N_USERS ) / ( |U_i^P| * |U_j^P| + EPS )

    Where:
      - U_i : 게임 i를 평가한 유저의 집합
      - U_i^P : 게임 i를 긍정적으로 평가한 유저의 집합
      - N_USERS : 전체 유저 수
      - EPS : 분모 0 방지용 작은 숫자
    """
    users_i = ITEM_USERS[i_idx]
    users_j = ITEM_USERS[j_idx]

    # 전체 교집합 크기 (모든 사용자가 평가한 항목 기준)
    inter_cnt = fast_intersection_size(users_i, users_j)
    if inter_cnt < MIN_INTERSECTION:
        return 0.0

    # positive-user sets for item i and j
    U_i_pos = users_i
    U_j_pos = users_j

    # sizes
    size_i_pos = len(U_i_pos)
    size_j_pos = len(U_j_pos)

    # intersection of positive sets
    pos_inter = fast_intersection_size(U_i_pos, U_j_pos)

    # safeguard
    EPS = 1e-6
    if size_i_pos == 0 or size_j_pos == 0:
        pos_term = 0.0
    else:
        pos_term = (pos_inter * N_USERS) / (size_i_pos * size_j_pos + EPS)

    min_factor = min(inter_cnt / BETA, 1.0)

    sim = float(min_factor * pos_term)
    return sim

# 5. 예측값 계산 (advanced sim 사용)
def predict_item_scores_advanced(user_idx, candidate_items, rated_items):
    """
    candidate_items: 추천 후보 아이템 리스트
    rated_items: 사용자가 추천한 아이템 리스트
    return: {item_idx: score}
    """
    scores = {}

    for item in candidate_items:
        s = 0.0
        for r in rated_items:
            sim = item_similarity_advanced(item, r)
            if sim > 0:
                s += sim  # weighted sum

        scores[item] = s

    return scores

# 6. Top-K 추출만 전담하는 함수
def top_k_recommend(scores_dict, k=5):
    sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:k]

# 7. recommend_by_item_advanced (advanced sim 사용)
def recommend_by_item_advanced(user_id: int):

    if user_id not in USER2IDX:
        return {
            "type": "item_based",
            "input_user_id": user_id,
            "result": [],
            "message": "사용자가 존재하지 않습니다."
        }

    u_idx = USER2IDX[user_id]
    rated_items = R[u_idx].indices
    rated_set = set(rated_items)

    if len(rated_items) > MAX_RATED_ITEMS:
        rated_items = rated_items[:MAX_RATED_ITEMS]
        rated_set = set(rated_items)

    # Step 1. 후보 아이템 수집 (TF-IDF style)
    common_counter = Counter()

    for item in rated_items:
        users = ITEM_USERS[item]
        if len(users) > POPULARITY_CAP:
            users = users[:POPULARITY_CAP]

        for u in users:
            for g in R[u].indices:
                if ITEM_USER_LEN[g] > POPULARITY_CAP:
                    continue
                common_counter[g] += 1 / (1 + ITEM_USER_LEN[g])

    candidate_items = [
        g for g, _ in common_counter.most_common(MAX_CANDIDATES)
        if g not in rated_set
    ]

    if len(candidate_items) == 0:
        return {
            "type": "item_based",
            "input_user_id": user_id,
            "result": [],
            "message": "추천 후보가 없습니다."
        }

    # Step 2. 예측값 계산 (advanced sim 사용)
    raw_scores = predict_item_scores_advanced(u_idx, candidate_items, rated_items)

    # Step 3. Min-Max Scaling
    vals = list(raw_scores.values())
    min_s, max_s = min(vals), max(vals)

    scaled_scores = {}
    for item, s in raw_scores.items():
        if max_s == min_s:
            scaled_scores[item] = 0.0
        else:
            scaled_scores[item] = (s - min_s) / (max_s - min_s)

    # Step 4. Top-K 추천선정
    top_items = top_k_recommend(scaled_scores, k=5)

    return {
        "type": "item_based",
        "input_user_id": user_id,
        "result": [
            {"title": IDX2GAME[i], "sim": round(float(s), 5)}
            for i, s in top_items
        ]
    }
