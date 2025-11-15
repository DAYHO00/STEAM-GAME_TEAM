import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter
from functools import lru_cache



# ----------------------------------------------------------
# 0. 데이터 로드
# ----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "processed" / "joined_filtered_6cols.csv"

df = pd.read_csv(DATA_PATH)
print(f"🚀 데이터 로드 완료. shape={df.shape}")



# ----------------------------------------------------------
# 1. 전역 구조 초기화
# ----------------------------------------------------------

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

# Item → Users mapping
R_T = R.T.tocsr()
ITEM_USERS = [np.sort(R_T[i].indices) for i in range(N_GAMES)]

# Pre-cache lengths and sqrt lengths
ITEM_USER_LEN = np.array([len(u) for u in ITEM_USERS])
SQRT_ITEM_USER_LEN = np.sqrt(ITEM_USER_LEN)


# ----------------------------------------------------------
# 2. 하이퍼파라미터
# ----------------------------------------------------------
BETA = 5000
MIN_INTERSECTION = 2
MAX_CANDIDATES = 200



# ----------------------------------------------------------
# 3. 초고속 two-pointer 교집합
# ----------------------------------------------------------
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



# ----------------------------------------------------------
# 4. 문서 기반 item-item similarity + LRU Cache
# ----------------------------------------------------------
@lru_cache(maxsize=300_000)
def item_similarity(i_idx, j_idx):
    """
    문서 기반 item-item similarity:
    - Cosine similarity
    - Discount factor
    - LRU 캐시로 반복 계산 최적화
    """

    # 두 게임을 좋아한 유저 목록
    users_i = ITEM_USERS[i_idx]
    users_j = ITEM_USERS[j_idx]

    # 교집합 크기
    inter_cnt = fast_intersection_size(users_i, users_j)
    if inter_cnt < MIN_INTERSECTION:
        return 0.0

    # 코사인 유사도
    denom = SQRT_ITEM_USER_LEN[i_idx] * SQRT_ITEM_USER_LEN[j_idx]
    if denom == 0:
        return 0.0

    sim = inter_cnt / denom

    # Discount 적용
    sim *= min(inter_cnt / BETA, 1.0)

    return sim



# ----------------------------------------------------------
# 5. 예측값 계산 (문서 정석)
# ----------------------------------------------------------
def predict_score(user_idx, target_item_idx):
    """
    문서 정석 공식:
    
    r_hat(a,p) =
        sum_q (DiscountSim(p,q) * r_(a,q))
        -----------------------------------
        sum_q |DiscountSim(p,q)|

    r=1 이므로 weighted_sum = sum(sim)
    """

    rated_items = R[user_idx].indices
    if len(rated_items) == 0:
        return 0

    sims = []
    weighted = []

    for q in rated_items:
        sim = item_similarity(target_item_idx, q)
        if sim > 0:
            sims.append(abs(sim))
            weighted.append(sim)  # r=1

    if not sims:
        return 0

    sims = np.array(sims)
    weighted = np.array(weighted)

    return weighted.sum() / sims.sum()



# ----------------------------------------------------------
# 6. 추천 함수 (정석 + 후보 Pruning 최적화)
# ----------------------------------------------------------
def recommend_by_item(user_id: int):
    if user_id not in USER2IDX:
        return {
            "type": "item_based_paper",
            "input_user_id": user_id,
            "result": [],
            "message": "사용자가 존재하지 않습니다."
        }

    u_idx = USER2IDX[user_id]
    rated_items = R[u_idx].indices
    rated_set = set(rated_items)

    if len(rated_items) == 0:
        return {
            "type": "item_based_paper",
            "input_user_id": user_id,
            "result": [],
            "message": "사용자가 추천한 게임이 없습니다."
        }

    # ------ 후보 아이템 수집 (Co-occurrence 기반) ------
    common_counter = Counter()
    for item in rated_items:
        for u in ITEM_USERS[item]:
            common_counter.update(R[u].indices)

    candidate_items = [
        g for g, _ in common_counter.most_common(MAX_CANDIDATES)
        if g not in rated_set
    ]

    # ------ 예측값 계산 ------
    predictions = []
    for item in candidate_items:
        score = predict_score(u_idx, item)
        if score > 0:
            predictions.append((item, score))

    if not predictions:
        return {
            "type": "item_based_paper",
            "input_user_id": user_id,
            "result": [],
            "message": "추천 결과를 생성할 수 없습니다."
        }

    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

    return {
        "type": "item_based_paper",
        "input_user_id": user_id,
        "result": [
            {"title": IDX2GAME[i], "sim": round(float(s), 5)}
            for i, s in predictions
        ]
    }
