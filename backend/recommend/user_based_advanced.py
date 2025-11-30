import pandas as pd
import numpy as np
from math import log
from pathlib import Path
from scipy.sparse import csr_matrix

# 전처리된 데이터 경로
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "processed" / "train_6cols.csv"

# CSV 로드
df = pd.read_csv(DATA_PATH)

print(f"✅ 데이터 로드 완료. Shape: {df.shape}")

# -------------------- 1. 전역 변수 초기화 (희소 행렬 + 인덱스) -------------------- 

print("\n⏳ 전역 희소 행렬 및 가중치 계산 중...")

# 1-1. 사용자 및 게임 매핑
USER_IDS_UNIQUE = df['user_id'].unique()
GAME_TITLES_UNIQUE = df['title'].unique()

USER_MAP = {uid: i for i, uid in enumerate(USER_IDS_UNIQUE)}
GAME_MAP = {title: i for i, title in enumerate(GAME_TITLES_UNIQUE)}
INV_GAME_MAP = {i: title for title, i in GAME_MAP.items()}

# 1-2. CSR 행렬 생성 (User x Game)
data = df['is_recommended'].values
row_ind = df['user_id'].map(USER_MAP).values
col_ind = df['title'].map(GAME_MAP).values

R_MATRIX_SPARSE = csr_matrix(
    (data, (row_ind, col_ind)),
    shape=(len(USER_IDS_UNIQUE), len(GAME_TITLES_UNIQUE))
)
N_USERS, N_GAMES = R_MATRIX_SPARSE.shape

# 1-3. CSC 행렬 & inverted index (Game -> Users)
R_MATRIX_CSC = R_MATRIX_SPARSE.tocsc()
# 각 게임 인덱스마다, 해당 게임을 추천한 사용자 인덱스 배열
ITEM_USERS = [R_MATRIX_CSC[:, j].indices for j in range(N_GAMES)]

# 2. 인기 게임 보정 가중치 (W_p) 계산
n_p = np.asarray(R_MATRIX_SPARSE.sum(axis=0)).flatten()   # 각 게임이 받은 추천 수
n_p[n_p == 0] = 1                                        # 0 나눔 방지
W_P_SERIES = pd.Series(np.log(N_USERS / n_p), index=GAME_TITLES_UNIQUE)
W_P_ARR = W_P_SERIES.values                              # numpy array로 캐시

# 3. 전역 파라미터
BETA = 5     # 교집합 감쇠 기준값
K = 20        # 이웃 수
MIN_INTERSECTION = 3      # 후보 이웃으로 인정할 최소 공통 게임 수
MAX_CANDIDATES = 5000     # 유사도 계산 대상 최대 이웃 수

# -------------------- POS / NEG 세트 (유사도 공식에 필요한 추가 정보) --------------------
# is_recommended==1 을 positive, is_recommended==0 을 negative 로 간주

POS_ITEMS_PER_USER = {i: np.array([], dtype=int) for i in range(len(USER_IDS_UNIQUE))}
NEG_ITEMS_PER_USER = {i: np.array([], dtype=int) for i in range(len(USER_IDS_UNIQUE))}
ALL_ITEMS_PER_USER = {i: np.array([], dtype=int) for i in range(len(USER_IDS_UNIQUE))}

if 'is_recommended' in df.columns:
    gp_pos = df[df['is_recommended'] == 1].groupby('user_id')['title'].apply(list)
    for uid, titles in gp_pos.items():
        uidx = USER_MAP.get(uid)
        if uidx is None:
            continue
        POS_ITEMS_PER_USER[uidx] = np.array([GAME_MAP[t] for t in titles], dtype=int)

    gp_neg = df[df['is_recommended'] == 0].groupby('user_id')['title'].apply(list)
    for uid, titles in gp_neg.items():
        uidx = USER_MAP.get(uid)
        if uidx is None:
            continue
        NEG_ITEMS_PER_USER[uidx] = np.array([GAME_MAP[t] for t in titles], dtype=int)

# ALL = union(pos, neg) ; if both empty, fallback to sparse row indices (dataset may contain only positives)
for uidx in range(len(USER_IDS_UNIQUE)):
    a = POS_ITEMS_PER_USER.get(uidx, np.array([], dtype=int))
    b = NEG_ITEMS_PER_USER.get(uidx, np.array([], dtype=int))
    if a.size == 0 and b.size == 0:
        row_idxs = R_MATRIX_SPARSE[uidx].indices
        ALL_ITEMS_PER_USER[uidx] = np.array(row_idxs, dtype=int)
        POS_ITEMS_PER_USER[uidx] = np.array(row_idxs, dtype=int)  # negatives absent -> treat seen as positive
    else:
        ALL_ITEMS_PER_USER[uidx] = np.unique(np.concatenate([a, b]))

# Advanced similarity parameters
ALPHA = 0.9
EPSILON = 1e-6

print(f"✅ 전역 초기화 완료. 희소 행렬 Shape: {R_MATRIX_SPARSE.shape}")
print(f"   파라미터: β={BETA}, K={K}, MIN_INTERSECTION={MIN_INTERSECTION}, MAX_CANDIDATES={MAX_CANDIDATES}")
print(f"   advanced sim params: ALPHA={ALPHA}, EPSILON={EPSILON}")

# -------------------- 2. 유사도 계산 함수 --------------------

def _calculate_advanced_similarity_idx(u_i_idx: int, u_j_idx: int) -> float:
    """
    아래의 식을 이용하여 유사도를 계산함

        sim(U_i, U_j) = min(|I_i ∩ I_j| / BETA, 1)
                        * [ ALPHA * (|I_i^P ∩ I_j^P| / (|I_i^P ∪ I_j^P| + EPS))
                        + (1-ALPHA) * (|I_i^N ∩ I_j^N| / (|I_i^N ∪ I_j^N| + EPS)) ]
    
    Where:
      - I_i : 유저 i가 평가한 게임의 집합
      - I_i^P : 유저 i가 긍정적으로 평가한 게임의 집합
      - I_i^N : 유저 i가 부정적으로 평가한 게임의 집합
      - EPS : 분모 0 방지용 작은 숫자
    """
    I_i_all = ALL_ITEMS_PER_USER.get(u_i_idx, np.array([], dtype=int))
    I_j_all = ALL_ITEMS_PER_USER.get(u_j_idx, np.array([], dtype=int))

    if I_i_all.size == 0 or I_j_all.size == 0:
        return 0.0

    inter_all = np.intersect1d(I_i_all, I_j_all)
    inter_all_cnt = inter_all.size
    if inter_all_cnt == 0:
        return 0.0

    min_factor = min(inter_all_cnt / BETA, 1.0)

    # positive
    I_i_pos = POS_ITEMS_PER_USER.get(u_i_idx, np.array([], dtype=int))
    I_j_pos = POS_ITEMS_PER_USER.get(u_j_idx, np.array([], dtype=int))
    if I_i_pos.size == 0 and I_j_pos.size == 0:
        pos_sim = 0.0
    else:
        pos_inter = np.intersect1d(I_i_pos, I_j_pos).size
        pos_union = np.union1d(I_i_pos, I_j_pos).size
        pos_sim = (pos_inter / (pos_union + EPSILON)) if pos_union > 0 else 0.0

    # negative
    I_i_neg = NEG_ITEMS_PER_USER.get(u_i_idx, np.array([], dtype=int))
    I_j_neg = NEG_ITEMS_PER_USER.get(u_j_idx, np.array([], dtype=int))
    if I_i_neg.size == 0 and I_j_neg.size == 0:
        neg_sim = 0.0
    else:
        neg_inter = np.intersect1d(I_i_neg, I_j_neg).size
        neg_union = np.union1d(I_i_neg, I_j_neg).size
        neg_sim = (neg_inter / (neg_union + EPSILON)) if neg_union > 0 else 0.0

    sim = min_factor * (ALPHA * pos_sim + (1.0 - ALPHA) * neg_sim)
    return float(sim)


def calculate_advanced_similarity(u_i_id, u_j_id) -> float:
    """
    Wrapper preserving original external interface (user IDs -> similarity)
    """
    u_i_idx = USER_MAP.get(u_i_id)
    u_j_idx = USER_MAP.get(u_j_id)
    if u_i_idx is None or u_j_idx is None:
        return 0.0
    return _calculate_advanced_similarity_idx(u_i_idx, u_j_idx)

# -------------------- 3. 추천 함수 --------------------

def recommend_by_user_advanced(user_id: int):

    if user_id not in USER_MAP:
        return {
            "type": "user_based",
            "input_user_id": user_id,
            "result": [],
            "message": "사용자 ID가 데이터셋에 없습니다."
        }

    print(f"\n 사용자 ID {user_id}에 대한 추천 요청 처리 중...")

    user_idx = USER_MAP[user_id]
    user_row = R_MATRIX_SPARSE[user_idx]
    user_game_indices = user_row.indices          # 이 유저가 추천한 게임들
    rated_game_indices = set(user_game_indices)   # 이미 본 게임

    # 1. 같은 게임을 본 적 있는 유저들만 후보로 모으기
    from collections import defaultdict
    co_count = defaultdict(int)   # neighbor_idx -> 공통 게임 개수

    for g_idx in user_game_indices:
        for other_idx in ITEM_USERS[g_idx]:
            if other_idx == user_idx:
                continue
            co_count[other_idx] += 1

    # 공통 게임 수가 너무 적은 유저는 버림
    candidates = [u for u, c in co_count.items() if c >= MIN_INTERSECTION]

    if not candidates:
        return {
            "type": "user_based",
            "input_user_id": int(user_id),
            "result": [],
            "message": "추천에 사용할 유효한 이웃을 찾을 수 없습니다. (공통 게임 부족)"
        }

    # 공통 게임이 많은 순으로 정렬 후, 상위 MAX_CANDIDATES만 사용
    candidates = sorted(
        candidates,
        key=lambda u: co_count[u],
        reverse=True
    )[:MAX_CANDIDATES]

    print(f"   → 공통 게임 >= {MIN_INTERSECTION}인 후보 이웃 수: {len(candidates)}")

    # 2. 후보들에 대해서만 유사도 계산
    similarity_scores = {}
    for other_idx in candidates:
        sim = _calculate_advanced_similarity_idx(user_idx, other_idx)
        if sim > 0:
            other_user_id = int(USER_IDS_UNIQUE[other_idx])
            similarity_scores[other_user_id] = sim

    if not similarity_scores:
        return {
            "type": "user_based",
            "input_user_id": int(user_id),
            "result": [],
            "message": "양수 유사도를 가진 이웃을 찾지 못했습니다."
        }

    sorted_sim = pd.Series(similarity_scores).sort_values(ascending=False)
    neighbors = sorted_sim.head(K)   # 상위 K명

    print(f"   → 최종 사용 이웃 수: {len(neighbors)} (K={K})")

    # 3. 이웃들이 실제로 본 게임들만 후보로 추천
    from collections import defaultdict
    candidate_games = defaultdict(float)  # game_idx -> 가중치(이웃 유사도 합)

    for neighbor_id, sim_score in neighbors.items():
        neighbor_idx = USER_MAP[neighbor_id]
        neighbor_row = R_MATRIX_SPARSE[neighbor_idx]
        for g_idx in neighbor_row.indices:
            if g_idx in rated_game_indices:
                continue  # 이미 본 게임은 제외
            candidate_games[g_idx] += abs(sim_score)

    if not candidate_games:
        return {
            "type": "user_based",
            "input_user_id": int(user_id),
            "result": [],
            "message": "이웃들이 본 게임 중에서 새로 추천할 후보가 없습니다."
        }

    sim_sum_abs = neighbors.abs().sum()

       # 4. 각 후보 게임에 대해 예측 점수 계산 (수정 버전)
    prediction_scores = {}
    for game_idx in candidate_games.keys():
        numerator = 0.0


        for neighbor_id, sim_score in neighbors.items():
            neighbor_idx = USER_MAP[neighbor_id]
            r_jp = R_MATRIX_SPARSE[neighbor_idx, game_idx]  # 0 또는 1
            if r_jp != 0:
                numerator += sim_score * r_jp

        if sim_sum_abs > 0 and numerator > 0:
            game_title = INV_GAME_MAP[game_idx]
            prediction_scores[game_title] = numerator / sim_sum_abs

    if not prediction_scores:
        return {
            "type": "user_based",
            "input_user_id": int(user_id),
            "result": [],
            "message": "추천할 게임을 찾지 못했습니다."
        }

    # 5. 상위 5개만 반환
    recommended_games = pd.Series(prediction_scores).sort_values(ascending=False)
    final_recommendations = recommended_games.head(5)

    recommendation_list = [
        {"title": title, "predicted_score": round(float(score), 5)}
        for title, score in zip(final_recommendations.index, final_recommendations.values)
    ]

    return {
        "type": "user_based",
        "input_user_id": int(user_id),
        "result": recommendation_list
    }


def predict_score_for_user_item(user_id, game_title) -> float:
    """
    user_id, game_title에 대해 0~1 사이의 예측 점수를 반환.
    """
    # 존재 여부 확인
    if user_id not in USER_MAP:
        return 0.0
    if game_title not in GAME_MAP:
        return 0.0

    user_idx = USER_MAP[user_id]
    game_idx = GAME_MAP[game_title]

    # 1) 후보 이웃 수집 (train에서 해당 유저가 본 게임을 통해)
    from collections import defaultdict
    user_row = R_MATRIX_SPARSE[user_idx]
    user_game_indices = user_row.indices

    co_count = defaultdict(int)
    for g_idx in user_game_indices:
        for other_idx in ITEM_USERS[g_idx]:
            if other_idx == user_idx:
                continue
            co_count[other_idx] += 1

    # 필터링
    candidates = [u for u, c in co_count.items() if c >= MIN_INTERSECTION]
    if not candidates:
        return 0.0

    candidates = sorted(candidates, key=lambda u: co_count[u], reverse=True)[:MAX_CANDIDATES]

    # 2) 후보들에 대해 유사도 계산 (모듈 내부 함수 재사용)
    sim_scores = {}
    for other_idx in candidates:
        sim = _calculate_advanced_similarity_idx(user_idx, other_idx)
        if sim > 0:
            other_user_id = int(USER_IDS_UNIQUE[other_idx])
            sim_scores[other_user_id] = sim
    
    if not sim_scores:
        return 0.0

    sorted_sim = pd.Series(sim_scores).sort_values(ascending=False)
    neighbors = sorted_sim.head(K)   # pandas Series: index=neighbor_user_id, value=sim
    sim_sum_abs = neighbors.abs().sum()
    if sim_sum_abs == 0:
        return 0.0
      # 3) (user, game)에 대한 예측 점수 계산 (수정 버전)
    numerator = 0.0

    for neighbor_id, sim_score in neighbors.items():
        neighbor_idx = USER_MAP.get(neighbor_id)
        if neighbor_idx is None:
            continue
        r_jp = R_MATRIX_SPARSE[neighbor_idx, game_idx]  # 0 또는 1
        if r_jp != 0:
            numerator += sim_score * r_jp

    if numerator <= 0:
        return 0.0

    return float(numerator / sim_sum_abs)
