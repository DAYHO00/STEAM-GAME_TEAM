import pandas as pd
import numpy as np
from math import log
from pathlib import Path
from scipy.sparse import csr_matrix

# 전처리된 데이터 경로
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "processed" / "joined_filtered_6cols.csv"

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
BETA = 5      # 교집합 감쇠 기준값
K = 20        # 이웃 수
MIN_INTERSECTION = 3      # 후보 이웃으로 인정할 최소 공통 게임 수
MAX_CANDIDATES = 5000     # 유사도 계산 대상 최대 이웃 수

USER_IDS = USER_IDS_UNIQUE

print(f"✅ 전역 초기화 완료. 희소 행렬 Shape: {R_MATRIX_SPARSE.shape}")
print(f"   파라미터: β={BETA}, K={K}, MIN_INTERSECTION={MIN_INTERSECTION}, MAX_CANDIDATES={MAX_CANDIDATES}")

# -------------------- 2. 유사도 계산 함수 --------------------

def _calculate_discounted_similarity_idx(u_i_idx: int, u_j_idx: int) -> float:
    """
    내부용: 사용자 인덱스(행 인덱스)로 유사도 계산
    가중 코사인 유사도 + 교집합 기반 감쇠
    """
    # 희소 행렬의 해당 사용자 행 (1 x N_GAMES)
    u_i = R_MATRIX_SPARSE[u_i_idx]
    u_j = R_MATRIX_SPARSE[u_j_idx]

    # u_i * u_j : 둘 다 추천한 게임만 1
    intersection_matrix = u_i.multiply(u_j)

    # intersection_matrix.data == 1 이고, indices는 해당 게임의 컬럼 인덱스
    cols = intersection_matrix.indices
    if cols.size == 0:
        return 0.0

    # 분자: sum(w_p * r_ip * r_jp) = sum(w_p) over 공통 게임
    numerator = W_P_ARR[cols].sum()

    # 분모: sqrt(sum(w_p * r_ip)) * sqrt(sum(w_p * r_jp))
    # u_i.data와 u_j.data는 모두 1이므로, 해당 인덱스의 w_p 합만 보면 됨
    denom_i = np.sqrt(W_P_ARR[u_i.indices].sum()) if u_i.indices.size > 0 else 0.0
    denom_j = np.sqrt(W_P_ARR[u_j.indices].sum()) if u_j.indices.size > 0 else 0.0

    if denom_i == 0.0 or denom_j == 0.0:
        sim = 0.0
    else:
        sim = numerator / (denom_i * denom_j)

    # 교집합 크기 기반 감쇠
    intersection_size = intersection_matrix.nnz
    discount_factor = min(intersection_size / BETA, 1.0)

    return sim * discount_factor

def calculate_discounted_similarity(u_i_id, u_j_id) -> float:
    """
    기존 인터페이스 유지: user_id로 호출할 수 있는 래퍼
    """
    u_i_idx = USER_MAP.get(u_i_id)
    u_j_idx = USER_MAP.get(u_j_id)
    if u_i_idx is None or u_j_idx is None:
        return 0.0
    return _calculate_discounted_similarity_idx(u_i_idx, u_j_idx)

# -------------------- 3. 추천 함수 --------------------

def recommend_by_user(user_id: int):
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
        sim = _calculate_discounted_similarity_idx(user_idx, other_idx)
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

    # 4. 각 후보 게임에 대해 예측 점수 계산
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
