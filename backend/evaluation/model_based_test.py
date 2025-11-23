import sys
from pathlib import Path
import pandas as pd
import numpy as np
from functools import lru_cache

# ----------------- 경로 / 모듈 설정 -----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ✅ 오직 model_based 모듈만 사용
from backend.recommend import model_based

# ----------------- 설정 값 -----------------
MODEL_PATH = PROJECT_ROOT / "backend" / "data" / "model" / "bpr_model.pt"
META_PATH  = PROJECT_ROOT / "backend" / "data" / "model" / "bpr_meta.pkl"

TEST_CSV = PROJECT_ROOT / "backend" / "processed" / "test_6cols.csv"

# 테스트에 사용할 샘플 개수 (None 이면 전체)
SAMPLE_SIZE = 500
# SAMPLE_SIZE = None

# 한 유저에 대해 추천받을 개수 (평가용 Top-K)
K_EVAL = 20    # 예: Top-20 추천 기준으로 평가


# ----------------- 모델 로드 -----------------
print(f"Loading BPR-MF model from:\n  {MODEL_PATH}\n  {META_PATH}")
model_based.load_model(MODEL_PATH, META_PATH)

# ----------------- 테스트셋 로드 -----------------
if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

if SAMPLE_SIZE is None:
    print(f"Loading full test data from: {TEST_CSV}")
    df_test = pd.read_csv(TEST_CSV)
else:
    print(f"Loading first {SAMPLE_SIZE} rows from: {TEST_CSV}")
    df_test = pd.read_csv(TEST_CSV, nrows=SAMPLE_SIZE)

print(f"✅ Test set loaded. Shape: {df_test.shape}")
print(df_test.head())


# ----------------- 추천 결과 캐시 -----------------
@lru_cache(maxsize=100_000)
def get_recommended_titles(u_id: int, k: int = K_EVAL):
    """
    특정 user_id에 대해 model_based.recommend_by_model 을 호출하고,
    추천된 title 리스트만 반환.
    """
    try:
        res = model_based.recommend_by_model(u_id, n_recommendations=k)
    except Exception as e:
        print(f"⚠️ recommend_by_model raised for user {u_id}: {e}")
        return []

    if not isinstance(res, dict):
        return []

    result_list = res.get("result", [])
    if not isinstance(result_list, list) or len(result_list) == 0:
        return []

    titles = []
    for r in result_list:
        if not isinstance(r, dict):
            continue
        t = r.get("title")
        if t is None:
            continue
        titles.append(t)
    return titles


# ----------------- NDCG 계산 함수 -----------------
def dcg_at_k(relevances, k):
    """
    relevances: 길이 <= k 인 리스트, 각 원소는 0 또는 1 (또는 정수 점수)
    DCG@k = Σ (2^rel_i - 1) / log2(i+2)
    """
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    gains = (2 ** relevances - 1) / discounts
    return float(gains.sum())


def ndcg_at_k(recommended, relevant_set, k):
    """
    recommended: 추천된 title 리스트
    relevant_set: 실제 정답 title 집합 (set)
    k: 상위 몇 개까지 볼지
    """
    if len(recommended) == 0 or len(relevant_set) == 0:
        return 0.0

    # 추천 순서대로 relevance (0 또는 1) 리스트 만들기
    rec_k = recommended[:k]
    rels = [1 if t in relevant_set else 0 for t in rec_k]

    dcg = dcg_at_k(rels, k)

    # ideal DCG: relevance가 1인 아이템들을 상위에 몰아놨다고 가정
    ideal_rels = [1] * min(len(relevant_set), k)
    idcg = dcg_at_k(ideal_rels, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ----------------- 유저 단위 Top-K 평가 -----------------
print("\n🔍 Evaluating model-based recommendations with Top-K metrics ...")

user_groups = df_test.groupby("user_id")

num_users_total = 0              # test에 등장한 유저 수
num_users_with_pos = 0           # test에서 양성(label=1)을 가진 유저 수
num_users_with_hit = 0           # 적어도 하나는 맞춘 유저 수 (Hit@K)

sum_precision = 0.0
sum_recall = 0.0
sum_ndcg = 0.0

for u, group in user_groups:
    num_users_total += 1
    # 이 유저가 test에서 실제로 좋아한(양성) 타이틀 집합
    true_pos_titles = set(group.loc[group["is_recommended"] == 1, "title"])

    if len(true_pos_titles) == 0:
        # 이 유저는 test에서 양성 샘플이 없으므로 평가에서 제외
        continue

    num_users_with_pos += 1

    # 모델이 추천한 Top-K 타이틀 리스트
    rec_titles = get_recommended_titles(int(u), k=K_EVAL)

    if len(rec_titles) == 0:
        # 추천 결과가 비어있으면, precision/recall/ndcg 모두 0
        continue

    rec_set = set(rec_titles)
    hit_items = true_pos_titles & rec_set   # 교집합

    hits = len(hit_items)

    if hits > 0:
        num_users_with_hit += 1

    # Precision@K: Top-K 중에서 맞춘 비율
    precision_k = hits / len(rec_titles) if len(rec_titles) > 0 else 0.0

    # Recall@K: 실제 정답 중에서 Top-K에 들어간 비율
    recall_k = hits / len(true_pos_titles) if len(true_pos_titles) > 0 else 0.0

    # NDCG@K: 순위를 고려한 지표
    ndcg_k = ndcg_at_k(rec_titles, true_pos_titles, K_EVAL)

    sum_precision += precision_k
    sum_recall += recall_k
    sum_ndcg += ndcg_k

    # 유저별로 디버깅용 출력 하고 싶다면 주석 해제
    # print(f"user {u}: |true_pos|={len(true_pos_titles)}, hits={hits}, P@{K_EVAL}={precision_k:.3f}, R@{K_EVAL}={recall_k:.3f}, NDCG@{K_EVAL}={ndcg_k:.3f}")


# ----------------- 최종 평균 지표 계산 -----------------
if num_users_with_pos > 0:
    avg_precision = sum_precision / num_users_with_pos
    avg_recall = sum_recall / num_users_with_pos
    avg_ndcg = sum_ndcg / num_users_with_pos
    hit_rate = num_users_with_hit / num_users_with_pos
else:
    avg_precision = avg_recall = avg_ndcg = hit_rate = 0.0

print("\n=== 📈 Top-K Recommendation Metrics ===")
print(f"Test Samples (rows)           : {len(df_test)}")
print(f"Unique users in test          : {num_users_total}")
print(f"Users with at least 1 positive: {num_users_with_pos}")
print(f"Evaluation Top-K (K_EVAL)     : {K_EVAL}")
print("----------------------------------------------")
print(f"Hit Rate@{K_EVAL}   (user-level) : {hit_rate:.4f}")
print(f"Precision@{K_EVAL} (macro-avg)  : {avg_precision:.4f}")
print(f"Recall@{K_EVAL}    (macro-avg)  : {avg_recall:.4f}")
print(f"NDCG@{K_EVAL}      (macro-avg)  : {avg_ndcg:.4f}")
print("----------------------------------------------")
print("※ Hit Rate@K: 양성 가진 유저들 중 적어도 하나는 맞춘 유저 비율")
print("※ Precision/Recall/NDCG@K: 유저별 값을 평균 낸 macro-avg 기준")
