import sys
from pathlib import Path
import pandas as pd
import numpy as np
from functools import lru_cache
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.recommend import model_based

MODEL_PATH = PROJECT_ROOT / "backend" / "data" / "model" / "bpr_model.pt"
META_PATH  = PROJECT_ROOT / "backend" / "data" / "model" / "bpr_meta.pkl"
TEST_CSV = PROJECT_ROOT / "backend" / "processed" / "test_6cols.csv"

SAMPLE_SIZE = 10000
K_EVAL = 100

model_based.load_model(MODEL_PATH, META_PATH)

if SAMPLE_SIZE is None:
    df_test = pd.read_csv(TEST_CSV)
else:
    df_test = pd.read_csv(TEST_CSV, nrows=SAMPLE_SIZE)

@lru_cache(maxsize=100_000)
def get_recommended_titles(u_id: int, k: int = K_EVAL):
    try:
        res = model_based.recommend_by_model(u_id, n_recommendations=k)
    except Exception:
        return []

    if not isinstance(res, dict):
        return []

    result_list = res.get("result", [])
    if not isinstance(result_list, list) or len(result_list) == 0:
        return []

    titles = []
    for r in result_list:
        t = r.get("title")
        if t is not None:
            titles.append(t)
    return titles

def dcg_at_k(relevances, k):
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    gains = (2 ** relevances - 1) / discounts
    return float(gains.sum())

def ndcg_at_k(recommended, relevant_set, k):
    if len(recommended) == 0 or len(relevant_set) == 0:
        return 0.0
    rec_k = recommended[:k]
    rels = [1 if t in relevant_set else 0 for t in rec_k]
    dcg = dcg_at_k(rels, k)
    ideal_rels = [1] * min(len(relevant_set), k)
    idcg = dcg_at_k(ideal_rels, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

def compute_user_auc(user_id, true_pos_titles, item2idx, num_neg=200):
    pos_ids = [item2idx[t] for t in true_pos_titles if t in item2idx]
    if len(pos_ids) == 0:
        return None

    all_items = np.array(list(item2idx.values()))
    neg_candidates = np.setdiff1d(all_items, np.array(pos_ids))
    if len(neg_candidates) == 0:
        return None

    sample_neg = np.random.choice(
        neg_candidates,
        size=min(num_neg, len(neg_candidates)),
        replace=False
    )

    eval_ids = np.concatenate([pos_ids, sample_neg])
    scores = model_based.predict_scores_for_items(user_id, eval_ids)
    if scores is None:
        return None

    labels = np.array([1] * len(pos_ids) + [0] * len(sample_neg))
    return roc_auc_score(labels, scores)

user_groups = df_test.groupby("user_id")

num_users_total = 0
num_users_with_pos = 0
num_users_with_hit = 0

sum_precision = 0.0
sum_recall = 0.0
sum_ndcg = 0.0

sum_auc = 0.0
num_auc_users = 0

item2idx = model_based._meta["item2idx"]

for u, group in user_groups:
    num_users_total += 1
    true_pos_titles = set(group.loc[group["is_recommended"] == 1, "title"])

    if len(true_pos_titles) == 0:
        continue

    num_users_with_pos += 1

    rec_titles = get_recommended_titles(int(u), k=K_EVAL)
    if len(rec_titles) == 0:
        continue

    rec_set = set(rec_titles)
    hit_items = true_pos_titles & rec_set
    hits = len(hit_items)

    if hits > 0:
        num_users_with_hit += 1

    precision_k = hits / len(rec_titles)
    recall_k = hits / len(true_pos_titles)
    ndcg_k = ndcg_at_k(rec_titles, true_pos_titles, K_EVAL)

    sum_precision += precision_k
    sum_recall += recall_k
    sum_ndcg += ndcg_k

    auc = compute_user_auc(int(u), true_pos_titles, item2idx, num_neg=200)
    if auc is not None:
        sum_auc += auc
        num_auc_users += 1

if num_users_with_pos > 0:
    avg_precision = sum_precision / num_users_with_pos
    avg_recall = sum_recall / num_users_with_pos
    avg_ndcg = sum_ndcg / num_users_with_pos
    hit_rate = num_users_with_hit / num_users_with_pos
else:
    avg_precision = avg_recall = avg_ndcg = hit_rate = 0.0

if num_auc_users > 0:
    avg_auc = sum_auc / num_auc_users
else:
    avg_auc = 0.0

print("\n=== Evaluation Metrics ===")
print(f"Test Samples: {len(df_test)}")
print(f"Unique users: {num_users_total}")
print(f"Users with positives: {num_users_with_pos}")
print(f"Top-K: {K_EVAL}")
print("--------------------------------")
print(f"Hit Rate@{K_EVAL}: {hit_rate:.4f}")
print(f"Precision@{K_EVAL}: {avg_precision:.4f}")
print(f"Recall@{K_EVAL}: {avg_recall:.4f}")
print(f"NDCG@{K_EVAL}: {avg_ndcg:.4f}")
print(f"AUC-ROC(sampled): {avg_auc:.4f}")
print("--------------------------------")
