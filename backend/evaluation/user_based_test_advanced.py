import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# import module
from backend.recommend import user_based_advanced as uba

TEST_CSV = Path(PROJECT_ROOT) / "backend" / "processed" / "test_6cols.csv"
if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

SAMPLE_SIZE = 500
# SAMPLE_SIZE = None

if SAMPLE_SIZE is None:
    print(f"Loading full test data from: {TEST_CSV} (may be slow)")
    df_test = pd.read_csv(TEST_CSV)
else:
    print(f"Loading first {SAMPLE_SIZE} rows from: {TEST_CSV}")
    df_test = pd.read_csv(TEST_CSV, nrows=SAMPLE_SIZE)

print(f"Test set loaded. Shape: {df_test.shape}")

# ---- find a predict function ----
predict_fns = [
    "predict_score_for_user_item_advanced",
    "predict_score_for_user_item",
    "predict_score_for_item_user",
]
predict_fn = None
for n in predict_fns:
    if hasattr(uba, n):
        predict_fn = getattr(uba, n)
        print(f"Using prediction function: uba.{n}()")
        break

if predict_fn is None:
    recommend_fns = ["recommend_by_user_advanced", "recommend_by_user"]
    recommend_fn = None
    for n in recommend_fns:
        if hasattr(uba, n):
            recommend_fn = getattr(uba, n)
            print(f"No direct predict() found — using uba.{n}() & binary scoring.")
            break
    if recommend_fn is None:
        raise RuntimeError("No suitable prediction or recommendation function found.")
else:
    recommend_fn = None

# ----------------- compute predictions -----------------
true_labels = []
pred_scores = []

print("Computing predictions using user_based_advanced ...")
for i, row in df_test.iterrows():
    u = int(row['user_id'])
    title = row['title']
    true = int(row['is_recommended'])

    if predict_fn is not None:
        try:
            score = float(predict_fn(u, title))
        except:
            score = 0.0
    else:
        try:
            res = recommend_fn(u)
            if isinstance(res, dict) and "result" in res:
                rec_titles = [r.get("title") for r in res["result"] if isinstance(r, dict)]
                score = 1.0 if title in rec_titles else 0.0
            else:
                score = 0.0
        except:
            score = 0.0

    true_labels.append(true)
    pred_scores.append(score)

true_labels = np.array(true_labels)
pred_scores = np.array(pred_scores)

# ----------------- 단일 threshold 평가 -----------------
threshold = 0.05 if predict_fn is not None else 0.5
pred_labels = (pred_scores >= threshold).astype(int)

TP = int(((pred_labels == 1) & (true_labels == 1)).sum())
FP = int(((pred_labels == 1) & (true_labels == 0)).sum())
TN = int(((pred_labels == 0) & (true_labels == 0)).sum())
FN = int(((pred_labels == 0) & (true_labels == 1)).sum())

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
accuracy  = (TP + TN) / len(df_test)
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

# ----------------- ROC / AUROC 계산 -----------------
thresholds = np.arange(0.0, 1.0001, 0.05)
TPRs = []
FPRs = []

for t in thresholds:
    pl = (pred_scores >= t).astype(int)
    tp = int(((pl == 1) & (true_labels == 1)).sum())
    fp = int(((pl == 1) & (true_labels == 0)).sum())
    tn = int(((pl == 0) & (true_labels == 0)).sum())
    fn = int(((pl == 0) & (true_labels == 1)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    TPRs.append(tpr)
    FPRs.append(fpr)

fpr_arr = np.array(FPRs)
tpr_arr = np.array(TPRs)
sort_idx = np.argsort(fpr_arr)
auroc = float(np.trapezoid(tpr_arr[sort_idx], fpr_arr[sort_idx]))

# 평가 지표 추가 (Hit@K, NDCG@K, MRR)

def hit_at_k(y_true, scores, k=5):
    """Hit@K = Recall@K (binary relevance)"""
    order = np.argsort(scores)[::-1][:k]
    return int(np.any(y_true[order] == 1))

def ndcg_at_k(y_true, scores, k=5):
    n_items = len(y_true)
    if n_items == 0:
        return 0.0

    actual_k = min(k, n_items)

    # sort by predicted score
    order = np.argsort(scores)[::-1][:actual_k]
    gains = (2 ** y_true[order] - 1)

    # discount for actual_k
    discounts = 1 / np.log2(np.arange(1, actual_k + 1) + 1)
    dcg = np.sum(gains * discounts)

    # ideal rankings
    ideal_order = np.argsort(y_true)[::-1][:actual_k]
    ideal_gains = (2 ** y_true[ideal_order] - 1)
    idcg = np.sum(ideal_gains * discounts)

    return dcg / idcg if idcg > 0 else 0.0

def mrr_score(y_true, scores):
    order = np.argsort(scores)[::-1]
    for rank, idx in enumerate(order, start=1):
        if y_true[idx] == 1:
            return 1 / rank
    return 0.0

# 사용자 단위 평가 위해 그룹핑
df_test["pred_score"] = pred_scores
results = []

for user, group in df_test.groupby("user_id"):
    y_true = group["is_recommended"].values
    y_score = group["pred_score"].values

    results.append({
        "hit5": hit_at_k(y_true, y_score, 5),
        "hit10": hit_at_k(y_true, y_score, 10),
        "ndcg5": ndcg_at_k(y_true, y_score, 5),
        "ndcg10": ndcg_at_k(y_true, y_score, 10),
        "mrr": mrr_score(y_true, y_score)
    })

rank_eval = pd.DataFrame(results)

# ----------------- 최종 출력 -----------------
print("\n=== Evaluation Results ===")
print(f"Threshold for classification: {threshold}")
print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-score:  {f1:.6f}")
print(f"Accuracy:  {accuracy:.6f}")
print(f"AUROC:     {auroc:.6f}")

print("\n--- Ranking-based Metrics ---")
print(f"Hit@5  (Recall@5): {rank_eval['hit5'].mean():.6f}")
print(f"NDCG@5:            {rank_eval['ndcg5'].mean():.6f}")
print(f"Hit@10 (Recall@10): {rank_eval['hit10'].mean():.6f}")
print(f"NDCG@10:            {rank_eval['ndcg10'].mean():.6f}")
print(f"MRR:                {rank_eval['mrr'].mean():.6f}")

# ----------------- ROC 시각화 -----------------
plt.figure(figsize=(6, 6))
plt.plot(fpr_arr[sort_idx], tpr_arr[sort_idx], label="ROC curve")
plt.plot([0, 1], [0, 1], linestyle='--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUROC = {auroc:.4f})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
