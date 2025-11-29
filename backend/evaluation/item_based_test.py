import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.recommend import item_based

TEST_PATH = PROJECT_ROOT / "backend" / "processed" / "test_6cols.csv"
df_test = pd.read_csv(TEST_PATH, nrows=1000)

print("Test loaded:", df_test.shape)

@lru_cache(maxsize=5_000_000)
def get_res(uid):
    try:
        return item_based.recommend_by_item(int(uid))
    except:
        return None

true_labels, scores, hit5_list, ndcg5_list = [], [], [], []

for _, row in df_test.iterrows():
    uid = int(row["user_id"])
    title = row["title"]
    y_true = int(row["is_recommended"])

    res = get_res(uid)

    if res is None:
        sim_score = 0.0
        hit5 = 0
        ndcg5 = 0
    else:
        # raw score (전체 후보에서 title 점수 가져오기)
        sim_score = float(res["scores"].get(title, 0.0))

        # Top-5 hit 계산
        top_items = res["result"]
        hit5 = 0
        ndcg5 = 0.0
        for rank, item in enumerate(top_items):
            if item["title"] == title:
                hit5 = 1
                ndcg5 = 1 / np.log2(rank + 2)
                break

    true_labels.append(y_true)
    scores.append(sim_score)
    hit5_list.append(hit5)
    ndcg5_list.append(ndcg5)

true_labels = np.array(true_labels)
scores = np.array(scores)
hit5_list = np.array(hit5_list)
ndcg5_list = np.array(ndcg5_list)

# Classification Metrics
threshold = 0.05
pred = (scores >= threshold).astype(int)

TP = int(((pred == 1) & (true_labels == 1)).sum())
FP = int(((pred == 1) & (true_labels == 0)).sum())
TN = int(((pred == 0) & (true_labels == 0)).sum())
FN = int(((pred == 0) & (true_labels == 1)).sum())

precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0
accuracy = (TP + TN) / len(true_labels)

# AUROC
thresholds = np.linspace(0, 1, 20)
tprs, fprs = [], []

for t in thresholds:
    pl = (scores >= t).astype(int)

    tp = int(((pl == 1) & (true_labels == 1)).sum())
    fp = int(((pl == 1) & (true_labels == 0)).sum())
    tn = int(((pl == 0) & (true_labels == 0)).sum())
    fn = int(((pl == 0) & (true_labels == 1)).sum())

    tpr = tp / (tp + fn) if (tp + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0

    tprs.append(tpr)
    fprs.append(fpr)

# 정렬된 ROC 커브 만들기
pairs = sorted(zip(fprs, tprs), key=lambda x: x[0])
fprs_sorted, tprs_sorted = zip(*pairs)

# np.trapezoid → 정상적 곡선 적분
auroc = float(np.trapezoid(tprs_sorted, fprs_sorted))

print("\n=== Item-based Evaluation ===")
print(f"Threshold: {threshold}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUROC:     {auroc:.4f}")
print(f"Hit@10:     {hit5_list.mean():.4f}")
print(f"NDCG@10:    {ndcg5_list.mean():.4f}")

plt.figure(figsize=(6,6))
plt.plot(fprs, tprs, marker="o")
plt.plot([0,1], [0,1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUROC={auroc:.4f})")
plt.grid(True)
plt.show()
