import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from functools import lru_cache

# -----------------------------
# 경로 설정
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.recommend import item_based

# -----------------------------
# 테스트 데이터 로드
# -----------------------------
TEST_PATH = PROJECT_ROOT / "backend" / "processed" / "test_6cols.csv"
df_test = pd.read_csv(TEST_PATH, nrows=500)

print("Test loaded:", df_test.shape)

# -----------------------------
# 추천 결과 캐싱
# -----------------------------
@lru_cache(maxsize=50_000)
def get_user_result(user_id):
    try:
        return item_based.recommend_by_item(int(user_id))
    except:
        return None


# -----------------------------
# 메트릭 계산 준비
# -----------------------------
true_labels = []
scores = []
hit5_list = []
ndcg5_list = []

for _, row in df_test.iterrows():
    user_id = int(row["user_id"])
    title = row["title"]
    y_true = int(row["is_recommended"])

    result = get_user_result(user_id)

    if result is None:
        sim_score = 0.0
        hit5 = 0
        ndcg5 = 0
    else:
        # Top-K list 안에서 title에 해당하는 sim을 가져온다
        top_items = result.get("result", [])

        sim_score = 0.0
        hit5 = 0
        ndcg5 = 0

        for rank, item in enumerate(top_items):
            if item["title"] == title:
                sim_score = item["sim"]         # ★ sim 점수 그대로 사용
                hit5 = 1                        # Top-5 안에 있음
                ndcg5 = 1 / np.log2(rank + 2)   # NDCG 계산
                break

    true_labels.append(y_true)
    scores.append(sim_score)
    hit5_list.append(hit5)
    ndcg5_list.append(ndcg5)

true_labels = np.array(true_labels)
scores = np.array(scores)
hit5_list = np.array(hit5_list)
ndcg5_list = np.array(ndcg5_list)


# -----------------------------
# Classification Metrics
# -----------------------------
threshold = 0.1
pred_labels = (scores >= threshold).astype(int)

TP = int(((pred_labels == 1) & (true_labels == 1)).sum())
FP = int(((pred_labels == 1) & (true_labels == 0)).sum())
TN = int(((pred_labels == 0) & (true_labels == 0)).sum())
FN = int(((pred_labels == 0) & (true_labels == 1)).sum())

precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
accuracy = (TP + TN) / len(true_labels)


# -----------------------------
# AUROC 계산
# -----------------------------
thresholds = np.linspace(0, 1, 20)
tprs, fprs = [], []

for t in thresholds:
    pred = (scores >= t).astype(int)

    tp = int(((pred == 1) & (true_labels == 1)).sum())
    fp = int(((pred == 1) & (true_labels == 0)).sum())
    tn = int(((pred == 0) & (true_labels == 0)).sum())
    fn = int(((pred == 0) & (true_labels == 1)).sum())

    tpr = tp / (tp + fn) if (tp + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0

    tprs.append(tpr)
    fprs.append(fpr)

auroc = float(np.trapezoid(tprs, fprs))


# -----------------------------
# 결과 출력
# -----------------------------
print("\n=== Item-based Evaluation ===")
print(f"Threshold: {threshold}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUROC:     {auroc:.4f}")
print(f"Hit@5:     {hit5_list.mean():.4f}")
print(f"NDCG@5:    {ndcg5_list.mean():.4f}")

# -----------------------------
# ROC Curve
# -----------------------------
plt.figure(figsize=(6,6))
plt.plot(fprs, tprs, marker="o")
plt.plot([0,1], [0,1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUROC={auroc:.4f})")
plt.grid(True)
plt.show()
