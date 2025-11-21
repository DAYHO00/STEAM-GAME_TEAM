import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

# ----------------- 경로 / 모듈 설정 -----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.recommend import item_based_advanced as iba

TEST_CSV = Path(PROJECT_ROOT) / "backend" / "processed" / "test_6cols.csv"
if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

# 테스트용으로 500개에 대해서만 처리
# 전체 데이터로 평가하려면 SAMPLE_SIZE = None 으로 바꾸기
SAMPLE_SIZE = 500
# SAMPLE_SIZE = None

if SAMPLE_SIZE is None:
    print(f"Loading full test data from: {TEST_CSV} (may be slow)")
    df_test = pd.read_csv(TEST_CSV)
else:
    print(f"Loading first {SAMPLE_SIZE} rows from: {TEST_CSV}")
    df_test = pd.read_csv(TEST_CSV, nrows=SAMPLE_SIZE)

print(f"Test set loaded. Shape: {df_test.shape}")

# ---- find a recommend function in advanced module (fallbacks) ----
recommend_fns = [
    "recommend_by_item_advanced",
    "recommend_by_item",
]

recommend_fn = None
for n in recommend_fns:
    if hasattr(iba, n):
        recommend_fn = getattr(iba, n)
        print(f"Using recommendation function: iba.{n}()")
        break

if recommend_fn is None:
    raise RuntimeError("No suitable recommend_by_item function found in item_based_advanced module.")

# ----------------- compute predictions -----------------
true_labels = []
pred_scores = []

print("Computing predictions using item_based_advanced ...")
for i, row in df_test.iterrows():
    u = int(row['user_id'])
    title = row['title']
    true = int(row['is_recommended'])

    try:
        res = recommend_fn(u)
    except Exception as e:
        print(f" Warning: recommend_fn raised for user {u}: {e}")
        res = None

    if isinstance(res, dict) and "result" in res:
        recs = res["result"]
        rec_titles = [r.get("title") for r in recs if isinstance(r, dict)]
        score = 1.0 if title in rec_titles else 0.0
    else:
        score = 0.0

    true_labels.append(true)
    pred_scores.append(score)

    if (i + 1) % 100 == 0 or (i + 1) == len(df_test):
        print(f" processed {i+1}/{len(df_test)} rows")

true_labels = np.array(true_labels)
pred_scores = np.array(pred_scores)

# ----------------- 단일 threshold 기준 평가 -----------------
threshold = 0.5
pred_labels = (pred_scores >= threshold).astype(int)

TP = int(((pred_labels == 1) & (true_labels == 1)).sum())
FP = int(((pred_labels == 1) & (true_labels == 0)).sum())
TN = int(((pred_labels == 0) & (true_labels == 0)).sum())
FN = int(((pred_labels == 0) & (true_labels == 1)).sum())

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

# ----------------- ROC / AUROC 계산 -----------------
# score가 이진이지만 동일한 방식으로 threshold sweep을 수행
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

    print(f"threshold={t:.2f}  TPR={tpr:.4f}  FPR={fpr:.4f}  (TP={tp}, FP={fp}, TN={tn}, FN={fn})")
    TPRs.append(tpr)
    FPRs.append(fpr)

fpr_arr = np.array(FPRs)
tpr_arr = np.array(TPRs)
sort_idx = np.argsort(fpr_arr)
fpr_sorted = fpr_arr[sort_idx]
tpr_sorted = tpr_arr[sort_idx]

auroc = float(np.trapezoid(tpr_sorted, fpr_sorted))

# ----------------- 최종 출력 -----------------
print("\n=== Evaluation Results ===")
print(f"Number of test rows: {len(df_test)}")
print(f"Threshold for classification: {threshold}")
print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-score:  {f1:.6f}")
print(f"Accuracy:  {accuracy:.6f}")
print(f"AUROC (sweep 0->1 step 0.05): {auroc:.6f}")

# ----------------- ROC 시각화 -----------------
plt.figure(figsize=(6, 6))
plt.plot(fpr_sorted, tpr_sorted, label="ROC curve")
plt.plot([0, 1], [0, 1], linestyle='--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUROC = {auroc:.4f})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
