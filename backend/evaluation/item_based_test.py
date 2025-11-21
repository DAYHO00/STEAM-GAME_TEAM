import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

# ----------------- 경로 / 모듈 설정 -----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.recommend import item_based

TEST_CSV = Path(PROJECT_ROOT) / "backend" / "processed" / "test_6cols.csv"
if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

# 테스트용으로 500개에 대해서만 처리
# 전체 데이터로 평가하려면 SAMPLE_SIZE = None 으로 바꾸기
SAMPLE_SIZE = 500
# SAMPLE_SIZE = None

if SAMPLE_SIZE is None:
    print(f"Loading full test data from: {TEST_CSV} (may be very slow / memory heavy)")
    df_test = pd.read_csv(TEST_CSV)
else:
    print(f"Loading first {SAMPLE_SIZE} rows from: {TEST_CSV}")
    df_test = pd.read_csv(TEST_CSV, nrows=SAMPLE_SIZE)

print(f"Test set loaded. Shape: {df_test.shape}")

# ----------------- 추천 결과 캐시 (사용자별) -----------------
# item_based.recommend_by_item(user_id) 호출 비용을 줄이기 위해 캐시 사용
@lru_cache(maxsize=10_000)
def get_recommend_titles_for_user(u_id):
    """
    recommend_by_item의 결과에서 추천된 타이틀들만 리스트로 반환.
    반환: list of titles (strings)
    """
    try:
        res = item_based.recommend_by_item(u_id)
    except Exception as e:
        # 예기치 못한 에러 발생 시 빈 리스트 반환
        print(f"Warning: recommend_by_item raised for user {u_id}: {e}")
        return []

    if not isinstance(res, dict):
        return []

    result_list = res.get("result", [])
    titles = [r.get("title") for r in result_list if isinstance(r, dict) and "title" in r]
    return titles

# ----------------- 예측 스코어 계산 (타이틀이 Top-K에 있으면 1.0, 아니면 0.0) -----------------
true_labels = []
pred_scores = []

print("Computing predictions using item_based.recommend_by_item ...")
for i, row in df_test.iterrows():
    u = row['user_id']
    title = row['title']
    true = int(row['is_recommended'])

    rec_titles = get_recommend_titles_for_user(int(u))
    # 단순 점수: 추천 목록에 있으면 1.0, 없으면 0.0
    score = 1.0 if title in rec_titles else 0.0

    true_labels.append(true)
    pred_scores.append(score)

    # row 100개마다 로그 출력
    if (i + 1) % 100 == 0 or (i + 1) == len(df_test):
        print(f" processed {i+1}/{len(df_test)} rows")

true_labels = np.array(true_labels)
pred_scores = np.array(pred_scores)

# ----------------- 단일 threshold 기준 평가 -----------------
# 이 테스트에서는 score가 0/1 이므로 threshold=0.5 를 사용
threshold = 0.5
pred_labels = (pred_scores >= threshold).astype(int)

TP = int(((pred_labels == 1) & (true_labels == 1)).sum())
FP = int(((pred_labels == 1) & (true_labels == 0)).sum())
TN = int(((pred_labels == 0) & (true_labels == 0)).sum())
FN = int(((pred_labels == 0) & (true_labels == 1)).sum())

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

# ===== F1-score 추가 부분 =====
if precision + recall > 0:
    f1 = 2 * (precision * recall) / (precision + recall)
else:
    f1 = 0.0

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

    print(
        f"threshold={t:.2f}  TPR={tpr:.4f}  FPR={fpr:.4f}  "
        f"(TP={tp}, FP={fp}, TN={tn}, FN={fn})"
    )

    TPRs.append(tpr)
    FPRs.append(fpr)

fpr_arr = np.array(FPRs)
tpr_arr = np.array(TPRs)
sort_idx = np.argsort(fpr_arr)
fpr_sorted = fpr_arr[sort_idx]
tpr_sorted = tpr_arr[sort_idx]

# AUROC 계산 (사다리꼴 적분)
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
