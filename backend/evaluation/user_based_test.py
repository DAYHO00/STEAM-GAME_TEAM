import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.recommend import user_based

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

true_labels = []
pred_scores = []

print("Computing predictions using user_based.predict_score_for_user_item ...")
for i, row in df_test.iterrows():
    u = row['user_id']
    title = row['title']
    true = int(row['is_recommended'])

    # user_based 모듈의 예측 함수 호출
    score = user_based.predict_score_for_user_item(u, title)

    true_labels.append(true)
    pred_scores.append(score)

    # row 100개마다 로그 출력
    if (i + 1) % 100 == 0 or (i + 1) == len(df_test):
        print(f" processed {i+1}/{len(df_test)} rows")

true_labels = np.array(true_labels)
pred_scores = np.array(pred_scores)

# ----- Compute Precision, Recall, Accuracy for threshold = 0.5 -----
threshold = 0.5
pred_labels = (pred_scores >= threshold).astype(int)

TP = int(((pred_labels == 1) & (true_labels == 1)).sum())
FP = int(((pred_labels == 1) & (true_labels == 0)).sum())
TN = int(((pred_labels == 0) & (true_labels == 0)).sum())
FN = int(((pred_labels == 0) & (true_labels == 1)).sum())

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

# Compute AUROC for thresholds = 0.00, 0.05, 0.10, 0.15, ..., 1.00
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

# AUROC 계산
auroc = float(np.trapezoid(tpr_sorted, fpr_sorted))

# ----- 출력 -----
print("\n=== Evaluation Results ===")
print(f"Number of test rows: {len(df_test)}")
print(f"Threshold for classification: {threshold}")
print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"Accuracy:  {accuracy:.6f}")
print(f"AUROC (sweep 0->1 step 0.05): {auroc:.6f}")

# ----- ROC visualization -----
plt.figure(figsize=(6, 6))
plt.plot(fpr_sorted, tpr_sorted)        # ROC curve
plt.plot([0, 1], [0, 1], linestyle='--')  # diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUROC = {auroc:.4f})")
plt.grid(True)
plt.tight_layout()

# 화면에 표시
plt.show()

# ----- (선택) ROC 사진 png로 저장 -----
# out_fig = Path(__file__).parent / "roc_curve.png"
# plt.savefig(out_fig)
# print(f"Saved ROC plot to: {out_fig}")

# ----- (선택) 결과 csv로 저장 -----
# out_df = df_test.copy()
# out_df['pred_score'] = pred_scores
# out_df['pred_label_0.5'] = pred_labels
# out_path = Path(__file__).parent / 'user_based_test_results.csv'
# out_df.to_csv(out_path, index=False)
# print(f"Saved results to: {out_path}")
