import pandas as pd
from pathlib import Path

# 전처리된 데이터 경로
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "processed" / "joined_filtered_6cols.csv"

# CSV 로드 (실제 추천 로직에서 사용)
df = pd.read_csv(DATA_PATH)

# def recommend_by_user(user_id: int):
#     """
#     사용자 기반 추천의 기본 골격 함수.
#     - 여기에서 df를 활용하여 해당 사용자와 비슷한 유저를 찾을 예정
#     """
#     # (아직 실제 알고리즘은 넣지 않음)
#     return {
#         "target_user_id": user_id,
#         "message": "전처리된 데이터를 이용해 사용자 기반 추천을 계산할 예정입니다.",
#         "loaded_rows": len(df)
#     }
def recommend_by_user(user_id: int):
    return {
        "type": "user_based",
        "input_user_id": user_id,
        "result": ["게임A", "게임B", "게임C"]  # 더미 추천
    }
