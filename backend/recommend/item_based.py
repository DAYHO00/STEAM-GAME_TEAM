import pandas as pd
from pathlib import Path

# 전처리된 데이터 경로
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "processed" / "joined_filtered_6cols.csv"

# CSV 로드 (실제 추천 로직에서 사용)
df = pd.read_csv(DATA_PATH)

# def recommend_by_item(app_id: int):
#     """
#     아이템 기반 추천의 기본 골격 함수.
#     - 여기에서 df를 활용하여 같은 게임을 추천할 예정
#     """
#     # (아직 실제 알고리즘은 넣지 않음)
#     # 예시 출력만 반환
#     return {
#         "target_app_id": app_id,
#         "message": "전처리된 데이터를 이용해 아이템 기반 추천을 계산할 예정입니다.",
#         "loaded_rows": len(df)
#     }

def recommend_by_item(app_id: int):
    return {
        "type": "item_based",
        "input_app_id": app_id,
        "result": ["비슷한게임1", "비슷한게임2"]  # 더미 추천
    }

