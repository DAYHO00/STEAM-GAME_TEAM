import pandas as pd
from pathlib import Path

import math

# 전처리된 데이터 경로
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "processed" / "joined_filtered_6cols.csv"

# CSV 로드 (실제 추천 로직에서 사용)
df = pd.read_csv(DATA_PATH)

# is_recomm == True 인 경우만 "추천함(1)" 으로 사용
df_recomm = df[df["is_recommended"] == True].copy()

# 각 게임(app_id)별로, 그 게임을 추천한 user_id 집합 U_p 만들기
item_users = (
    df_recomm.groupby("app_id")["user_id"]
    .apply(set)
    .to_dict()
)

# 편의를 위해 app_id -> title 매핑
app_titles = (
    df.drop_duplicates("app_id")
      .set_index("app_id")["title"]
      .to_dict()
)

# 교집합 기반 discount에 쓰일 β 
BETA = 5

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
    TOP_K = 5

    # 데이터에 해당 app_id가 없으면 빈 결과
    if app_id not in item_users:
        return {
            "type": "item_based",
            "input_app_id": app_id,
            "result": [] 
        }

    target_users = item_users[app_id]
    sims = []  # (다른 app_id, 유사도)

    for other_app_id, other_users in item_users.items():
        if other_app_id == app_id:
            continue

        # 교집합 크기 |U_i ∩ U_j|
        inter = len(target_users & other_users)
        if inter == 0:
            continue

        # 1) 코사인 유사도
        #    Sim(p_i, p_j) = |U_i ∩ U_j| / sqrt(|U_i| * |U_j|)
        denom = math.sqrt(len(target_users) * len(other_users))
        if denom == 0:
            continue
        base_sim = inter / denom

        # 2) 교집합 기반 Discount
        #    DiscountedSim = Sim * min(|U_i ∩ U_j| / β, 1)
        discount = min(inter / BETA, 1.0)
        sim = base_sim * discount

        if sim > 0:
            sims.append((other_app_id, sim))

    # 유사도 높은 순으로 정렬 후 상위 TOP_K 선택
    sims.sort(key=lambda x: x[1], reverse=True)
    top_items = sims[:TOP_K]

    result: list[str] = []
    for other_app_id, sim in top_items:
        title = app_titles.get(other_app_id, f"app_id {other_app_id}")
        result.append(f"{title} (app_id={other_app_id})")
        # result.append(f"{title} (app_id={other_app_id}, sim={sim:.3f})")
        
    return {
        "type": "item_based",
        "input_app_id": app_id,
        "result": result
    }

