import pandas as pd
from pathlib import Path

# --------- 0. 경로 설정 --------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# 원본 CSV 경로
REC_PATH = DATA_DIR / "recommendations.csv"
GAMES_PATH = DATA_DIR / "games.csv"
USERS_PATH = DATA_DIR / "users.csv"

# --------- 1. CSV 로드 --------- #
print("✅ CSV 파일 로드 중...")

rec_df = pd.read_csv(REC_PATH)
games_df = pd.read_csv(GAMES_PATH)
users_df = pd.read_csv(USERS_PATH)

print(f"recommendations: {rec_df.shape}")
print(f"games          : {games_df.shape}")
print(f"users          : {users_df.shape}")

# --------- 2. 필요한 컬럼만 선택 --------- #
# ⚠️ 실제 CSV 컬럼명이 다르면 여기 이름만 맞춰서 바꿔 주세요.

rec_df = rec_df[["app_id", "user_id", "is_recommended"]]
games_df = games_df[["app_id", "title", "user_reviews"]]
users_df = users_df[["user_id", "reviews"]]

print("\n✅ 필요한 컬럼만 선택 완료")
print("rec_df columns   :", rec_df.columns.tolist())
print("games_df columns :", games_df.columns.tolist())
print("users_df columns :", users_df.columns.tolist())

# --------- 3. 데이터 조인 --------- #
print("\n✅ 1단계 조인: recommendations + games (app_id 기준)")

rec_games_df = rec_df.merge(
    games_df,
    on="app_id",
    how="left"
)

print("rec_games_df shape:", rec_games_df.shape)

print("✅ 2단계 조인: 위 결과 + users (user_id 기준)")

full_df = rec_games_df.merge(
    users_df,
    on="user_id",
    how="left"
)

print("최종 full_df shape:", full_df.shape)

# 이제 full_df 컬럼은 정확히 다음 6개가 됩니다:
# ['app_id', 'user_id', 'is_recommended', 'title', 'user_reviews', 'reviews']

print("full_df columns:", full_df.columns.tolist())

# --------- 4. 희소 데이터 필터링 (선택 사항이지만 발표 내용에 맞게 추가) --------- #
print("\n✅ 희소 데이터 필터링 중...")

# 기준값 (원하시면 나중에 조정 가능)
MIN_USER_REVIEWS = 5    # 사용자가 최소 몇 개 이상의 리뷰를 남겼는지
MIN_GAME_REVIEWS = 5    # 게임이 최소 몇 개 이상의 리뷰를 받았는지

filtered_df = full_df[
    (full_df["reviews"] >= MIN_USER_REVIEWS) &
    (full_df["user_reviews"] >= MIN_GAME_REVIEWS)
].copy()

print("필터링 후 shape:", filtered_df.shape)

# --------- 5. 결과 저장 --------- #
joined_path = PROCESSED_DIR / "joined_full_6cols.csv"
filtered_path = PROCESSED_DIR / "joined_filtered_6cols.csv"

print("\n✅ CSV 저장 중...")
full_df.to_csv(joined_path, index=False)
filtered_df.to_csv(filtered_path, index=False)

print(f"조인 완료 데이터(6컬럼)  : {joined_path}")
print(f"필터링 완료 데이터(6컬럼): {filtered_path}")
print("\n🎉 전처리(조인 + 필터링) 파이프라인이 완료되었습니다.")
