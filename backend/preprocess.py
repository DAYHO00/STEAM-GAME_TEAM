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

# 필요한 컬럼만 미리 읽어서 I/O + 메모리 최적화
rec_df = pd.read_csv(REC_PATH, usecols=["app_id", "user_id", "is_recommended"])
games_df = pd.read_csv(GAMES_PATH, usecols=["app_id", "title", "user_reviews"])
users_df = pd.read_csv(USERS_PATH, usecols=["user_id", "reviews"])

print(f"recommendations: {rec_df.shape}")
print(f"games          : {games_df.shape}")
print(f"users          : {users_df.shape}")

# --------- 2. 필요한 컬럼만 선택 (이미 usecols로 읽었지만 형태 확인용) --------- #
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

# --------- 4. 희소 데이터 필터링 --------- #
print("\n✅ 희소 데이터 필터링 중...")

# 기준값 (원하시면 나중에 조정 가능)
MIN_USER_REVIEWS = 5    # 사용자가 최소 몇 개 이상의 리뷰를 남겼는지

filtered_df = full_df[
    (full_df["reviews"] >= MIN_USER_REVIEWS)
].copy()

print("필터링 후 shape:", filtered_df.shape)

# --------- 5. 조인/필터링 결과 저장 --------- #
joined_path = PROCESSED_DIR / "joined_full_6cols.csv"
filtered_path = PROCESSED_DIR / "joined_filtered_6cols.csv"

print("\n✅ CSV 저장 중... (조인 결과 / 필터링 결과)")
full_df.to_csv(joined_path, index=False)
filtered_df.to_csv(filtered_path, index=False)

print(f"조인 완료 데이터(6컬럼)  : {joined_path}")
print(f"필터링 완료 데이터(6컬럼): {filtered_path}")
print("🎉 1차 전처리(조인 + 필터링) 완료")

# --------- 6. 사용자 단위 6:2:2 분할 (train / valid / test) --------- #
print("\n✅ 사용자 단위 6:2:2 분할(train/valid/test) 진행 중...")

RANDOM_STATE = 42

# 6-1. 전체 데이터를 한 번 랜덤 셔플
#  → 각 user_id 그룹 내부도 랜덤해지니, 이후 idx 기준으로 나누면
#    기존 for-loop에서 grp.sample() 한 것과 같은 효과
shuffled = filtered_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

# 6-2. 각 유저별로 "유저 내부 인덱스" 부여
shuffled["idx_in_user"] = shuffled.groupby("user_id").cumcount()

# 6-3. 각 유저별 총 개수
shuffled["cnt_in_user"] = shuffled.groupby("user_id")["user_id"].transform("size")

# 6-4. 유저별 train/valid 크기 계산 (벡터화된 형태)
shuffled["n_train"] = (shuffled["cnt_in_user"] * 0.6).astype(int)
shuffled["n_valid"] = (shuffled["cnt_in_user"] * 0.2).astype(int)

idx = shuffled["idx_in_user"]
cnt = shuffled["cnt_in_user"]
n_train = shuffled["n_train"]
n_valid = shuffled["n_valid"]

# 너무 적은 유저(n < 5)는 모두 train으로 보내는 로직 유지
small_user = cnt < 5

# 6-5. 마스크로 train / valid / test 한 번에 분리
train_mask = small_user | (idx < n_train)
valid_mask = (~small_user) & (idx >= n_train) & (idx < n_train + n_valid)
test_mask  = (~small_user) & (idx >= n_train + n_valid)

# sanity check: 겹치는 부분 없는지 확인
assert not (train_mask & valid_mask).any()
assert not (train_mask & test_mask).any()
assert not (valid_mask & test_mask).any()

train_df = shuffled[train_mask].drop(columns=["idx_in_user", "cnt_in_user", "n_train", "n_valid"])
valid_df = shuffled[valid_mask].drop(columns=["idx_in_user", "cnt_in_user", "n_train", "n_valid"])
test_df  = shuffled[test_mask ].drop(columns=["idx_in_user", "cnt_in_user", "n_train", "n_valid"])

print("train_df shape:", train_df.shape)
print("valid_df shape:", valid_df.shape)
print("test_df  shape:", test_df.shape)
print("합이 같은가? ->",
      len(train_df) + len(valid_df) + len(test_df) == len(filtered_df))

# --------- 7. 분할 결과 저장 --------- #
train_path = PROCESSED_DIR / "train_6cols.csv"
valid_path = PROCESSED_DIR / "valid_6cols.csv"
test_path  = PROCESSED_DIR / "test_6cols.csv"

train_df.to_csv(train_path, index=False)
valid_df.to_csv(valid_path, index=False)
test_df.to_csv(test_path,  index=False)

print("\n✅ 6:2:2 분할 CSV 저장 완료")
print(f"train : {train_path}")
print(f"valid : {valid_path}")
print(f"test  : {test_path}")
print("\n🎉 전체 전처리 파이프라인 (조인 + 필터링 + 6:2:2 split) 이 완료되었습니다.")
