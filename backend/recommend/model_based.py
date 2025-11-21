"""
bpr_mf.py

- processed/train_6cols.csv 를 이용해 BPR-MF 학습
- user_id, app_id 를 인덱스로 매핑
- triple (u, i, j)를 샘플링하여 SGD로 학습
- predict_score, predict_scores_for_user, get_top_k_for_user 로 예측/추천
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class BPRMF:
    def __init__(
        self,
        n_factors: int = 64,
        learning_rate: float = 0.05,
        reg: float = 0.01,
        n_epochs: int = 20,
        n_samples_per_epoch: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        n_factors          : 잠재 차원 수 (d)
        learning_rate      : 학습률
        reg                : L2 정규화 계수 (벡터에 그대로 곱하게 구현)
        n_epochs           : 에폭 수
        n_samples_per_epoch: 에폭당 triple 업데이트 횟수 (None이면 positive 수의 10배)
        random_state       : 랜덤 시드
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_samples_per_epoch = n_samples_per_epoch
        self.random_state = random_state

        # 학습 후 채워질 것들
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}

        self.n_users: int = 0
        self.n_items: int = 0

        # 유저별 positive 아이템 인덱스 리스트
        self.user_pos_items: Dict[int, List[int]] = {}

        # 학습된 파라미터
        self.P: Optional[np.ndarray] = None  # (n_users, n_factors)
        self.Q: Optional[np.ndarray] = None  # (n_items, n_factors)

        # 랜덤 생성기
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # 1. 데이터 준비 (user/item 인덱스 매핑 및 positive 리스트 구성)
    # ------------------------------------------------------------------
    def _build_mappings(self, train_df: pd.DataFrame):
        """
        train_df: train_6cols.csv (또는 동일 컬럼 구조를 가진 DataFrame)
        필요한 컬럼: 'user_id', 'app_id', 'is_recommended'
        """
        # 1) positive 상호작용만 사용 (BPR 특성상)
        pos_df = train_df[train_df["is_recommended"] == 1].copy()

        # user_id / app_id (아이템) 고유값 추출
        unique_users = pos_df["user_id"].unique()
        unique_items = pos_df["app_id"].unique()

        self.n_users = len(unique_users)
        self.n_items = len(unique_items)

        # 매핑 생성
        self.user2idx = {u: idx for idx, u in enumerate(unique_users)}
        self.idx2user = {idx: u for u, idx in self.user2idx.items()}
        self.item2idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx2item = {idx: i for i, idx in self.item2idx.items()}

        # 2) 유저별 positive 아이템 인덱스 리스트
        self.user_pos_items = defaultdict(list)
        for row in pos_df.itertuples(index=False):
            u = getattr(row, "user_id")
            i = getattr(row, "app_id")
            u_idx = self.user2idx[u]
            i_idx = self.item2idx[i]
            self.user_pos_items[u_idx].append(i_idx)

        # positive 가 1개도 없는 유저는 제거 (있다면)
        empty_users = [u for u, items in self.user_pos_items.items() if len(items) == 0]
        for u in empty_users:
            del self.user_pos_items[u]

        # 3) 파라미터 초기화
        # 작은 값으로 정규분포 초기화
        self.P = 0.01 * self.rng.standard_normal(size=(self.n_users, self.n_factors))
        self.Q = 0.01 * self.rng.standard_normal(size=(self.n_items, self.n_factors))

        # 4) sample 수 설정
        n_pos = len(pos_df)
        if self.n_samples_per_epoch is None:
            # 관측 수가 아무리 커도 너무 크지 않게 상한
            self.n_samples_per_epoch = min(n_pos, 50000)

        print(f"[BPRMF] n_users={self.n_users}, n_items={self.n_items}, n_pos={n_pos}")
        print(f"[BPRMF] n_samples_per_epoch={self.n_samples_per_epoch}")

    # ------------------------------------------------------------------
    # 2. triple (u, i, j) 샘플링
    # ------------------------------------------------------------------
    def _sample_triplet(self) -> Tuple[int, int, int]:
        """
        BPR 학습용 triple (u, i, j) 샘플링
        u: user index
        i: positive item index
        j: negative item index (u의 positive가 아닌 아이템)
        """
        # 1) positive interaction이 있는 유저 중 하나 샘플
        u = self.rng.choice(list(self.user_pos_items.keys()))
        pos_items_u = self.user_pos_items[u]

        # 2) 그 유저의 positive 아이템 중 하나 샘플
        i = self.rng.choice(pos_items_u)

        # 3) negative 아이템 샘플
        #    전체 아이템에서 하나 뽑되, u의 positive 에 포함되어 있으면 다시 뽑기
        while True:
            j = int(self.rng.integers(0, self.n_items))
            if j not in pos_items_u:
                break

        return u, i, j

    # ------------------------------------------------------------------
    # 3. 한 step(한 triple)에 대한 SGD 업데이트
    # ------------------------------------------------------------------
    def _update_one(self, u: int, i: int, j: int):
        """
        한 triple (u, i, j)에 대한 BPR 업데이트
        """
        assert self.P is not None and self.Q is not None

        pu = self.P[u]         # (d,)
        qi = self.Q[i]         # (d,)
        qj = self.Q[j]         # (d,)

        # 예측 점수
        x_ui = np.dot(pu, qi)
        x_uj = np.dot(pu, qj)
        x_uij = x_ui - x_uj    # Δ = x_ui - x_uj

        # BPR loss 의 gradient factor: σ(-Δ)
        # σ(-Δ) = 1 / (1 + exp(Δ))
        # 값이 너무 커지지 않도록 clip
        if x_uij > 50:
            s = 0.0
        elif x_uij < -50:
            s = 1.0
        else:
            s = 1.0 / (1.0 + np.exp(x_uij))

        lr = self.learning_rate
        reg = self.reg

        # gradient descent step:
        # pu_new = pu - lr * dL/dpu
        # dL/dpu = -s*(qi - qj) + reg*pu   (reg는 L2 항의 계수를 reg라고 가정)
        # ⇒ pu <- pu + lr*s*(qi - qj) - lr*reg*pu
        pu += lr * (s * (qi - qj) - reg * pu)

        # qi_new = qi - lr * dL/dqi
        # dL/dqi = -s*pu + reg*qi
        qi += lr * (s * pu - reg * qi)

        # qj_new = qj - lr * dL/dqj
        # dL/dqj = s*pu + reg*qj
        qj += lr * (-s * pu - reg * qj)

        # 다시 저장 (in-place 업데이트라 사실상 필요 없지만, 명시적으로 작성)
        self.P[u] = pu
        self.Q[i] = qi
        self.Q[j] = qj

    # ------------------------------------------------------------------
    # 4. 학습 루프
    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame):
        """
        train_df: train_6cols.csv
        - 반드시 'user_id', 'app_id', 'is_recommended' 컬럼이 있어야 함
        """
        print("[BPRMF] Building mappings and initializing parameters...")
        self._build_mappings(train_df)

        print("[BPRMF] Start training BPR-MF...")
        for epoch in range(1, self.n_epochs + 1):
            for _ in range(self.n_samples_per_epoch):
                u, i, j = self._sample_triplet()
                self._update_one(u, i, j)

            # 간단한 로그 (원하면 여기서 valid 성능 측정도 가능)
            print(f"  - Epoch {epoch}/{self.n_epochs} completed.")

        print("[BPRMF] Training finished.")

    # ------------------------------------------------------------------
    # 5. 예측 및 추천
    # ------------------------------------------------------------------
    def _get_user_idx(self, user_id: int) -> Optional[int]:
        return self.user2idx.get(user_id, None)

    def _get_item_idx(self, item_id: int) -> Optional[int]:
        return self.item2idx.get(item_id, None)

    def predict_score(self, user_id: int, item_id: int) -> float:
        """
        (1) 특정 (user_id, app_id)에 대한 단일 평점 예측 함수
        - user/item 이 학습에 없으면 0.0 반환
        """
        if self.P is None or self.Q is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")

        u_idx = self._get_user_idx(user_id)
        i_idx = self._get_item_idx(item_id)
        if u_idx is None or i_idx is None:
            return 0.0

        score = float(np.dot(self.P[u_idx], self.Q[i_idx]))
        return score

    def predict_scores_for_user(
        self,
        user_id: int,
        exclude_train_interactions: bool = True,
        train_df: Optional[pd.DataFrame] = None,
    ) -> Dict[int, float]:
        """
        (2) 한 유저에 대해 '모든 후보 아이템'의 예측 점수를 반환하는 함수
        - 반환: {app_id: score, ...} 형태의 dict
        - exclude_train_interactions=True 이면 train에서 이미 본 아이템은 제외
        """
        if self.P is None or self.Q is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")

        u_idx = self._get_user_idx(user_id)
        if u_idx is None:
            # cold-start 유저: 빈 dict
            return {}

        # 모든 아이템에 대한 점수 계산
        user_vec = self.P[u_idx]  # (d,)
        scores = self.Q @ user_vec  # (n_items,)

        # 이미 본 아이템 제거 옵션
        if exclude_train_interactions and train_df is not None:
            seen_items = set(
                train_df[
                    (train_df["user_id"] == user_id) &
                    (train_df["is_recommended"] == 1)
                ]["app_id"].map(self.item2idx.get).dropna().astype(int).tolist()
            )
        else:
            seen_items = set()

        # 후보 아이템 인덱스
        candidate_indices = [i for i in range(self.n_items) if i not in seen_items]
        if not candidate_indices:
            return {}

        # dict: app_id -> score
        result = {
            self.idx2item[i]: float(scores[i])
            for i in candidate_indices
        }
        return result

    def get_top_k_for_user(
        self,
        user_id: int,
        k: int = 5,
        exclude_train_interactions: bool = True,
        train_df: Optional[pd.DataFrame] = None,
    ) -> List[Tuple[int, float]]:
        """
        (3) 한 유저에 대해 Top-K 아이템만 뽑아내는 함수
        - 내부적으로 predict_scores_for_user를 사용
        - 반환: [(app_id, score), ...] 점수 내림차순 정렬
        """
        scores_dict = self.predict_scores_for_user(
            user_id=user_id,
            exclude_train_interactions=exclude_train_interactions,
            train_df=train_df,
        )

        if not scores_dict:
            return []

        # score 기준 내림차순 정렬 후 상위 K개 선택
        sorted_items = sorted(
            scores_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_k = sorted_items[:k]  # [(app_id, score), ...]
        return top_k

    # 기존 이름 유지하고 싶으면 recommend_for_user는 thin wrapper로 유지
    def recommend_for_user(
        self,
        user_id: int,
        n_recommendations: int = 5,
        exclude_train_interactions: bool = True,
        train_df: Optional[pd.DataFrame] = None,
    ) -> List[Tuple[int, float]]:
        """
        기존 인터페이스 유지용 wrapper:
        내부적으로 get_top_k_for_user를 호출
        """
        return self.get_top_k_for_user(
            user_id=user_id,
            k=n_recommendations,
            exclude_train_interactions=exclude_train_interactions,
            train_df=train_df,
        )
