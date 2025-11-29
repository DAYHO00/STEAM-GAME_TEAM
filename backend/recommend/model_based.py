"""
model_based.py

벡터화된 PyTorch 기반 BPR-MF
- fit_model(): 학습 후 모델/메타 저장
- load_model(): 저장된 모델/메타 로딩
- recommend_by_model(): 추천
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from typing import Optional, Dict, Any


# ============================================================
# 0. 전역 상태 (로드 후 FastAPI에서 사용)
# ============================================================
_device: torch.device = torch.device("cpu")
_model: Optional["BPRMF"] = None
_meta: Dict[str, Any] = {}  # user2idx, idx2user, item2idx, idx2item, user_pos_items 등


# ============================================================
# 1. PyTorch BPR-MF 모델 정의
# ============================================================
class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int = 32):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u_idx: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor):
        """
        u_idx, i_idx, j_idx: (batch,)
        """
        u = self.user_emb(u_idx)          # (batch, d)
        i = self.item_emb(i_idx)          # (batch, d)
        j = self.item_emb(j_idx)          # (batch, d)

        x_ui = (u * i).sum(dim=1)         # (batch,)
        x_uj = (u * j).sum(dim=1)         # (batch,)
        return x_ui, x_uj


# ============================================================
# 2. 학습 유틸 함수
# ============================================================
def _build_mappings(train_df: pd.DataFrame):
    """
    train_df: train_6cols.csv
    필요한 컬럼: user_id, title, is_recommended
    """
    pos_df = train_df[train_df["is_recommended"] == 1].copy()

    unique_users = pos_df["user_id"].unique()
    unique_items = pos_df["title"].unique()

    user2idx = {u: idx for idx, u in enumerate(unique_users)}
    idx2user = {idx: u for u, idx in user2idx.items()}

    item2idx = {title: idx for idx, title in enumerate(unique_items)}
    idx2item = {idx: title for title, idx in item2idx.items()}

    # positive pairs (user_idx, item_idx)
    u_idx = pos_df["user_id"].map(user2idx).astype("int64").to_numpy()
    i_idx = pos_df["title"].map(item2idx).astype("int64").to_numpy()

    # 유저별 positive 아이템 (추천 시 이미 본 아이템 제외용)
    user_pos_items = {}
    for u, i in zip(u_idx, i_idx):
        user_pos_items.setdefault(int(u), set()).add(int(i))

    n_users = len(user2idx)
    n_items = len(item2idx)

    print(f"[BPRMF] n_users={n_users}, n_items={n_items}, n_pos={len(u_idx)}")

    return (
        n_users,
        n_items,
        u_idx,
        i_idx,
        user2idx,
        idx2user,
        item2idx,
        idx2item,
        user_pos_items,
    )


def _train_bpr(
    u_idx: np.ndarray,
    i_idx: np.ndarray,
    n_users: int,
    n_items: int,
    n_factors: int = 32,
    epochs: int = 5,
    batch_size: int = 4096,
    num_batches_per_epoch: int = 1000,
    lr: float = 0.05,
    reg: float = 1e-4,
) -> BPRMF:
    """
    벡터화된 PyTorch 학습 루프
    - u_idx, i_idx: positive (user,item) 인덱스 배열
    """
    global _device

    model = BPRMF(n_users, n_items, n_factors=n_factors).to(_device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_pos = len(u_idx)
    u_idx_t = torch.from_numpy(u_idx).long()
    i_idx_t = torch.from_numpy(i_idx).long()

    print(
        f"[BPRMF] Training with epochs={epochs}, batch_size={batch_size}, "
        f"num_batches_per_epoch={num_batches_per_epoch}, device={_device}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        print(f"\n===== Epoch {epoch}/{epochs} =====")
        for batch in range(1, num_batches_per_epoch + 1):
            # 1) positive 샘플 인덱스 뽑기
            idx = np.random.randint(0, n_pos, size=batch_size)
            u_batch = u_idx_t[idx].to(_device)
            i_batch = i_idx_t[idx].to(_device)

            # 2) negative j 무작위 샘플 (간단하게: uniform sampling, positive일 수도 있지만 확률은 매우 작음)
            j_batch = torch.randint(0, n_items, (batch_size,), device=_device)

            optimizer.zero_grad()

            x_ui, x_uj = model(u_batch, i_batch, j_batch)

            # BPR loss = -log σ(x_ui - x_uj) + 정규화
            x_uij = x_ui - x_uj
            loss = -torch.log(torch.sigmoid(x_uij) + 1e-10).mean()

            # L2 regularization
            l2_norm = (
                model.user_emb(u_batch).pow(2).sum()
                + model.item_emb(i_batch).pow(2).sum()
                + model.item_emb(j_batch).pow(2).sum()
            ) / batch_size
            loss = loss + reg * l2_norm

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 진행률 출력 (10% 단위)
            if batch % max(1, (num_batches_per_epoch // 10)) == 0:
                percent = int(batch / num_batches_per_epoch * 100)
                avg_loss = epoch_loss / batch
                print(
                    f"  Batch {batch}/{num_batches_per_epoch} "
                    f"({percent:3d}%) - avg_loss={avg_loss:.4f}"
                )

        print(f"→ Epoch {epoch} finished. avg_loss={epoch_loss / num_batches_per_epoch:.4f}")

    print("[BPRMF] Training finished.")
    return model


# ============================================================
# 3. 외부에서 호출하는 API 함수들
# ============================================================
def fit_model(train_df: pd.DataFrame, model_path, meta_path):
    """
    학습 후 모델/메타 파일 저장
    - model_path: torch state_dict 를 저장할 경로 (예: data/model/bpr_model.pt)
    - meta_path : 매핑 정보/positive 목록을 저장할 경로 (예: data/model/bpr_meta.pkl)
    """
    global _model, _meta

    (
        n_users,
        n_items,
        u_idx,
        i_idx,
        user2idx,
        idx2user,
        item2idx,
        idx2item,
        user_pos_items,
    ) = _build_mappings(train_df)

    _model = _train_bpr(
        u_idx=u_idx,
        i_idx=i_idx,
        n_users=n_users,
        n_items=n_items,
        n_factors=128,
        epochs=10,
        batch_size=4096,
        num_batches_per_epoch=1000,
        lr=0.01,
        reg=1e-2,
    )

    # 모델 저장
    torch.save(
        {
            "state_dict": _model.state_dict(),
            "n_users": n_users,
            "n_items": n_items,
            "n_factors": _model.n_factors,
        },
        model_path,
    )
    print(f"✅ Saved model weights to {model_path}")

    # 메타 정보 저장 (매핑, positive 아이템 목록 등)
    _meta = {
        "user2idx": user2idx,
        "idx2user": idx2user,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "user_pos_items": user_pos_items,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(_meta, f)
    print(f"✅ Saved meta info to {meta_path}")

    return _model


def load_model(model_path, meta_path):
    """
    서버 시작 시 호출: 저장해둔 모델/메타를 로딩
    """
    global _model, _meta, _device

    # 메타 로드
    with open(meta_path, "rb") as f:
        _meta = pickle.load(f)

    user2idx = _meta["user2idx"]
    item2idx = _meta["item2idx"]

    n_users = len(user2idx)
    n_items = len(item2idx)

    # 모델 로드
    checkpoint = torch.load(model_path, map_location=_device)
    n_factors = checkpoint.get("n_factors", 32)

    _model = BPRMF(n_users, n_items, n_factors=n_factors).to(_device)
    _model.load_state_dict(checkpoint["state_dict"])
    _model.eval()

    print(f"✅ Loaded BPR-MF model from {model_path} (users={n_users}, items={n_items})")
    return _model


def recommend_by_model(user_id: int, n_recommendations: int = 5):
    global _model, _meta, _device

    if _model is None or not _meta:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    user2idx = _meta["user2idx"]
    idx2item = _meta["idx2item"]
    user_pos_items = _meta["user_pos_items"]

    if user_id not in user2idx:
        return {
            "type": "model_based",
            "input_user_id": int(user_id),
            "result": [],
            "message": "사용자 ID가 데이터셋에 없습니다.",
        }

    u_idx = user2idx[user_id]
    u_idx_t = torch.tensor([u_idx], device=_device, dtype=torch.long)

    with torch.no_grad():
        user_vec = _model.user_emb(u_idx_t)               # (1, d)
        item_vecs = _model.item_emb.weight               # (n_items, d)

        # --- 코사인 유사도 계산 ---
        user_norm = torch.norm(user_vec)                 # scalar
        item_norms = torch.norm(item_vecs, dim=1)        # (n_items,)

        # dot / (norm_u * norm_items)
        cos_scores = torch.matmul(item_vecs, user_vec.t()).squeeze(1) / (
            item_norms * user_norm + 1e-10
        )

        # [-1, 1] → [0, 1] 스케일링
        scores = (cos_scores + 1) / 2

    scores_np = scores.cpu().numpy().astype(float)

    # 이미 본 아이템은 제외
    seen = user_pos_items.get(u_idx, set())
    for i in seen:
        scores_np[i] = -1e9

    n_items = scores_np.shape[0]
    k = min(n_recommendations, n_items)

    if k == 0:
        return {
            "type": "model_based",
            "input_user_id": int(user_id),
            "result": [],
            "message": "추천할 아이템이 없습니다.",
        }
    
    # 상위 k개 인덱스 추출
    top_k_idx = np.argpartition(-scores_np, k - 1)[:k]
    top_k_idx = top_k_idx[np.argsort(-scores_np[top_k_idx])]

    # title, predict_score 형태로 변환
    recommendation_list = []
    for i in top_k_idx:
        title = str(idx2item[int(i)])   # title 문자열
        score = float(scores_np[int(i)])
        recommendation_list.append(
            {
                "title": title,
                "predicted_score": round(score, 5),
            }
        )

    return {
        "type": "model_based",
        "input_user_id": int(user_id),
        "result": recommendation_list
    }