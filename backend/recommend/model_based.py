import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from typing import Optional, Dict, Any

_device: torch.device = torch.device("cpu")
_model: Optional["BPRMF"] = None
_meta: Dict[str, Any] = {}

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
        u = self.user_emb(u_idx)
        i = self.item_emb(i_idx)
        j = self.item_emb(j_idx)
        x_ui = (u * i).sum(dim=1)
        x_uj = (u * j).sum(dim=1)
        return x_ui, x_uj

def _build_mappings(train_df: pd.DataFrame):
    pos_df = train_df[train_df["is_recommended"] == 1].copy()
    unique_users = pos_df["user_id"].unique()
    unique_items = pos_df["title"].unique()
    user2idx = {u: idx for idx, u in enumerate(unique_users)}
    idx2user = {idx: u for u, idx in user2idx.items()}
    item2idx = {title: idx for idx, title in enumerate(unique_items)}
    idx2item = {idx: title for title, idx in item2idx.items()}
    u_idx = pos_df["user_id"].map(user2idx).astype("int64").to_numpy()
    i_idx = pos_df["title"].map(item2idx).astype("int64").to_numpy()

    user_pos_items = {}
    for u, i in zip(u_idx, i_idx):
        user_pos_items.setdefault(int(u), set()).add(int(i))

    n_users = len(user2idx)
    n_items = len(item2idx)

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
    global _device
    model = BPRMF(n_users, n_items, n_factors=n_factors).to(_device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_pos = len(u_idx)
    u_idx_t = torch.from_numpy(u_idx).long()
    i_idx_t = torch.from_numpy(i_idx).long()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in range(1, num_batches_per_epoch + 1):
            idx = np.random.randint(0, n_pos, size=batch_size)
            u_batch = u_idx_t[idx].to(_device)
            i_batch = i_idx_t[idx].to(_device)
            j_batch = torch.randint(0, n_items, (batch_size,), device=_device)
            optimizer.zero_grad()
            x_ui, x_uj = model(u_batch, i_batch, j_batch)
            x_uij = x_ui - x_uj
            loss = -torch.log(torch.sigmoid(x_uij) + 1e-10).mean()
            l2_norm = (
                model.user_emb(u_batch).pow(2).sum()
                + model.item_emb(i_batch).pow(2).sum()
                + model.item_emb(j_batch).pow(2).sum()
            ) / batch_size
            loss = loss + reg * l2_norm
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    return model

def fit_model(train_df: pd.DataFrame, model_path, meta_path):
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

    torch.save(
        {
            "state_dict": _model.state_dict(),
            "n_users": n_users,
            "n_items": n_items,
            "n_factors": _model.n_factors,
        },
        model_path,
    )

    _meta = {
        "user2idx": user2idx,
        "idx2user": idx2user,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "user_pos_items": user_pos_items,
    }

    with open(meta_path, "wb") as f:
        pickle.dump(_meta, f)

    return _model

def load_model(model_path, meta_path):
    global _model, _meta, _device

    with open(meta_path, "rb") as f:
        _meta = pickle.load(f)

    user2idx = _meta["user2idx"]
    item2idx = _meta["item2idx"]

    n_users = len(user2idx)
    n_items = len(item2idx)

    checkpoint = torch.load(model_path, map_location=_device)
    n_factors = checkpoint.get("n_factors", 32)

    _model = BPRMF(n_users, n_items, n_factors=n_factors).to(_device)
    _model.load_state_dict(checkpoint["state_dict"])
    _model.eval()

    return _model

def recommend_by_model(user_id: int, n_recommendations: int = 5):
    global _model, _meta, _device

    user2idx = _meta["user2idx"]
    idx2item = _meta["idx2item"]
    user_pos_items = _meta["user_pos_items"]

    if user_id not in user2idx:
        return {
            "type": "model_based",
            "input_user_id": int(user_id),
            "result": [],
        }

    u_idx = user2idx[user_id]
    u_idx_t = torch.tensor([u_idx], device=_device, dtype=torch.long)

    with torch.no_grad():
        user_vec = _model.user_emb(u_idx_t)
        item_vecs = _model.item_emb.weight
        user_norm = torch.norm(user_vec)
        item_norms = torch.norm(item_vecs, dim=1)
        cos_scores = torch.matmul(item_vecs, user_vec.t()).squeeze(1) / (
            item_norms * user_norm + 1e-10
        )
        scores = (cos_scores + 1) / 2

    scores_np = scores.cpu().numpy().astype(float)
    seen = user_pos_items.get(u_idx, set())
    for i in seen:
        scores_np[i] = -1e9

    n_items = scores_np.shape[0]
    k = min(n_recommendations, n_items)
    top_k_idx = np.argpartition(-scores_np, k - 1)[:k]
    top_k_idx = top_k_idx[np.argsort(-scores_np[top_k_idx])]

    recommendation_list = []
    for i in top_k_idx:
        title = str(idx2item[int(i)])
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

def predict_scores_for_items(user_id: int, item_indices: np.ndarray):
    global _model, _meta, _device
    user2idx = _meta["user2idx"]
    if user_id not in user2idx:
        return None

    u_idx = user2idx[user_id]
    u_idx_t = torch.tensor([u_idx], device=_device, dtype=torch.long)

    with torch.no_grad():
        user_vec = _model.user_emb(u_idx_t)
        item_vecs = _model.item_emb.weight[item_indices]
        user_norm = torch.norm(user_vec)
        item_norms = torch.norm(item_vecs, dim=1)
        cos_scores = (item_vecs @ user_vec.t()).squeeze(1) / (
            item_norms * user_norm + 1e-10
        )
        scores = (cos_scores + 1) / 2
        return scores.cpu().numpy()
