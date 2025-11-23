# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from recommend import (
    recommend_by_user,
    recommend_by_item,
    recommend_by_model,
    recommend_by_user_advanced,
    recommend_by_item_advanced,
)
from recommend.model_based import load_model

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_bpr_model():
    print("🔄 Loading saved BPR-MF model...")

    base = Path(__file__).parent
    model_path = base / "data" / "model" / "bpr_model.pt"
    meta_path = base / "data" / "model" / "bpr_meta.pkl"

    if not model_path.exists():
        raise RuntimeError(f"❌ Model file not found: {model_path}")
    if not meta_path.exists():
        raise RuntimeError(f"❌ Meta file not found: {meta_path}")

    load_model(model_path, meta_path)

    print("✅ BPR-MF model loaded successfully!")


# -------------------------
# Recommendation Endpoints
# -------------------------
@app.get("/recommend/user/{user_id}")
def get_user_based_recommendation(user_id: int):
    return recommend_by_user(user_id)


@app.get("/recommend/item/{app_id}")
def get_item_based_recommendation(app_id: int):
    return recommend_by_item(app_id)


@app.get("/recommend/model/{user_id}")
def get_model_based_recommendation(user_id: int):
    return recommend_by_model(user_id)


# 🔹 Advanced: 모두 user_id 기반
@app.get("/recommend/user-advanced/{user_id}")
def get_user_based_advanced_recommendation(user_id: int):
    return recommend_by_user_advanced(user_id)


@app.get("/recommend/item-advanced/{user_id}")
def get_item_based_advanced_recommendation(user_id: int):
    return recommend_by_item_advanced(user_id)
