# backend/model.py

from pathlib import Path
import pandas as pd

from recommend.model_based import fit_model

if __name__ == "__main__":
    base = Path(__file__).parent
    train_path = base / "processed" / "train_6cols.csv"

    model_dir = base / "data" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "bpr_model.pt"
    meta_path = model_dir / "bpr_meta.pkl"

    print(f"📂 Loading train data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"✅ train_df loaded. shape={train_df.shape}")

    fit_model(train_df, model_path=model_path, meta_path=meta_path)

    print("🎉 Training finished & model/meta saved!")
