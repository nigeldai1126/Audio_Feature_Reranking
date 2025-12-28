import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import torch
from recbole.model.general_recommender import LightGCN
from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import data_preparation


def rerank_mmr_latent(
    baseline_path: str,
    checkpoint_dir: str,
    idmap_path: str,
    output_path: str,
    lambda_mmr: float = 0.5,
    K: int = 100
):
    """
    Latent MMR with GLOBAL normalization for relevance and similarity.
    """

    print("🔹 Loading baseline Top-K...")
    recs = pd.read_parquet(baseline_path)
    recs = recs.sort_values(["user_id", "rank"]).reset_index(drop=True)

    print("🔹 Loading ID mappings...")
    with open(idmap_path, "r") as f:
        idmap = json.load(f)

    raw2internal = {str(k): int(v) for k, v in idmap["item2id"].items()}

    # ------------------------
    # Load LightGCN checkpoint
    # ------------------------
    ckpt_files = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pth")
    ]
    assert len(ckpt_files) > 0, "No checkpoint found!"

    ckpt_path = max(ckpt_files, key=os.path.getmtime)
    print(f"🔹 Using checkpoint: {ckpt_path}")

    config = Config(
        model="LightGCN",
        dataset="my_data_set",
        config_dict={
            "data_path": "/scratch/jd5316/Capstone/Capstone/Data/recbole_data/",
            "checkpoint_dir": checkpoint_dir,
        }
    )
    config["field_separator"] = ","
    dataset = create_dataset(config)
    _, _, _ = data_preparation(config, dataset)

    model = LightGCN(config, dataset).to(config["device"])
    ckpt = torch.load(ckpt_path, map_location=config["device"])
    model.load_state_dict(ckpt["state_dict"])

    print("🔹 Extracting item embeddings...")
    item_emb = model.item_embedding.weight.detach().cpu().numpy()

    # ------------------------
    # Map raw → internal IDs
    # ------------------------
    print("🔹 Mapping raw item IDs to latent embeddings...")
    recs["internal_item_id"] = recs["item_id"].astype(str).map(raw2internal)
    recs = recs.dropna(subset=["internal_item_id"])
    recs["internal_item_id"] = recs["internal_item_id"].astype(int)

    # ------------------------
    # GLOBAL normalization of scores
    # ------------------------
    print("🔹 GLOBAL score normalization...")

    global_score_min = recs["score"].min()
    global_score_max = recs["score"].max()

    recs["score_norm"] = (
        recs["score"] - global_score_min
    ) / (global_score_max - global_score_min + 1e-9)

    # ------------------------
    # Function per user
    # ------------------------
    def mmr_user(df):
        """
        df columns: [user_id, item_id, internal_item_id, score_norm]
        """

        items = df["internal_item_id"].tolist()
        raw_ids = df["item_id"].tolist()
        scores = df["score_norm"].values.astype(float)

        # Latent embeddings for candidates
        emb = item_emb[items]

        # Compute similarity matrix ONCE
        sim_matrix = cosine_similarity(emb)

        # GLOBAL similarity normalization
        sim_matrix -= sim_matrix.min()
        sim_matrix /= (sim_matrix.max() - sim_matrix.min() + 1e-9)

        selected = []
        remaining = list(range(len(items)))

        for _ in range(min(K, len(items))):
            if len(selected) == 0:
                # Pick highest relevance
                first = np.argmax(scores)
                selected.append(first)
                remaining.remove(first)
                continue

            mmr_scores = []
            for idx in remaining:
                max_sim = max(sim_matrix[idx][selected])
                mmr_value = (
                    lambda_mmr * scores[idx]
                    - (1 - lambda_mmr) * max_sim
                )
                mmr_scores.append((mmr_value, idx))

            _, best_index = max(mmr_scores, key=lambda x: x[0])
            selected.append(best_index)
            remaining.remove(best_index)

        out = df.iloc[selected].copy()
        out["rank"] = range(1, len(selected) + 1)
        out["score"] = scores[selected]  # final processed score
        return out[["user_id", "item_id", "score", "rank"]]

    # ------------------------
    # Apply per user
    # ------------------------
    print("🔹 Running GLOBAL-normalized latent MMR...")
    result = (
        recs.groupby("user_id")
        .apply(mmr_user)
        .reset_index(drop=True)
    )

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_parquet(output_path, index=False)

    print(f"✅ Saved latent MMR reranked list to {output_path}")
