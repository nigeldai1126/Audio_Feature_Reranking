import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def rerank_linear_hybrid(
    baseline_path: str,
    audio_prefsim_path: str,
    popularity_path: str,
    output_path: str,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    K: int = 100
) -> None:
    """
    Linear Hybrid Reranker with GLOBAL normalization:
    Score = α * norm(score) + β * norm(audio_prefsim) - γ * norm(log_pop)
    """

    print("🔹 Loading baseline top-K recommendations...")
    recs = pd.read_parquet(baseline_path)
    recs = recs.sort_values(["user_id", "rank"]).reset_index(drop=True)

    print("🔹 Loading AudioPrefSim...")
    audio = pd.read_csv(audio_prefsim_path)

    print("🔹 Loading popularity...")
    pop = pd.read_csv(popularity_path)
    pop = pop.rename(columns={"track_id": "item_id"})
    pop = pop.drop_duplicates("item_id")[["item_id", "popularity"]]

    # Ensure string IDs
    recs["item_id"] = recs["item_id"].astype(str)
    audio["item_id"] = audio["item_id"].astype(str)

    print("🔹 Merging AudioPrefSim...")
    recs = recs.merge(audio, on=["user_id", "item_id"], how="left")
    recs["audio_prefsim"] = recs["audio_prefsim"].fillna(0)

    print("🔹 Merging popularity...")
    recs = recs.merge(pop, on="item_id", how="left")
    recs["popularity"] = recs["popularity"].fillna(0)
    recs["log_pop"] = np.log1p(recs["popularity"])

    # ==========================================
    # 🚨 GLOBAL normalization
    # ==========================================
    print("🔹 Computing global normalization stats...")

    global_score_min = recs["score"].min()
    global_score_max = recs["score"].max()

    global_audio_min = recs["audio_prefsim"].min()
    global_audio_max = recs["audio_prefsim"].max()

    global_pop_min = recs["log_pop"].min()
    global_pop_max = recs["log_pop"].max()

    print(f"   score range: {global_score_min:.4f} → {global_score_max:.4f}")
    print(f"   audio range: {global_audio_min:.4f} → {global_audio_max:.4f}")
    print(f"   log_pop range: {global_pop_min:.4f} → {global_pop_max:.4f}")

    # Normalize across ALL users, ALL items
    recs["score_norm"] = (recs["score"] - global_score_min) / (global_score_max - global_score_min + 1e-9)
    recs["audio_prefsim_norm"] = (recs["audio_prefsim"] - global_audio_min) / (global_audio_max - global_audio_min + 1e-9)
    recs["log_pop_norm"] = (recs["log_pop"] - global_pop_min) / (global_pop_max - global_pop_min + 1e-9)

    # ==========================================
    # Hybrid Score
    # ==========================================
    print("🔹 Computing hybrid score...")
    recs["hybrid_score"] = (
        alpha * recs["score_norm"] +
        beta  * recs["audio_prefsim_norm"] -
        gamma * recs["log_pop_norm"]
    )

    # ==========================================
    # Reranking
    # ==========================================
    print("🔹 Reranking per user...")
    reranked = (
        recs
        .sort_values(["user_id", "hybrid_score"], ascending=[True, False])
        .groupby("user_id")
        .head(K)
        .reset_index(drop=True)
    )

    # Assign rank
    reranked["rank"] = reranked.groupby("user_id").cumcount() + 1
    reranked["score"] = reranked["hybrid_score"]  # use hybrid score

    # Final output
    result = reranked[["user_id", "item_id", "score", "rank"]].copy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_parquet(output_path, index=False)

    print(f"✅ Saved hybrid reranked results to:\n{output_path}")
    print(f"📊 Users processed: {result['user_id'].nunique()}, rows: {result.shape[0]}")
