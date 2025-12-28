import numpy as np
import pandas as pd
from scipy.stats import entropy


# =============================
#   NDCG@K
# =============================
def ndcg_at_k(recs, ground_truth, k, debug=False):
    """
    Compute NDCG@K with binary relevance.
    
    Args:
        recs: list of recommended item IDs
        ground_truth: set of relevant item IDs
        k: cutoff rank
    
    Returns:
        float: NDCG@K score between 0 and 1
    """
    if not recs or not ground_truth or k <= 0:
        return 0.0
    
    # Compute DCG
    dcg = 0.0
    for i, item in enumerate(recs[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    
    # Compute IDCG (ideal ranking has all relevant items first)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    
    if debug and dcg > 0:
        print(f"\n--- NDCG DEBUG ---")
        print(f"DCG: {dcg:.4f}, IDCG: {idcg:.4f}, NDCG: {dcg/idcg:.4f}")
        print(f"Hits: {sum(1 for item in recs[:k] if item in ground_truth)}/{k}")
    
    return dcg / idcg if idcg > 0 else 0.0


# =============================
#   Mean Popularity@K
# =============================
def mean_popularity_at_k(recs, item_df, k, debug=False):
    """
    Average popularity of top-K recommendations.
    
    Args:
        recs: list of recommended item IDs
        item_df: DataFrame indexed by track_id with 'popularity' column
        k: cutoff rank
    
    Returns:
        float: mean popularity score
    """
    if not recs or k <= 0:
        return 0.0
    
    top_k = recs[:k]
    
    # Filter to items that exist in item_df to avoid KeyError
    valid_items = [item for item in top_k if item in item_df.index]
    
    if not valid_items:
        return 0.0
    
    pops = item_df.loc[valid_items, "popularity"]
    mean_pop = float(pops.mean())
    
    if debug:
        print("\n--- Mean Popularity DEBUG ---")
        print(f"Valid items: {len(valid_items)}/{k}")
        print(f"Popularity range: [{pops.min():.2f}, {pops.max():.2f}]")
        print(f"Mean popularity: {mean_pop:.2f}")

    return mean_pop


# =============================
#   Long-Tail Exposure@K
# =============================
def long_tail_exposure_at_k(recs, item_df, long_tail_threshold, k, debug=False):
    """
    Fraction of long-tail items (popularity <= threshold) in top-K recommendations.
    
    Args:
        recs: list of recommended item IDs
        item_df: DataFrame indexed by track_id with 'popularity' column
        long_tail_threshold: popularity threshold for long-tail items
        k: cutoff rank
    
    Returns:
        float: fraction between 0 and 1
    """
    if not recs or k <= 0:
        return 0.0
    
    top_k = recs[:k]
    
    # Filter to items that exist in item_df
    valid_items = [item for item in top_k if item in item_df.index]
    
    if not valid_items:
        return 0.0
    
    pops = item_df.loc[valid_items, "popularity"]
    lt_ratio = float((pops <= long_tail_threshold).mean())
    
    if debug:
        print("\n--- Long-Tail DEBUG ---")
        print(f"Valid items: {len(valid_items)}/{k}")
        print(f"Threshold: {long_tail_threshold:.2f}")
        print(f"LT items: {(pops <= long_tail_threshold).sum()}/{len(valid_items)}")
        print(f"LT Exposure ratio: {lt_ratio:.4f}")

    return lt_ratio


# =============================
#   Exposure Gini
# =============================
def exposure_gini(user_recs, num_items, debug=False):
    """
    Compute Gini coefficient of item exposure distribution.
    Measures inequality in how items are exposed across all recommendations.
    
    Args:
        user_recs: dict {user_id: [list of recommended item IDs]}
        num_items: total number of items in the catalog
    
    Returns:
        float: Gini coefficient between 0 (perfect equality) and 1 (perfect inequality)
    """
    if not user_recs or num_items <= 0:
        return 0.0
    
    # Count exposures for each item
    exposure_counts = {}
    for recs in user_recs.values():
        for item in recs:
            exposure_counts[item] = exposure_counts.get(item, 0) + 1
    
    # Use only exposed items (items with 0 exposure would skew Gini too much)
    if len(exposure_counts) == 0:
        return 0.0
    
    exp_vals = np.array(list(exposure_counts.values()))
    
    if debug:
        print("\n--- Exposure Gini DEBUG ---")
        print(f"Unique items exposed: {len(exp_vals)}/{num_items}")
        print(f"Exposure counts (first 20): {exp_vals[:20].tolist()}")
        print(f"Min: {exp_vals.min()}, Max: {exp_vals.max()}")
        print(f"Mean: {exp_vals.mean():.2f}, Std: {exp_vals.std():.2f}")

    # Handle edge cases
    if len(exp_vals) <= 1:
        return 0.0
    
    # Sort exposure values
    exp_sorted = np.sort(exp_vals)
    n = len(exp_sorted)
    
    # Compute cumulative sum
    cum = np.cumsum(exp_sorted)
    
    # Check for zero total exposure
    total_exposure = cum[-1]
    if total_exposure == 0:
        return 0.0
    
    # Gini coefficient formula
    gini = (n + 1 - 2 * np.sum(cum) / total_exposure) / n
    
    # Clamp to [0, 1] for numerical stability
    return float(max(0.0, min(1.0, gini)))

# =============================
#   Audio Preference KL Divergence
# =============================
def audio_kl_divergence(user_hist_items, user_rec_items, item_df, feature_cols, debug=False):
    """
    KL divergence between user history and recommendation audio feature distributions.
    
    CORRECTION applied:
    Uses GLOBAL min-max normalization based on the entire item_df. 
    This preserves the absolute magnitude of features (e.g., distinguishing 
    between low-energy and high-energy tracks) which local normalization destroys.
    """
    # 1. Input Validation
    if not user_hist_items or not user_rec_items:
        return np.inf

    # Filter to items present in item_df
    hist_items = [item for item in user_hist_items if item in item_df.index]
    rec_items = [item for item in user_rec_items if item in item_df.index]

    if not hist_items or not rec_items:
        return np.inf

    # Keep only available feature columns
    cols = [c for c in feature_cols if c in item_df.columns]
    if not cols:
        return np.inf

    # 2. Extract feature matrices
    hist = item_df.loc[hist_items, cols].astype(float).values
    rec  = item_df.loc[rec_items, cols].astype(float).values

    # Replace NaNs with column means (safety)
    if np.isnan(hist).any():
        hist = np.nan_to_num(hist, nan=np.nanmean(hist))
    if np.isnan(rec).any():
        rec = np.nan_to_num(rec, nan=np.nanmean(rec))

    # -----------------------------
    # 3. GLOBAL MIN-MAX NORMALIZATION
    # -----------------------------
    # We must use the min/max of the ENTIRE dataset (item_df), not just the user's subset.
    # Otherwise, a subset of only quiet songs looks identical to a subset of only loud songs.
    
    global_min = item_df[cols].min().values
    global_max = item_df[cols].max().values
    
    # Calculate range and handle division by zero (if a feature is constant)
    global_range = global_max - global_min
    global_range[global_range == 0] = 1.0  # Avoid div by zero
    
    # Normalize
    hist_norm = (hist - global_min) / global_range
    rec_norm  = (rec - global_min) / global_range

    # -----------------------------
    # 4. Convert to "Feature Preference" distributions
    # -----------------------------
    # Calculate the average feature profile for history and recs
    # e.g., History might be [0.2 Energy, 0.8 Acousticness]
    ph_profile = hist_norm.mean(axis=0)
    qr_profile = rec_norm.mean(axis=0)

    # Normalize these profile vectors to sum to 1 so they act as probabilities
    # (Add small epsilon to avoid zero-division)
    ph = ph_profile / (ph_profile.sum() + 1e-12)
    qr = qr_profile / (qr_profile.sum() + 1e-12)

    # -----------------------------
    # 5. KL Divergence
    # -----------------------------
    # Add epsilon to qr positions to avoid inf if qr has 0 probability where ph > 0
    # scipy.stats.entropy handles normalization, but adding epsilon manually is safer for stability
    epsilon = 1e-10
    ph = ph + epsilon
    qr = qr + epsilon
    
    kl = float(entropy(ph, qr)) 

    if debug:
        print("\n--- AUDIO KL DEBUG ---")
        print("Feature columns:", cols)
        print(f"Global Mins: {global_min[:3]}...") # Print first 3
        print(f"Global Maxs: {global_max[:3]}...")
        print("History Profile (normalized):", ph)
        print("Rec Profile (normalized):    ", qr)
        print("KL Score:", kl)

    return kl


# =============================
#   Recall@K
# =============================
def recall_at_k(recs, ground_truth, k):
    """
    Compute Recall@K: fraction of relevant items that are recommended.
    """
    if not recs or not ground_truth or k <= 0:
        return 0.0
    
    hits = len(set(recs[:k]) & ground_truth)
    return hits / len(ground_truth)


# =============================
#   MAIN evaluate_all FUNCTION
# =============================
def evaluate_all(
    user_recs,               # dict: user → list of track_ids (recommendations)
    user_test_items,         # dict: user → set of track_ids (ground truth)
    user_history_items,      # dict: user → list of track_ids (actual user history)
    item_df,                 # DataFrame indexed by track_id
    long_tail_threshold,     # numeric threshold for long-tail classification
    K=100                    # cutoff rank
):
    """
    Evaluate recommendation quality across multiple metrics.
    
    Args:
        user_recs: recommendations for each user
        user_test_items: held-out test items (ground truth for accuracy)
        user_history_items: actual user listening history (for calibration)
        item_df: item metadata DataFrame with columns:
                 - 'popularity': item popularity score
                 - audio features (any of: danceability, energy, valence, tempo, etc.)
        long_tail_threshold: popularity cutoff for long-tail items
        K: evaluation cutoff rank (default 100)
    
    Returns:
        dict: summary metrics averaged across users
    """
    # Validate inputs
    if not user_recs:
        raise ValueError("user_recs is empty")
    
    if item_df.empty:
        raise ValueError("item_df is empty")
    
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    
    # Define audio feature columns (use only those available)
    all_feature_cols = [
        "danceability", "energy", "valence", "tempo", 
        "key", "loudness", "speechiness", "acousticness", 
        "instrumentalness", "liveness"
    ]
    feature_cols = [col for col in all_feature_cols if col in item_df.columns]
    
    # Check that required columns exist
    if "popularity" not in item_df.columns:
        raise ValueError("item_df missing 'popularity' column")
    
    if not feature_cols:
        print("Warning: No audio feature columns found in item_df")
    
    # Initialize metric collectors
    ndcgs = []
    recalls = []
    meanpops = []
    ltexp = []
    klvals = []
    
    # Track errors
    skipped_users = 0
    zero_hits = 0

    for user, recs in user_recs.items():
        if not recs:
            skipped_users += 1
            continue
        
        try:
            # ---- Accuracy (NDCG@K and Recall@K) ----
            gt = user_test_items.get(user, set())
            if gt:  # Only compute if user has test items
                ndcg = ndcg_at_k(recs, gt, K)
                recall = recall_at_k(recs, gt, K)
                ndcgs.append(ndcg)
                recalls.append(recall)
                
                if ndcg == 0:
                    zero_hits += 1
            
            # ---- Popularity Metrics ----
            meanpops.append(mean_popularity_at_k(recs, item_df, K))
            
            # ---- Long-Tail Exposure ----
            ltexp.append(long_tail_exposure_at_k(recs, item_df, long_tail_threshold, K))
            
            # ---- Audio Calibration KL ----
            hist = user_history_items.get(user, [])
            if hist and feature_cols:  # Only compute if user has history and features exist
                kl = audio_kl_divergence(
                    user_hist_items=hist,
                    user_rec_items=recs[:K],
                    item_df=item_df,
                    feature_cols=feature_cols
                )
                if np.isfinite(kl):  # Only include finite values
                    klvals.append(kl)
        
        except Exception as e:
            skipped_users += 1
            print(f"Warning: Error evaluating user {user[:40]}...: {e}")
            continue
    
    # Report if any users were skipped
    if skipped_users > 0:
        print(f"Warning: Skipped {skipped_users}/{len(user_recs)} users due to errors")
    
    if zero_hits > 0:
        print(f"Warning: {zero_hits}/{len(ndcgs)} users had zero hits (NDCG=0)")
    
    # ---- Exposure Gini (system-level fairness) ----
    exposure_g = exposure_gini(user_recs, item_df.shape[0], debug=False)
    
    # Compute averages (handle empty lists gracefully)
    results = {
        "NDCG@K": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "Recall@K": float(np.mean(recalls)) if recalls else 0.0,
        "MeanPop@K": float(np.mean(meanpops)) if meanpops else 0.0,
        "LongTailExposure@K": float(np.mean(ltexp)) if ltexp else 0.0,
        "AudioKL": float(np.mean(klvals)) if klvals else np.inf,
        "ExposureGini": exposure_g,
        "NumUsers": len(user_recs),
        "NumUsersWithHits": len(ndcgs) - zero_hits
    }
    
    return results