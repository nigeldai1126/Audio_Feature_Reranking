import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# --------------------------
# GENRE PROCESSING HELPERS
# --------------------------
def parse_genre_cell(cell, sep_chars=None):
    if pd.isna(cell):
        return []
    else:
        sep = ', '
    parts = [p.strip() for p in str(cell).split(sep) if p.strip() != ""]
    return parts


def build_genre_mappings(music_info_df, genre_col="genre", sep_chars=None, min_items_per_genre=1):
    genre2items = defaultdict(set)
    item2genres = dict()

    for _, row in music_info_df.iterrows():
        item = row["track_id"]
        genres = parse_genre_cell(row.get(genre_col, None), sep_chars=sep_chars)
        if len(genres) == 0:
            genres = []
        item2genres[item] = genres
        for g in genres:
            genre2items[g].add(item)

    for item, genres in item2genres.items():
        item2genres[item] = [g for g in genres if g in genre2items]

    return genre2items, item2genres


def estimate_user_genre_pref(listening_df, item2genres, genre_list, weight_col="playcount"):
    user_genre_counts = defaultdict(lambda: defaultdict(float))

    for _, row in listening_df.iterrows():
        user = row["user_id"]
        track = row["track_id"]
        wt = row.get(weight_col, 1.0)
        genres = item2genres.get(track, [])
        if len(genres) == 0:
            continue
        share = wt / len(genres)
        for g in genres:
            user_genre_counts[user][g] += share

    user2genreprob = dict()
    for user, counts in user_genre_counts.items():
        total = sum(counts.values())
        if total <= 0:
            user2genreprob[user] = {g: 1.0 / len(genre_list) for g in genre_list}
        else:
            probs = {g: counts.get(g, 0.0) / total for g in genre_list}
            user2genreprob[user] = probs

    return user2genreprob


def compute_p_i_given_c(genre2items, items_in_candidates):
    p_i_given_c = dict()
    for g, items in genre2items.items():
        avail = set(items).intersection(items_in_candidates)
        if len(avail) == 0:
            p_i_given_c[g] = dict()
        else:
            p = 1.0 / len(avail)
            p_i_given_c[g] = {item: p for item in avail}
    return p_i_given_c


# --------------------------
# XQuAD USER-LEVEL RERANKING
# --------------------------
def xquad_rerank_for_user(
    user_candidates,
    user_genre_prob,
    p_i_given_c,
    global_min_rel,
    global_max_rel,
    lambda_xquad=0.5,
    K=100,
    genre_list=None
):
    """
    Greedy xQuAD selection for a single user with **GLOBAL relevance normalization**
    """

    if genre_list is None:
        genre_list = list(user_genre_prob.keys())

    items = [it for it, _ in user_candidates]
    raw_rel = np.array([score for _, score in user_candidates], dtype=float)

    # -----------------------------
    # 🔥 GLOBAL NORMALIZATION HERE
    # -----------------------------
    rel_norm = {
        item: (raw_score - global_min_rel) / (global_max_rel - global_min_rel + 1e-9)
        for (item, raw_score) in user_candidates
    }

    remaining = items.copy()
    selected = []
    coverage = {g: 0.0 for g in genre_list}

    # -----------------------------
    # xQuAD greedy selection
    # -----------------------------
    while len(selected) < min(K, len(remaining)):
        best_item = None
        best_val = -1e12

        for i in remaining:
            rel = rel_norm[i]

            # Diversity term = probabilistic → already between 0 and 1
            div = 0.0
            for c in genre_list:
                p_c_u = user_genre_prob.get(c, 0.0)
                p_i_c = p_i_given_c.get(c, {}).get(i, 0.0)
                div += p_c_u * p_i_c * (1.0 - coverage[c])

            score = (1 - lambda_xquad) * rel + lambda_xquad * div

            if score > best_val:
                best_val = score
                best_item = i

        if best_item is None:
            break

        selected.append((best_item, float(best_val)))
        remaining.remove(best_item)

        for c in genre_list:
            coverage[c] += p_i_given_c.get(c, {}).get(best_item, 0.0)

    out = []
    for rank, (it, sc) in enumerate(selected, start=1):
        out.append((it, sc, rank))
    return out


# --------------------------
# FULL XQuAD PIPELINE
# --------------------------
def rerank_xquad(
    baseline_path: str,
    music_info_path: str,
    listening_history_path: str,
    output_path: str,
    lambda_xquad: float = 0.5,
    K: int = 100,
    genre_col: str = "tags",
    sep_chars: str = None,
    min_items_per_genre: int = 1,
    weight_col: str = "playcount"
):

    print("Loading baseline top-K...")
    recs = pd.read_parquet(baseline_path)
    recs = recs.sort_values(["user_id", "rank"]).reset_index(drop=True)

    print("Loading music info...")
    music = pd.read_csv(music_info_path)

    print("Building genre mappings...")
    genre2items, item2genres = build_genre_mappings(
        music, genre_col=genre_col, sep_chars=sep_chars, min_items_per_genre=min_items_per_genre
    )
    genre_list = sorted(list(genre2items.keys()))

    if len(genre_list) == 0:
        raise ValueError("No genres found. Check genre column and separators.")

    print("Loading listening history...")
    hist = pd.read_csv(listening_history_path)
    hist["track_id"] = hist["track_id"].astype(str)
    user2genreprob = estimate_user_genre_pref(hist, item2genres, genre_list, weight_col=weight_col)

    print("Preparing candidate item universe...")
    candidate_items = set(recs["item_id"].astype(str).unique().tolist())

    print("Computing P(i|c) globally...")
    p_i_given_c_global = compute_p_i_given_c(genre2items, candidate_items)

    # -----------------------------
    # 🔥 GLOBAL relevance min/max
    # -----------------------------
    global_min_rel = recs["score"].min()
    global_max_rel = recs["score"].max()
    print(f"Global score range: {global_min_rel:.4f} → {global_max_rel:.4f}")

    print("Running xQuAD reranking...")
    rows = []

    for uid, grp in tqdm(recs.groupby("user_id"), total=recs["user_id"].nunique()):
        user_candidates = list(zip(grp["item_id"].astype(str).tolist(), grp["score"].tolist()))

        user_genre_prob = user2genreprob.get(uid, None)
        if user_genre_prob is None:
            user_genre_prob = {g: 1.0 / len(genre_list) for g in genre_list}

        selected = xquad_rerank_for_user(
            user_candidates=user_candidates,
            user_genre_prob=user_genre_prob,
            p_i_given_c=p_i_given_c_global,
            global_min_rel=global_min_rel,
            global_max_rel=global_max_rel,
            lambda_xquad=lambda_xquad,
            K=K,
            genre_list=genre_list
        )

        for item_id, score, rank in selected:
            rows.append({
                "user_id": uid,
                "item_id": item_id,
                "score": score,
                "rank": rank
            })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"Saved xQuAD reranked to {output_path}")
