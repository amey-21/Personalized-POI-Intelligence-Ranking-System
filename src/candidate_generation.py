import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def min_max_normalize(series):
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series(0.5, index=series.index)

    return (series - min_val) / (max_val - min_val)


def generate_candidates(traveler_id, candidate_count=80):

    travelers = pd.read_csv(DATA_DIR / "travelers.csv")
    pois = pd.read_csv(DATA_DIR / "pois.csv")
    affinity = pd.read_csv(
        DATA_DIR / "traveler_category_affinity.csv"
    )

    traveler = travelers[
        travelers["traveler_id"] == traveler_id
    ]

    if traveler.empty:
        raise ValueError(f"Unknown traveler: {traveler_id}")

    traveler = traveler.iloc[0]

    # Only consider POIs in the traveler's destination.
    destination_pois = pois[
        pois["destination"] == traveler["destination"]
    ].copy()

    # ---------------------------------------------------------
    # 1. Interest-based candidates
    # ---------------------------------------------------------

    interests = set(
        str(traveler["interests"])
        .lower()
        .split("|")
    )

    destination_pois["interest_score"] = (
        destination_pois["category"]
        .str.lower()
        .isin(interests)
        .astype(float)
    )

    interest_candidates = (
        destination_pois
        .sort_values(
            ["interest_score", "rating"],
            ascending=False
        )
        .head(30)
    )

    # ---------------------------------------------------------
    # 2. Geographic candidates
    # ---------------------------------------------------------

    lat1 = np.radians(float(traveler["start_latitude"]))
    lon1 = np.radians(float(traveler["start_longitude"]))

    lat2 = np.radians(destination_pois["latitude"])
    lon2 = np.radians(destination_pois["longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    destination_pois["distance"] = (
        6371 * 2 * np.arcsin(np.sqrt(a))
    )

    geographic_candidates = (
        destination_pois
        .sort_values("distance")
        .head(25)
    )

    # ---------------------------------------------------------
    # 3. Historical category affinity
    # ---------------------------------------------------------

    traveler_affinity = affinity[
        affinity["traveler_id"] == traveler_id
    ]

    affinity_pois = destination_pois.merge(
        traveler_affinity[
            ["category", "category_affinity"]
        ],
        on="category",
        how="left"
    )

    affinity_pois["category_affinity"] = (
        affinity_pois["category_affinity"]
        .fillna(0)
    )

    behavioral_candidates = (
        affinity_pois
        .sort_values(
            "category_affinity",
            ascending=False
        )
        .head(25)
    )

    # ---------------------------------------------------------
    # 4. Popularity candidates
    # ---------------------------------------------------------

    popularity_candidates = (
        destination_pois
        .sort_values(
            ["popularity", "rating"],
            ascending=False
        )
        .head(20)
    )

    # ---------------------------------------------------------
    # 5. Long-tail exploration
    # ---------------------------------------------------------

    # Select less-popular POIs so that candidate generation
    # does not become popularity-only.
    long_tail_candidates = (
        destination_pois
        .sort_values("popularity")
        .head(10)
    )

    # ---------------------------------------------------------
    # Combine candidate pools
    # ---------------------------------------------------------

    candidate_ids = set()

    for frame in [
        interest_candidates,
        geographic_candidates,
        behavioral_candidates,
        popularity_candidates,
        long_tail_candidates
    ]:
        candidate_ids.update(frame["poi_id"])

    candidates = destination_pois[
        destination_pois["poi_id"].isin(candidate_ids)
    ].copy()

    # ---------------------------------------------------------
    # Candidate-generation score
    # ---------------------------------------------------------

    candidates = candidates.merge(
        traveler_affinity[
            ["category", "category_affinity"]
        ],
        on="category",
        how="left"
    )

    candidates["category_affinity"] = (
        candidates["category_affinity"]
        .fillna(0)
    )

    candidates["distance_score"] = 1 / (
        1 + candidates["distance"]
    )

    candidates["popularity_score"] = (
        min_max_normalize(candidates["popularity"])
    )

    candidates["interest_score"] = (
        candidates["category"]
        .str.lower()
        .isin(interests)
        .astype(float)
    )

    candidates["candidate_score"] = (
        0.35 * candidates["interest_score"]
        + 0.25 * candidates["category_affinity"]
        + 0.20 * candidates["distance_score"]
        + 0.20 * candidates["popularity_score"]
    )

    candidates = (
        candidates
        .sort_values("candidate_score", ascending=False)
        .head(candidate_count)
        .reset_index(drop=True)
    )

    return candidates


if __name__ == "__main__":

    traveler_id = "U0001"

    candidates = generate_candidates(
        traveler_id,
        candidate_count=80
    )

    print(f"Generated {len(candidates)} candidates for {traveler_id}")

    print(
        candidates[
            [
                "poi_id",
                "name",
                "category",
                "popularity",
                "distance",
                "candidate_score"
            ]
        ].head(10)
    )