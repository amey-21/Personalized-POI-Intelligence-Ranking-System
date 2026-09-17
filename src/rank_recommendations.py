import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRanker
from candidate_generation import generate_candidates

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


FEATURE_COLUMNS = [
    "interest_match",
    "preference_match",
    "budget_match",
    "mobility_match",
    "party_match",
    "poi_rating",
    "poi_review_count",
    "poi_popularity",
    "tourist_level",
    "expected_duration",
    "trip_duration",
    "geographic_distance",
    "total_interactions",
    "total_behavior_strength",
    "avg_poi_rating",
    "avg_poi_popularity",
    "avg_price_level",
    "avg_tourist_level",
    "category_affinity"
]


def load_model():
    model = XGBRanker()
    model.load_model(MODEL_DIR / "poi_ranker.json")
    return model


def generate_recommendations(traveler_id, top_k=10):

    model = load_model()

    features = pd.read_csv(
        DATA_DIR / "training_data_with_behavior.csv"
    )

    # Generate a reduced candidate pool first.
    candidate_ids = generate_candidates(
        traveler_id,
        candidate_count=80
    )["poi_id"]

    # Keep only generated candidates.
    candidates = features[
        (features["traveler_id"] == traveler_id)
        & (features["poi_id"].isin(candidate_ids))
    ].copy()

    if candidates.empty:
        raise ValueError(
            f"No candidates generated for {traveler_id}"
        )

    X = candidates[FEATURE_COLUMNS].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    candidates["ranking_score"] = model.predict(X)

    recommendations = (
        candidates
        .sort_values(
            "ranking_score",
            ascending=False
        )
        .drop_duplicates("poi_id")
        .head(top_k)
        .reset_index(drop=True)
    )

    return recommendations[
        [
            "poi_id",
            "category",
            "ranking_score",
            "preference_match",
            "category_affinity",
            "geographic_distance",
            "total_interactions"
        ]
    ]

if __name__ == "__main__":

    traveler_id = "U0001"

    recommendations = generate_recommendations(
        traveler_id,
        top_k=10
    )

    print(f"\nTop recommendations for {traveler_id}:\n")

    print(
        recommendations.to_string(index=False)
    )