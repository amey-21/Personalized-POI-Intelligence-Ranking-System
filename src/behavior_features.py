import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def build_behavior_features():
    interactions = pd.read_csv(DATA_DIR / "train_interactions.csv")
    pois = pd.read_csv(DATA_DIR / "pois.csv")

    # Map interaction types to behavioral strength
    interaction_strength = {
        "view": 1,
        "click": 2,
        "save": 4,
        "share": 5,
        "navigate": 6,
        "visit": 8,
        "booking": 10,
        "dismiss": -3
    }

    interactions["interaction_strength"] = (
        interactions["interaction_type"]
        .map(interaction_strength)
        .fillna(0)
    )

    # Join POI information to understand what the traveler interacted with
    history = interactions.merge(
        pois[
            [
                "poi_id",
                "category",
                "price_level",
                "tourist_level",
                "rating",
                "popularity"
            ]
        ],
        on="poi_id",
        how="left"
    )

    # Total behavioral activity
    traveler_stats = history.groupby("traveler_id").agg(
        total_interactions=("poi_id", "count"),
        total_behavior_strength=("interaction_strength", "sum"),
        avg_poi_rating=("rating", "mean"),
        avg_poi_popularity=("popularity", "mean"),
        avg_price_level=("price_level", "mean"),
        avg_tourist_level=("tourist_level", "mean")
    ).reset_index()

    # Category affinity
    category_strength = (
        history
        .groupby(["traveler_id", "category"])["interaction_strength"]
        .sum()
        .reset_index()
    )

    # Normalize category strength within each traveler
    category_strength["category_affinity"] = (
        category_strength
        .groupby("traveler_id")["interaction_strength"]
        .transform(
            lambda x: x / (x.abs().sum() + 1e-8)
        )
    )

    category_strength = category_strength[
        ["traveler_id", "category", "category_affinity"]
    ]

    # Save outputs
    traveler_stats.to_csv(
        DATA_DIR / "traveler_behavior_stats.csv",
        index=False
    )

    category_strength.to_csv(
        DATA_DIR / "traveler_category_affinity.csv",
        index=False
    )

    print("Behavioral features generated.")

    print("\nTraveler behavior stats:")
    print(traveler_stats.head())

    print("\nCategory affinity:")
    print(category_strength.head())

    print("\nFiles saved:")
    print("data/traveler_behavior_stats.csv")
    print("data/traveler_category_affinity.csv")


if __name__ == "__main__":
    build_behavior_features()