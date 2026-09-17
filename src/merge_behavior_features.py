import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def merge_behavior_features():

    training = pd.read_csv(DATA_DIR / "training_data.csv")
    pois = pd.read_csv(DATA_DIR / "pois.csv")

    traveler_stats = pd.read_csv(
        DATA_DIR / "traveler_behavior_stats.csv"
    )

    category_affinity = pd.read_csv(
        DATA_DIR / "traveler_category_affinity.csv"
    )

    # Add POI category to the training pairs
    training = training.merge(
        pois[["poi_id", "category"]],
        on="poi_id",
        how="left"
    )

    # Add general traveler behavioral statistics
    training = training.merge(
        traveler_stats,
        on="traveler_id",
        how="left"
    )

    # Add traveler affinity for the POI's category
    training = training.merge(
        category_affinity,
        on=["traveler_id", "category"],
        how="left"
    )

    # No historical interaction with this category
    # means neutral affinity.
    training["category_affinity"] = (
        training["category_affinity"]
        .fillna(0.0)
    )

    output_path = DATA_DIR / "training_data_with_behavior.csv"

    training.to_csv(output_path, index=False)

    print("Behavioral features merged successfully.")

    print(f"\nRows: {len(training)}")
    print(f"Columns: {len(training.columns)}")

    print("\nColumns:")
    print(training.columns.tolist())

    behavioral_columns = [
        "total_interactions",
        "total_behavior_strength",
        "avg_poi_rating",
        "avg_poi_popularity",
        "avg_price_level",
        "avg_tourist_level",
        "category_affinity"
    ]

    print("\nMissing values in behavioral features:")
    print(training[behavioral_columns].isnull().sum())

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    merge_behavior_features()