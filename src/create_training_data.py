import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


INTERACTION_LABELS = {
    "dismiss": 0.00,
    "view": 0.15,
    "click": 0.30,
    "save": 0.55,
    "share": 0.65,
    "visit": 0.85,
    "booking": 1.00,
}


def create_training_data():
    features = pd.read_csv(
        DATA_DIR / "traveler_poi_features.csv"
    )

    interactions = pd.read_csv(
        DATA_DIR / "interactions.csv"
    )

    interactions["relevance_label"] = (
        interactions["interaction_type"]
        .map(INTERACTION_LABELS)
    )

    # A traveler may interact with the same POI
    # more than once. Keep the strongest observed signal.
    interaction_labels = (
        interactions
        .groupby(
            ["traveler_id", "poi_id"]
        )["relevance_label"]
        .max()
        .reset_index()
    )

    training_data = features.merge(
        interaction_labels,
        on=["traveler_id", "poi_id"],
        how="left",
    )

    # No interaction is treated as unknown/negative,
    # rather than assuming that the traveler disliked it.
    training_data["relevance_label"] = (
        training_data["relevance_label"]
        .fillna(0.0)
    )

    return training_data


if __name__ == "__main__":

    training_data = create_training_data()

    output_path = (
        DATA_DIR / "training_data.csv"
    )

    training_data.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Training dataset shape: "
        f"{training_data.shape}"
    )

    print("\nLabel distribution:")

    print(
        training_data[
            "relevance_label"
        ].value_counts()
        .sort_index()
    )

    print("\nMean relevance label:")

    print(
        training_data[
            "relevance_label"
        ].mean()
    )

    print("\nTraining data sample:")

    print(
        training_data.head(10).to_string(
            index=False
        )
    )