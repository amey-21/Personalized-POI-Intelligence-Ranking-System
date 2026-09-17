import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def temporal_split(
    interactions,
    train_ratio=0.8,
):
    """
    Split each traveler's interaction history
    chronologically into train and test sets.
    """

    interactions = interactions.copy()

    interactions["timestamp"] = pd.to_datetime(
        interactions["timestamp"]
    )

    interactions = interactions.sort_values(
        ["traveler_id", "timestamp"]
    ).reset_index(drop=True)

    train_parts = []
    test_parts = []

    for traveler_id, group in interactions.groupby(
        "traveler_id"
    ):

        group = group.sort_values("timestamp")

        split_index = int(
            len(group) * train_ratio
        )

        # Ensure every traveler has at least
        # one training example and one test example.
        split_index = max(
            1,
            min(
                split_index,
                len(group) - 1,
            ),
        )

        train_parts.append(
            group.iloc[:split_index]
        )

        test_parts.append(
            group.iloc[split_index:]
        )

    train = pd.concat(
        train_parts,
        ignore_index=True,
    )

    test = pd.concat(
        test_parts,
        ignore_index=True,
    )

    return train, test


if __name__ == "__main__":

    interactions = pd.read_csv(
        DATA_DIR / "interactions.csv"
    )

    train, test = temporal_split(
        interactions,
        train_ratio=0.8,
    )

    train.to_csv(
        DATA_DIR / "train_interactions.csv",
        index=False,
    )

    test.to_csv(
        DATA_DIR / "test_interactions.csv",
        index=False,
    )

    print("Temporal split completed.")

    print(
        f"\nTotal interactions: {len(interactions)}"
    )

    print(
        f"Training interactions: {len(train)}"
    )

    print(
        f"Test interactions: {len(test)}"
    )

    print("\nTraining date range:")

    print(
        train["timestamp"].min(),
        "→",
        train["timestamp"].max(),
    )

    print("\nTest date range:")

    print(
        test["timestamp"].min(),
        "→",
        test["timestamp"].max(),
    )

    print("\nTraining interactions per traveler:")

    print(
        train.groupby("traveler_id")
        .size()
        .describe()
    )

    print("\nTest interactions per traveler:")

    print(
        test.groupby("traveler_id")
        .size()
        .describe()
    )