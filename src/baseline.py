import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def popularity_recommendations(
    traveler_id,
    top_k=10,
):
    pois = pd.read_csv(DATA_DIR / "pois.csv")
    travelers = pd.read_csv(DATA_DIR / "travelers.csv")

    traveler = travelers[
        travelers["traveler_id"] == traveler_id
    ]

    if traveler.empty:
        raise ValueError(
            f"Unknown traveler: {traveler_id}"
        )

    destination = traveler.iloc[0]["destination"]

    # Popularity baseline:
    # recommend the most popular POIs in the traveler's destination.
    recommendations = (
        pois[pois["destination"] == destination]
        .sort_values(
            ["popularity", "rating"],
            ascending=False,
        )
        .head(top_k)
        .copy()
    )

    recommendations["score"] = recommendations[
        "popularity"
    ]

    return recommendations[
        [
            "poi_id",
            "name",
            "category",
            "destination",
            "popularity",
            "rating",
            "score",
        ]
    ]


if __name__ == "__main__":

    recommendations = popularity_recommendations(
        traveler_id="U0004",
        top_k=10,
    )

    print("\nPopularity Baseline")
    print("===================")

    print(
        recommendations.to_string(
            index=False
        )
    )