import pandas as pd
import numpy as np
from pathlib import Path

from rank_recommendations import generate_recommendations

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TOP_K = 10


def get_popularity_recommendations(
    traveler_id,
    travelers,
    pois,
    top_k=10
):
    traveler = travelers[
        travelers["traveler_id"] == traveler_id
    ]

    if traveler.empty:
        return pd.DataFrame()

    destination = traveler.iloc[0]["destination"]

    destination_pois = pois[
        pois["destination"] == destination
    ].copy()

    return (
        destination_pois
        .sort_values(
            ["popularity", "rating"],
            ascending=False
        )
        .head(top_k)
    )


def get_recommendations():

    test = pd.read_csv(
        DATA_DIR / "test_interactions.csv"
    )

    travelers = pd.read_csv(
        DATA_DIR / "travelers.csv"
    )

    pois = pd.read_csv(
        DATA_DIR / "pois.csv"
    )

    personalized = {}
    popularity = {}

    for traveler_id in test["traveler_id"].unique():

        recommendations = generate_recommendations(
            traveler_id,
            top_k=TOP_K
        )

        personalized[traveler_id] = recommendations

        popularity[traveler_id] = (
            get_popularity_recommendations(
                traveler_id,
                travelers,
                pois,
                TOP_K
            )
        )

    return test, pois, personalized, popularity


def catalog_coverage(recommendations, total_pois):

    recommended_pois = set()

    for frame in recommendations.values():
        if not frame.empty:
            recommended_pois.update(
                frame["poi_id"].tolist()
            )

    return len(recommended_pois) / total_pois


def long_tail_rate(recommendations, pois):

    popularity_threshold = pois["popularity"].quantile(0.50)

    total = 0
    long_tail = 0

    for frame in recommendations.values():

        if frame.empty:
            continue

        poi_ids = frame["poi_id"].tolist()

        selected = pois[
            pois["poi_id"].isin(poi_ids)
        ]

        total += len(selected)

        long_tail += (
            selected["popularity"] <
            popularity_threshold
        ).sum()

    if total == 0:
        return 0.0

    return long_tail / total


def category_diversity(recommendations):

    diversity_scores = []

    for frame in recommendations.values():

        if frame.empty:
            continue

        categories = frame["category"].nunique()

        diversity_scores.append(
            categories / min(TOP_K, len(frame))
        )

    if not diversity_scores:
        return 0.0

    return np.mean(diversity_scores)


def constraint_compatibility(
    recommendations,
    travelers,
    pois
):

    scores = []

    for traveler_id, frame in recommendations.items():

        if frame.empty:
            continue

        traveler = travelers[
            travelers["traveler_id"] == traveler_id
        ]

        if traveler.empty:
            continue

        traveler = traveler.iloc[0]

        selected = pois[
            pois["poi_id"].isin(
                frame["poi_id"]
            )
        ].copy()

        if selected.empty:
            continue

        compatible = 0

        for _, poi in selected.iterrows():

            score = 1.0

            # Budget compatibility
            if traveler["budget"] == "low":
                if poi["price_level"] > 2:
                    score *= 0.5

            elif traveler["budget"] == "medium":
                if poi["price_level"] > 3:
                    score *= 0.5

            # Mobility compatibility
            if (
                traveler["mobility"] == "walking"
                and poi["public_transport_access"] == 0
            ):
                score *= 0.7

            # Family compatibility
            if (
                traveler["party_type"] == "family"
                and poi["family_friendly"] == 0
            ):
                score *= 0.5

            compatible += score

        scores.append(
            compatible / len(selected)
        )

    if not scores:
        return 0.0

    return np.mean(scores)


def main():

    test, pois, personalized, popularity = (
        get_recommendations()
    )

    travelers = pd.read_csv(
        DATA_DIR / "travelers.csv"
    )

    total_pois = len(pois)

    print("\n========================================")
    print("       EXTENDED POI EVALUATION")
    print("========================================")

    print("\nPersonalized Ranker")
    print("-------------------")

    print(
        f"Catalog Coverage: "
        f"{catalog_coverage(personalized, total_pois):.4f}"
    )

    print(
        f"Long-tail Rate: "
        f"{long_tail_rate(personalized, pois):.4f}"
    )

    print(
        f"Category Diversity: "
        f"{category_diversity(personalized):.4f}"
    )

    print(
        f"Constraint Compatibility: "
        f"{constraint_compatibility(personalized, travelers, pois):.4f}"
    )

    print("\nPopularity Baseline")
    print("-------------------")

    print(
        f"Catalog Coverage: "
        f"{catalog_coverage(popularity, total_pois):.4f}"
    )

    print(
        f"Long-tail Rate: "
        f"{long_tail_rate(popularity, pois):.4f}"
    )

    print(
        f"Category Diversity: "
        f"{category_diversity(popularity):.4f}"
    )

    print(
        f"Constraint Compatibility: "
        f"{constraint_compatibility(popularity, travelers, pois):.4f}"
    )


if __name__ == "__main__":
    main()