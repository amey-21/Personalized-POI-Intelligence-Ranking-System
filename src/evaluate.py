import pandas as pd
import numpy as np
from pathlib import Path

from rank_recommendations import generate_recommendations

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

K = 10

RELEVANCE_WEIGHTS = {
    "save": 3,
    "share": 3,
    "navigate": 4,
    "visit": 5,
    "booking": 6
}


def precision_at_k(recommended, relevant, k=10):
    recommended = recommended[:k]

    if not recommended:
        return 0.0

    hits = len(set(recommended) & set(relevant))

    return hits / k


def recall_at_k(recommended, relevant, k=10):
    recommended = recommended[:k]

    if not relevant:
        return 0.0

    hits = len(set(recommended) & set(relevant))

    return hits / len(relevant)


def ndcg_at_k(recommended, relevance_dict, k=10):

    recommended = recommended[:k]

    dcg = 0.0

    for rank, poi_id in enumerate(recommended, start=1):

        relevance = relevance_dict.get(poi_id, 0)

        dcg += (
            (2 ** relevance - 1)
            / np.log2(rank + 1)
        )

    ideal_relevances = sorted(
        relevance_dict.values(),
        reverse=True
    )[:k]

    idcg = 0.0

    for rank, relevance in enumerate(
        ideal_relevances,
        start=1
    ):
        idcg += (
            (2 ** relevance - 1)
            / np.log2(rank + 1)
        )

    if idcg == 0:
        return 0.0

    return dcg / idcg


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
        return []

    destination = traveler.iloc[0]["destination"]

    destination_pois = pois[
        pois["destination"] == destination
    ].copy()

    recommendations = (
        destination_pois
        .sort_values(
            ["popularity", "rating"],
            ascending=False
        )
        .head(top_k)
    )

    return recommendations["poi_id"].tolist()


def evaluate():

    test = pd.read_csv(
        DATA_DIR / "test_interactions.csv"
    )

    travelers = pd.read_csv(
        DATA_DIR / "travelers.csv"
    )

    pois = pd.read_csv(
        DATA_DIR / "pois.csv"
    )

    personalized_precision = []
    personalized_recall = []
    personalized_ndcg = []

    popularity_precision = []
    popularity_recall = []
    popularity_ndcg = []

    evaluated_travelers = 0

    for traveler_id in test["traveler_id"].unique():

        traveler_test = test[
            test["traveler_id"] == traveler_id
        ].copy()

        # Strong future interactions are treated as relevant.
        relevant = traveler_test[
            traveler_test["interaction_type"].isin(
                RELEVANCE_WEIGHTS.keys()
            )
        ].drop_duplicates("poi_id")

        if relevant.empty:
            continue

        relevant_pois = relevant["poi_id"].tolist()

        relevance_dict = {}

        for _, row in relevant.iterrows():

            relevance_dict[row["poi_id"]] = (
                RELEVANCE_WEIGHTS[
                    row["interaction_type"]
                ]
            )

        # Personalized model
        personalized = generate_recommendations(
            traveler_id,
            top_k=K
        )

        personalized_pois = (
            personalized["poi_id"].tolist()
        )

        # Popularity baseline
        popularity_pois = get_popularity_recommendations(
            traveler_id,
            travelers,
            pois,
            top_k=K
        )

        # Personalized metrics
        personalized_precision.append(
            precision_at_k(
                personalized_pois,
                relevant_pois,
                K
            )
        )

        personalized_recall.append(
            recall_at_k(
                personalized_pois,
                relevant_pois,
                K
            )
        )

        personalized_ndcg.append(
            ndcg_at_k(
                personalized_pois,
                relevance_dict,
                K
            )
        )

        # Popularity metrics
        popularity_precision.append(
            precision_at_k(
                popularity_pois,
                relevant_pois,
                K
            )
        )

        popularity_recall.append(
            recall_at_k(
                popularity_pois,
                relevant_pois,
                K
            )
        )

        popularity_ndcg.append(
            ndcg_at_k(
                popularity_pois,
                relevance_dict,
                K
            )
        )

        evaluated_travelers += 1

    print("\n========================================")
    print("       POI RANKING EVALUATION")
    print("========================================")

    print(f"\nEvaluated travelers: {evaluated_travelers}")

    print("\nPersonalized Ranker:")
    print(
        f"Precision@10: "
        f"{np.mean(personalized_precision):.4f}"
    )
    print(
        f"Recall@10:    "
        f"{np.mean(personalized_recall):.4f}"
    )
    print(
        f"NDCG@10:      "
        f"{np.mean(personalized_ndcg):.4f}"
    )

    print("\nPopularity Baseline:")
    print(
        f"Precision@10: "
        f"{np.mean(popularity_precision):.4f}"
    )
    print(
        f"Recall@10:    "
        f"{np.mean(popularity_recall):.4f}"
    )
    print(
        f"NDCG@10:      "
        f"{np.mean(popularity_ndcg):.4f}"
    )

    print("\nImprovement of Personalized Ranker:")

    p_improvement = (
        np.mean(personalized_precision)
        - np.mean(popularity_precision)
    )

    r_improvement = (
        np.mean(personalized_recall)
        - np.mean(popularity_recall)
    )

    n_improvement = (
        np.mean(personalized_ndcg)
        - np.mean(popularity_ndcg)
    )

    print(f"Precision@10: {p_improvement:+.4f}")
    print(f"Recall@10:    {r_improvement:+.4f}")
    print(f"NDCG@10:      {n_improvement:+.4f}")


if __name__ == "__main__":
    evaluate()