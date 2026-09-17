import pandas as pd
import numpy as np
from pathlib import Path

from rank_recommendations import generate_recommendations
from diversity_reranking import diversity_rerank
from cold_start import get_traveler_history_strength
from cold_start import is_cold_start_traveler, cold_start_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def score_budget(traveler_budget, price_level):
    if traveler_budget == "low":
        if price_level <= 1:
            return 1.0
        elif price_level == 2:
            return 0.8
        elif price_level == 3:
            return 0.4
        return 0.1

    if traveler_budget == "medium":
        if price_level <= 2:
            return 1.0
        elif price_level == 3:
            return 0.8
        return 0.4

    # High budget
    return 1.0


def score_mobility(mobility, public_transport_access, distance):
    if mobility == "walking":
        if distance <= 5:
            return 1.0
        elif distance <= 10:
            return 0.8
        return 0.5

    if mobility == "public_transport":
        if public_transport_access == 1:
            return 1.0
        return 0.6

    # Car
    return 1.0


def score_party(party_type, family_friendly):
    if party_type == "family":
        return 1.0 if family_friendly == 1 else 0.5

    return 1.0


def score_distance(distance):
    if distance <= 5:
        return 1.0
    elif distance <= 10:
        return 0.85
    elif distance <= 20:
        return 0.65
    elif distance <= 40:
        return 0.4

    return 0.2


def calculate_context_scores(
    traveler,
    recommendations,
    pois
):

    poi_columns = [
        "poi_id",
        "name",
        "price_level",
        "public_transport_access",
        "family_friendly"
    ]

    result = recommendations.copy()

    # Add POI context features that are not already present.
    missing_columns = [
        col
        for col in poi_columns
        if col != "poi_id" and col not in result.columns
    ]

    if missing_columns:
        result = result.merge(
            pois[["poi_id"] + missing_columns],
            on="poi_id",
            how="left"
        )

    result["budget_compatibility"] = result.apply(
        lambda row: score_budget(
            traveler["budget"],
            row["price_level"]
        ),
        axis=1
    )

    result["mobility_compatibility"] = result.apply(
        lambda row: score_mobility(
            traveler["mobility"],
            row["public_transport_access"],
            row["geographic_distance"]
        ),
        axis=1
    )

    result["party_compatibility"] = result.apply(
        lambda row: score_party(
            traveler["party_type"],
            row["family_friendly"]
        ),
        axis=1
    )

    # geographic_distance comes from the ranking feature set.
    if "geographic_distance" in result.columns:
        result["distance_compatibility"] = (
            result["geographic_distance"]
            .apply(score_distance)
        )
    else:
        result["distance_compatibility"] = 0.5

    # Normalize model score into [0, 1].
    min_score = result["ranking_score"].min()
    max_score = result["ranking_score"].max()

    if max_score == min_score:
        result["preference_score"] = 0.5
    else:
        result["preference_score"] = (
            (result["ranking_score"] - min_score)
            / (max_score - min_score)
        )

    # Practical compatibility.
    result["context_compatibility"] = (
        0.35 * result["budget_compatibility"]
        + 0.25 * result["mobility_compatibility"]
        + 0.20 * result["party_compatibility"]
        + 0.20 * result["distance_compatibility"]
    )

    # Final utility:
    # Preference remains the strongest signal, while
    # contextual constraints influence the final ranking.
    # Explicit preference.
    explicit_preference_score = (
        recommendations["preference_match"]
        if "preference_match" in recommendations.columns
        else 0.5
    )

    result["explicit_preference_score"] = (
        explicit_preference_score
    )

    # Historical support determines how much we trust
    # behavior-based personalization.
    history_strength = get_traveler_history_strength(
        traveler["traveler_id"]
    )

    # Adaptive weighting:
    #
    # More history:
    #   learned preference = 0.50
    #   explicit preference = 0.20
    #   context = 0.30
    #
    # Little/no history:
    #   learned preference = 0.20
    #   explicit preference = 0.50
    #   context = 0.30
    #
    # The weights transition smoothly according to
    # available historical evidence.

    learned_weight = (
        0.20
        + 0.30 * history_strength
    )

    explicit_weight = (
        0.50
        - 0.30 * history_strength
    )

    context_weight = 0.30

    result["history_strength"] = history_strength

    result["final_utility"] = (
        learned_weight * result["preference_score"]
        + explicit_weight * result["explicit_preference_score"]
        + context_weight * result["context_compatibility"]
    )

    # Confidence is based primarily on available history,
    # with recommendation/context signals providing additional support.
    result["confidence"] = (
        0.50 * history_strength
        + 0.30 * result["preference_score"]
        + 0.20 * result["context_compatibility"]
    )

    return result


def generate_final_recommendations(
    traveler_id,
    top_k=10
):
    travelers = pd.read_csv(
        DATA_DIR / "travelers.csv"
    )

    pois = pd.read_csv(
        DATA_DIR / "pois.csv"
    )

    traveler_row = travelers[
        travelers["traveler_id"] == traveler_id
    ]

    if traveler_row.empty:
        raise ValueError(
            f"Unknown traveler: {traveler_id}"
        )

    traveler = traveler_row.iloc[0]

    # --------------------------------------------------
    # 1. Candidate ranking
    # --------------------------------------------------

    if is_cold_start_traveler(traveler_id):

        # No historical behavior available.
        # Use content + explicit preference + constraints.
        recommendations = cold_start_score(
            traveler,
            pois
        ).head(30)

        recommendations["ranking_score"] = (
            recommendations["cold_start_score"]
        )

    else:

        # Normal path: behavioral features + XGBoost.
        recommendations = generate_recommendations(
            traveler_id,
            top_k=30
        )

    # --------------------------------------------------
    # 2. Attach engineered features
    # --------------------------------------------------

    feature_data = pd.read_csv(
        DATA_DIR / "training_data_with_behavior.csv"
    )

    feature_data = feature_data[
        feature_data["traveler_id"] == traveler_id
    ].copy()

    feature_columns = [
        "poi_id",
        "preference_match",
        "category_affinity",
        "geographic_distance",
        "total_interactions",
        "tourist_level"
    ]

    # Existing traveler has historical features.
    if not is_cold_start_traveler(traveler_id):

        feature_data = (
            feature_data[feature_columns]
            .drop_duplicates("poi_id")
        )

        columns_to_remove = [
            "preference_match",
            "category_affinity",
            "geographic_distance",
            "total_interactions",
            "tourist_level"
        ]

        recommendations = recommendations.drop(
            columns=columns_to_remove,
            errors="ignore"
        )

        recommendations = recommendations.merge(
            feature_data,
            on="poi_id",
            how="left"
        )

    else:

        # Cold-start traveler has no behavioral features.
        # Derive only the features required by the
        # context/explanation layer.
        recommendations["preference_match"] = (
            recommendations["explicit_preference_score"]
        )

        recommendations["category_affinity"] = 0.0

        recommendations["total_interactions"] = 0

        recommendations["geographic_distance"] = 10.0

    # --------------------------------------------------
    # 3. Context scoring
    # --------------------------------------------------

    recommendations = calculate_context_scores(
        traveler,
        recommendations,
        pois
    )

    # --------------------------------------------------
    # 4. Initial utility ranking
    # --------------------------------------------------

    recommendations = recommendations.sort_values(
        "final_utility",
        ascending=False
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 5. Diversity-aware post-ranking
    # --------------------------------------------------

    recommendations = diversity_rerank(
        recommendations,
        top_k=top_k,
        diversity_weight=0.08
    )

    return recommendations



if __name__ == "__main__":

    traveler_id = "U0001"

    recommendations = (
        generate_final_recommendations(
            traveler_id,
            top_k=10
        )
    )

    print(
        f"\nFinal recommendations for {traveler_id}:\n"
    )

    print(
        recommendations.to_string(
            index=False
        )
    )