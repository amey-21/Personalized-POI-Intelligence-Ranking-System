import pandas as pd


def diversity_rerank(
    recommendations: pd.DataFrame,
    top_k: int = 5,
    diversity_weight: float = 0.08
) -> pd.DataFrame:

    if recommendations.empty:
        return recommendations.copy()

    candidates = recommendations.copy().reset_index(drop=True)

    selected_indices = []
    category_counts = {}

    while len(selected_indices) < min(top_k, len(candidates)):

        best_idx = None
        best_score = float("-inf")

        for idx, row in candidates.iterrows():

            if idx in selected_indices:
                continue

            category = row["category"]

            repetition_penalty = category_counts.get(
                category,
                0
            )

            adjusted_score = (
                row["final_utility"]
                - diversity_weight * repetition_penalty
            )

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_idx = idx

        selected_indices.append(best_idx)

        category = candidates.loc[
            best_idx,
            "category"
        ]

        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    result = candidates.loc[
        selected_indices
    ].copy()

    # Reconstruct the score that was used during selection.
    running_counts = {}

    diversity_scores = []

    for _, row in result.iterrows():

        category = row["category"]

        penalty = running_counts.get(
            category,
            0
        )

        score = (
            row["final_utility"]
            - diversity_weight * penalty
        )

        diversity_scores.append(score)

        running_counts[category] = (
            running_counts.get(category, 0) + 1
        )

    result["diversity_adjusted_score"] = diversity_scores

    # Final ranking is based on the score actually used
    # during diversity-aware selection.
    result = result.sort_values(
        "diversity_adjusted_score",
        ascending=False
    ).reset_index(drop=True)

    return result