import pandas as pd
from pathlib import Path

from context_scoring import generate_final_recommendations

DATA_DIR = Path(__file__).resolve().parent.parent / "data"



def generate_reasons(row, traveler):
    reasons = []

    # Explicit preference
    explicit_score = row.get( 
        "explicit_preference_score",
        row.get("preference_match", 0.0)
    )

    if explicit_score >= 0.75:
        reasons.append(
            "Strong match with your explicit preference"
        )
    elif explicit_score >= 0.50:
        reasons.append(
            "Good match with your explicit preference"
        )
    elif explicit_score >= 0.30:
        reasons.append(
            "Some alignment with your explicit preference"
        )

    # Outdoor preference
    if (
        "outdoor" in str(
            traveler["explicit_preferences"]
        ).lower()
        and row["category"] in ["nature", "activity"]
    ):
        reasons.append(
            "Matches your outdoor-activity preference"
        )

    # Historical behavior
    category_affinity = row.get(
        "category_affinity",
        0.0
    )

    if category_affinity >= 0.20:
        reasons.append(
            "Matches categories you've engaged with frequently"
        )
    elif category_affinity > 0:
        reasons.append(
            "Shows some similarity to your past interests"
        )

    # Budget
    if traveler["budget"] == "low":
        reasons.append(
            "Fits a budget-conscious trip"
        )
    elif traveler["budget"] == "medium":
        reasons.append(
            "Fits a moderate budget"
        )

    # Mobility
    if traveler["mobility"] == "walking":
        distance = row.get(
            "geographic_distance",
            None
        )

        if distance is not None:
            if distance <= 10:
                reasons.append(
                    "Relatively close to your starting location"
                )
            else:
                reasons.append(
                    "Accessible for your travel mode"
                )

    elif traveler["mobility"] == "public_transport":
        reasons.append(
            "Compatible with public-transport travel"
        )

    # Family
    if traveler["party_type"] == "family":
        reasons.append(
            "Suitable for family travel"
        )

    # Less touristy preference
    if (
        "less touristy"
        in str(
            traveler["explicit_preferences"]
        ).lower()
        and row["tourist_level"] < 0.4
    ):
        reasons.append(
            "Lower tourist concentration"
        )

    # Local experience
    if (
        "local"
        in str(
            traveler["explicit_preferences"]
        ).lower()
        and row["tourist_level"] < 0.5
    ):
        reasons.append(
            "Offers a relatively local experience"
        )

    # Confidence
    if row["confidence"] >= 0.70:
        reasons.append("Higher confidence")
    elif row["confidence"] >= 0.45:
        reasons.append("Moderate confidence")
    else:
        reasons.append(
            "Limited historical evidence"
        )

    return reasons




def explain_recommendations(traveler_id, top_k=10):

    travelers = pd.read_csv(
        DATA_DIR / "travelers.csv"
    )

    traveler_rows = travelers[
        travelers["traveler_id"] == traveler_id
    ]

    if traveler_rows.empty:
        raise ValueError(
            f"Unknown traveler: {traveler_id}"
        )

    traveler = traveler_rows.iloc[0]

    recommendations = generate_final_recommendations(
        traveler_id,
        top_k=top_k
    )

    recommendations["reasons"] = recommendations.apply(
        lambda row: generate_reasons(
            row,
            traveler
        ),
        axis=1
    )

    return recommendations