import pandas as pd
from pathlib import Path

from context_scoring import generate_final_recommendations
from explain_recommendations import explain_recommendations

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def run_scenario(title, traveler_id):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    travelers = pd.read_csv(DATA_DIR / "travelers.csv")
    traveler = travelers[
        travelers["traveler_id"] == traveler_id
    ].iloc[0]

    print("\nTraveler profile:")
    print(f"Destination: {traveler['destination']}")
    print(f"Trip duration: {traveler['trip_duration']} days")
    print(f"Interests: {traveler['interests']}")
    print(f"Budget: {traveler['budget']}")
    print(f"Party: {traveler['party_type']}")
    print(f"Mobility: {traveler['mobility']}")
    print(f"Preference: {traveler['explicit_preferences']}")

    recommendations = explain_recommendations(
        traveler_id,
        top_k=5
    )

    # # Add feature-grounded explanations
    # recommendations = explain_recommendations(
    #     traveler_id,
    #     recommendations
    # )

    print("\nTop recommendations:\n")

    for i, row in recommendations.iterrows():
        print(
            f"{i + 1}. {row['name']} ({row['category']})"
        )
        print(f"   Utility: {row['final_utility']:.3f}")
        print(
            f"   Final score: "
            f"{row['diversity_adjusted_score']:.3f}"
        )
        print(f"   Preference: {row['preference_score']:.3f}")
        print(f"   Context: {row['context_compatibility']:.3f}")
        print(f"   Confidence: {row['confidence']:.3f}")
        print(
            "   Reasons: "
            + "; ".join(row["reasons"])
        )


if __name__ == "__main__":

    run_scenario(
        "SCENARIO 1 — History & Less-Touristy Traveler",
        "U0001"
    )

    run_scenario(
        "SCENARIO 2 — Family & Local-Experience Traveler",
        "U0002"
    )

    run_scenario(
        "SCENARIO 3 — Budget Family & Outdoor Traveler",
        "U0003"
    )

    run_scenario("SCENARIO 4 — Cold Start Traveler", "U0101")