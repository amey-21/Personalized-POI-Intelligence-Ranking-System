import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


random.seed(42)
np.random.seed(42)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


INTERACTION_WEIGHTS = {
    "view": 1,
    "click": 2,
    "save": 4,
    "share": 5,
    "navigate": 6,
    "visit": 8,
    "booking": 10,
    "dismiss": -3,
}


INTEREST_CATEGORY_MAP = {
    "food": ["food"],
    "history": ["historical", "museum"],
    "architecture": ["historical", "museum"],
    "museums": ["museum", "historical"],
    "nature": ["nature"],
    "shopping": ["shopping"],
    "activities": ["activity", "entertainment"],
    "neighborhoods": ["neighborhood", "food", "shopping"],
    "entertainment": ["entertainment", "activity"],
}


def parse_interests(interests):
    return interests.split("|")


def preference_score(traveler, poi):
    score = 0.0

    interests = parse_interests(traveler["interests"])

    for interest in interests:
        compatible_categories = INTEREST_CATEGORY_MAP.get(
            interest,
            [],
        )

        if poi["category"] in compatible_categories:
            score += 0.30

    budget_map = {
        "low": 1,
        "medium": 2,
        "high": 3,
    }

    traveler_budget = budget_map[traveler["budget"]]

    if poi["price_level"] <= traveler_budget:
        score += 0.15

    if traveler["mobility"] == "public_transport":
        if poi["public_transport_access"]:
            score += 0.15

    elif traveler["mobility"] == "walking":
        if poi["public_transport_access"]:
            score += 0.05

    elif traveler["mobility"] == "car":
        score += 0.10

    if traveler["party_type"] == "family":
        if poi["family_friendly"]:
            score += 0.15
    else:
        score += 0.05

    preference = traveler["explicit_preferences"].lower()

    if "less touristy" in preference:
        score += (1 - poi["tourist_level"]) * 0.20

    elif "local experiences" in preference:
        score += (1 - poi["tourist_level"]) * 0.20

    elif "famous landmarks" in preference:
        score += poi["popularity"] * 0.20

    elif "family friendly" in preference:
        if poi["family_friendly"]:
            score += 0.20

    elif "outdoor activities" in preference:
        if poi["category"] in ["nature", "activity"]:
            score += 0.20

    elif "cultural experiences" in preference:
        if poi["category"] in [
            "historical",
            "museum",
            "neighborhood",
        ]:
            score += 0.20

    elif "budget friendly" in preference:
        if poi["price_level"] <= 2:
            score += 0.20

    elif "highly rated" in preference:
        score += (poi["rating"] / 5.0) * 0.20

    score += poi["popularity"] * 0.05

    return min(score, 1.0)


def choose_interaction(score):
    random_value = random.random()

    if score < 0.25:
        if random_value < 0.15:
            return "view"
        elif random_value < 0.25:
            return "dismiss"
        return None

    if score < 0.50:
        if random_value < 0.55:
            return "view"
        elif random_value < 0.80:
            return "click"
        elif random_value < 0.90:
            return "dismiss"
        return None

    if score < 0.75:
        if random_value < 0.25:
            return "view"
        elif random_value < 0.50:
            return "click"
        elif random_value < 0.72:
            return "save"
        elif random_value < 0.90:
            return "visit"
        return "share"

    if random_value < 0.15:
        return "click"
    elif random_value < 0.35:
        return "save"
    elif random_value < 0.55:
        return "share"
    elif random_value < 0.80:
        return "visit"
    else:
        return "booking"


def generate_interactions(
    travelers,
    pois,
    interactions_per_traveler=120,
):
    rows = []

    pois_by_destination = {
        destination: group
        for destination, group in pois.groupby("destination")
    }

    # Every traveler has their own chronological history.
    base_date = datetime(2025, 1, 1)

    for _, traveler in travelers.iterrows():

        destination_pois = pois_by_destination[
            traveler["destination"]
        ]

        sample_size = min(
            interactions_per_traveler,
            len(destination_pois),
        )

        sampled_pois = destination_pois.sample(
            n=sample_size,
            random_state=random.randint(0, 100000),
        )

        # Generate an ordered sequence of dates.
        interaction_dates = sorted(
            [
                base_date
                + timedelta(
                    days=random.randint(0, 365),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                )
                for _ in range(sample_size)
            ]
        )

        for (_, poi), timestamp in zip(
            sampled_pois.iterrows(),
            interaction_dates,
        ):

            score = preference_score(
                traveler,
                poi,
            )

            # Behavioral noise prevents deterministic labels.
            score += np.random.normal(0, 0.08)
            score = float(np.clip(score, 0, 1))

            interaction_type = choose_interaction(score)

            if interaction_type is None:
                continue

            rows.append(
                {
                    "traveler_id": traveler["traveler_id"],
                    "poi_id": poi.poi_id,
                    "interaction_type": interaction_type,
                    "timestamp": timestamp.isoformat(),
                    "interaction_weight": INTERACTION_WEIGHTS[
                        interaction_type
                    ],
                }
            )

    interactions = pd.DataFrame(rows)

    # Final global chronological ordering.
    interactions = interactions.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return interactions


if __name__ == "__main__":

    travelers = pd.read_csv(
        DATA_DIR / "travelers.csv"
    )

    pois = pd.read_csv(
        DATA_DIR / "pois.csv"
    )

    interactions = generate_interactions(
        travelers,
        pois,
        interactions_per_traveler=120,
    )

    output_path = DATA_DIR / "interactions.csv"

    interactions.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Generated {len(interactions)} interactions"
    )

    print(
        f"Saved to: {output_path}"
    )

    print("\nInteraction distribution:")

    print(
        interactions[
            "interaction_type"
        ].value_counts()
    )

    print("\nDate range:")

    print(
        interactions["timestamp"].min(),
        "→",
        interactions["timestamp"].max(),
    )

    print("\nFirst 10 interactions:")

    print(
        interactions.head(10).to_string(
            index=False
        )
    )