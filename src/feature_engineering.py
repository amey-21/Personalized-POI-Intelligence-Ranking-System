import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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


BUDGET_MAP = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


def interest_match(traveler, poi):
    """
    Measures how well the POI category matches
    the traveler's stated interests.
    """

    interests = traveler["interests"].split("|")

    matches = 0

    for interest in interests:
        compatible_categories = INTEREST_CATEGORY_MAP.get(
            interest,
            [],
        )

        if poi["category"] in compatible_categories:
            matches += 1

    return matches / len(interests)


def budget_match(traveler, poi):
    """
    Measures whether the POI price is compatible
    with the traveler's budget.
    """

    traveler_budget = BUDGET_MAP[traveler["budget"]]

    difference = abs(
        traveler_budget - poi["price_level"]
    )

    if difference == 0:
        return 1.0

    if difference == 1:
        return 0.6

    if difference == 2:
        return 0.2

    return 0.0


def mobility_match(traveler, poi):
    """
    Measures compatibility between the traveler's
    preferred mobility and POI accessibility.
    """

    mobility = traveler["mobility"]

    if mobility == "public_transport":
        return float(
            poi["public_transport_access"]
        )

    if mobility == "walking":
        return float(
            poi["public_transport_access"]
        )

    # Car users are less dependent on public transport.
    if mobility == "car":
        return 1.0

    return 0.5


def party_match(traveler, poi):
    """
    Family travelers receive a stronger compatibility
    score for family-friendly POIs.
    """

    if traveler["party_type"] == "family":
        return float(
            poi["family_friendly"]
        )

    return 1.0


def preference_match(traveler, poi):
    """
    Matches the traveler's explicit preference
    against POI characteristics.
    """

    preference = traveler[
        "explicit_preferences"
    ].lower()

    if "less touristy" in preference:
        return 1.0 - poi["tourist_level"]

    if "local experiences" in preference:
        return 1.0 - poi["tourist_level"]

    if "famous landmarks" in preference:
        return poi["popularity"]

    if "family friendly" in preference:
        return float(
            poi["family_friendly"]
        )

    if "outdoor activities" in preference:
        return float(
            poi["category"] in [
                "nature",
                "activity",
            ]
        )

    if "cultural experiences" in preference:
        return float(
            poi["category"] in [
                "historical",
                "museum",
                "neighborhood",
            ]
        )

    if "budget friendly" in preference:
        return float(
            poi["price_level"] <= 2
        )

    if "highly rated" in preference:
        return poi["rating"] / 5.0

    return 0.5


def geographic_distance(traveler, poi):
    """
    Calculate approximate distance between the
    traveler starting location and the POI.

    Uses the Haversine formula.
    """

    lat1 = np.radians(traveler["start_latitude"])
    lon1 = np.radians(traveler["start_longitude"])

    lat2 = np.radians(poi["latitude"])
    lon2 = np.radians(poi["longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    earth_radius_km = 6371

    return earth_radius_km * c


def build_pair_features(traveler, poi):
    """
    Create features describing a traveler–POI pair.
    """

    return {
        "traveler_id": traveler["traveler_id"],
        "poi_id": poi["poi_id"],

        # Preference/content features
        "interest_match": interest_match(
            traveler,
            poi,
        ),

        "preference_match": preference_match(
            traveler,
            poi,
        ),

        # Practical compatibility
        "budget_match": budget_match(
            traveler,
            poi,
        ),

        "mobility_match": mobility_match(
            traveler,
            poi,
        ),

        "party_match": party_match(
            traveler,
            poi,
        ),

        # POI quality signals
        "poi_rating": poi["rating"],
        "poi_review_count": np.log1p(
            poi["review_count"]
        ),
        "poi_popularity": poi["popularity"],

        # POI characteristics
        "tourist_level": poi["tourist_level"],
        "expected_duration": poi[
            "expected_duration"
        ],

        # Traveler context
        "trip_duration": traveler[
            "trip_duration"
        ],

        "geographic_distance": geographic_distance(
            traveler,
            poi,
        ),
    }


def build_feature_dataset(
    travelers,
    pois,
):
    """
    Generate one row for every traveler–POI pair
    in the same destination.
    """

    rows = []

    for _, traveler in travelers.iterrows():

        destination_pois = pois[
            pois["destination"]
            == traveler["destination"]
        ]

        for _, poi in destination_pois.iterrows():

            features = build_pair_features(
                traveler,
                poi,
            )

            rows.append(features)

    return pd.DataFrame(rows)


if __name__ == "__main__":

    travelers = pd.read_csv(
        DATA_DIR / "travelers.csv"
    )

    pois = pd.read_csv(
        DATA_DIR / "pois.csv"
    )

    features = build_feature_dataset(
        travelers,
        pois,
    )

    output_path = (
        DATA_DIR / "traveler_poi_features.csv"
    )

    features.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Generated {len(features)} traveler-POI pairs"
    )

    print(
        f"Saved to: {output_path}"
    )

    print("\nFeature columns:")
    print(features.columns.tolist())

    print("\nFeature statistics:")
    print(features.describe())