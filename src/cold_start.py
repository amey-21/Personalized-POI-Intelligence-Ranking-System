import pandas as pd

from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two coordinates in kilometers.
    """

    R = 6371.0

    lat1, lon1, lat2, lon2 = map(
        radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    return 2 * R * atan2(
        sqrt(a),
        sqrt(1 - a)
    )


def is_cold_start_traveler(traveler_id, threshold=5):
    behavior = pd.read_csv(
        "data/traveler_behavior_stats.csv"
    )

    row = behavior[
        behavior["traveler_id"] == traveler_id
    ]

    if row.empty:
        return True

    return row.iloc[0]["total_interactions"] < threshold


def get_traveler_history_strength(traveler_id):
    behavior = pd.read_csv(
        "data/traveler_behavior_stats.csv"
    )

    row = behavior[
        behavior["traveler_id"] == traveler_id
    ]

    if row.empty:
        return 0.0

    interactions = row.iloc[0]["total_interactions"]

    return min(interactions / 50.0, 1.0)


def cold_start_score(traveler, pois):
    """
    Content/context-based score for a traveler with
    insufficient historical interaction data.

    Does not use behavioral features.
    """

    destination_pois = pois[
        pois["destination"] == traveler["destination"]
    ].copy()

    destination_pois["geographic_distance"] = (
        destination_pois.apply(
            lambda row: haversine_distance(
                traveler["start_latitude"],
                traveler["start_longitude"],
                row["latitude"],
                row["longitude"]
            ),
            axis=1
        )
    )

    def distance_score(row):
        distance = row["geographic_distance"]

        if distance <= 5:
            return 1.0
        if distance <= 10:
            return 0.85
        if distance <= 20:
            return 0.65
        if distance <= 40:
            return 0.40

        return 0.20

    interests = {
        x.strip().lower()
        for x in str(traveler["interests"]).split("|")
    }

    preference = str(
        traveler["explicit_preferences"]
    ).lower()

    def interest_score(row):
        category = str(row["category"]).lower()
        subcategory = str(row["subcategory"]).lower()
        tags = str(row["tags"]).lower()

        score = 0.0

        if category in interests:
            score += 0.6

        if subcategory in interests:
            score += 0.2

        for interest in interests:
            if interest in tags:
                score += 0.2
                break

        return min(score, 1.0)

    def preference_score(row):
        score = 0.0

        if "local" in preference:
            if row["tourist_level"] < 0.5:
                score += 1.0

        elif "less touristy" in preference:
            score = 1.0 - row["tourist_level"]

        elif "famous" in preference:
            score = row["popularity"]

        elif "family" in preference:
            score = float(row["family_friendly"])

        elif "outdoor" in preference:
            if row["category"] in [
                "nature",
                "activity"
            ]:
                score = 1.0

        elif "cultural" in preference:
            if row["category"] in [
                "historical",
                "museum",
                "neighborhood"
            ]:
                score = 1.0

        elif "budget" in preference:
            score = 1.0 - (
                row["price_level"] / 4.0
            )

        elif "highly rated" in preference:
            score = row["rating"] / 5.0

        return min(max(score, 0.0), 1.0)

    def budget_score(row):
        budget = traveler["budget"]
        price = row["price_level"]

        if budget == "low":
            if price <= 1:
                return 1.0
            if price == 2:
                return 0.8
            if price == 3:
                return 0.4
            return 0.1

        if budget == "medium":
            if price <= 2:
                return 1.0
            if price == 3:
                return 0.8
            return 0.4

        return 1.0

    destination_pois["distance_score"] = (
        destination_pois.apply(
            distance_score,
            axis=1
        )
    )

    destination_pois["interest_score"] = (
        destination_pois.apply(
            interest_score,
            axis=1
        )
    )

    destination_pois["explicit_preference_score"] = (
        destination_pois.apply(
            preference_score,
            axis=1
        )
    )

    destination_pois["budget_score"] = (
        destination_pois.apply(
            budget_score,
            axis=1
        )
    )

    destination_pois["quality_score"] = (
        0.7 * destination_pois["rating"] / 5.0
        + 0.3 * destination_pois["popularity"]
    )

    # Cold-start score intentionally avoids historical behavior.
    destination_pois["cold_start_score"] = (
        0.35 * destination_pois["interest_score"]
        + 0.25 * destination_pois["explicit_preference_score"]
        + 0.15 * destination_pois["budget_score"]
        + 0.15 * destination_pois["distance_score"]
        + 0.10 * destination_pois["quality_score"]
    )

    return destination_pois.sort_values(
        "cold_start_score",
        ascending=False
    ).reset_index(drop=True)