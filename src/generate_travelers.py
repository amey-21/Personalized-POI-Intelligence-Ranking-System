import random
from pathlib import Path

import pandas as pd


random.seed(42)


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


DESTINATIONS = [
    "Pune",
    "Mumbai",
    "Goa",
    "Bengaluru",
    "Delhi",
]


INTERESTS = [
    "food",
    "history",
    "architecture",
    "museums",
    "nature",
    "shopping",
    "activities",
    "neighborhoods",
    "entertainment",
]


BUDGETS = [
    "low",
    "medium",
    "high",
]


PARTY_TYPES = [
    "solo",
    "couple",
    "family",
    "friends",
]


MOBILITY_TYPES = [
    "walking",
    "public_transport",
    "car",
]


PREFERENCE_TEMPLATES = [
    "prefer less touristy experiences",
    "prefer famous landmarks",
    "prefer local experiences",
    "prefer family friendly places",
    "prefer outdoor activities",
    "prefer cultural experiences",
    "prefer budget friendly places",
    "prefer highly rated places",
]

def generate_location(destination):
    """
    Generate a synthetic starting location within
    the geographic bounds of the destination.
    """

    bounds = {
        "Pune": {
            "lat_range": (18.45, 18.65),
            "lon_range": (73.75, 73.95),
        },
        "Mumbai": {
            "lat_range": (18.90, 19.25),
            "lon_range": (72.75, 73.10),
        },
        "Goa": {
            "lat_range": (15.25, 15.65),
            "lon_range": (73.70, 74.20),
        },
        "Bengaluru": {
            "lat_range": (12.85, 13.15),
            "lon_range": (77.45, 77.75),
        },
        "Delhi": {
            "lat_range": (28.45, 28.75),
            "lon_range": (76.85, 77.35),
        },
    }

    destination_bounds = bounds[destination]

    latitude = random.uniform(
        *destination_bounds["lat_range"]
    )

    longitude = random.uniform(
        *destination_bounds["lon_range"]
    )

    return round(latitude, 6), round(longitude, 6)

def generate_travelers(n_travelers=100):
    rows = []

    for i in range(n_travelers):
        traveler_id = f"U{i + 1:04d}"

        destination = random.choice(DESTINATIONS)

        start_latitude, start_longitude = generate_location(destination)

        trip_duration = random.choice([2, 3, 4, 5, 7])

        # Each traveler has 2–4 interests.
        interests = random.sample(
            INTERESTS,
            k=random.randint(2, 4),
        )

        budget = random.choice(BUDGETS)

        party_type = random.choice(PARTY_TYPES)

        mobility = random.choice(MOBILITY_TYPES)

        explicit_preference = random.choice(
            PREFERENCE_TEMPLATES
        )

        rows.append(
            {
                "traveler_id": traveler_id,
                "destination": destination,
                "start_latitude": start_latitude,
                "start_longitude": start_longitude,
                "trip_duration": trip_duration,
                "interests": "|".join(interests),
                "budget": budget,
                "party_type": party_type,
                "mobility": mobility,
                "explicit_preferences": explicit_preference,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    travelers = generate_travelers(100)

    output_path = OUTPUT_DIR / "travelers.csv"

    travelers.to_csv(
        output_path,
        index=False,
    )

    print(f"Generated {len(travelers)} travelers")
    print(f"Saved to: {output_path}")

    print("\nDestination distribution:")
    print(travelers["destination"].value_counts())

    print("\nBudget distribution:")
    print(travelers["budget"].value_counts())

    print("\nParty type distribution:")
    print(travelers["party_type"].value_counts())

    print("\nFirst 5 travelers:")
    print(travelers.head())