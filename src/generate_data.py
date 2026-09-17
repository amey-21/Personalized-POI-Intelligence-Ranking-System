import random
from pathlib import Path

import numpy as np
import pandas as pd


# Reproducibility
random.seed(42)
np.random.seed(42)


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


DESTINATIONS = {
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


POI_TYPES = {
    "food": {
        "subcategories": ["local_restaurant", "cafe", "street_food", "food_experience"],
        "tags": ["local", "food", "authentic", "neighborhood"],
    },
    "historical": {
        "subcategories": ["fort", "palace", "monument", "heritage_site"],
        "tags": ["history", "heritage", "architecture", "culture"],
    },
    "museum": {
        "subcategories": ["art", "history_museum", "science", "cultural"],
        "tags": ["museum", "culture", "education", "history"],
    },
    "nature": {
        "subcategories": ["park", "waterfall", "beach", "garden"],
        "tags": ["nature", "outdoor", "scenic", "relaxation"],
    },
    "shopping": {
        "subcategories": ["market", "street_market", "mall", "local_crafts"],
        "tags": ["shopping", "local", "market", "souvenirs"],
    },
    "activity": {
        "subcategories": ["adventure", "workshop", "walking_tour", "family_activity"],
        "tags": ["activity", "experience", "fun", "interactive"],
    },
    "neighborhood": {
        "subcategories": ["local_area", "cultural_district", "old_town", "food_district"],
        "tags": ["local", "neighborhood", "culture", "authentic"],
    },
    "entertainment": {
        "subcategories": ["cinema", "live_music", "theater", "nightlife"],
        "tags": ["entertainment", "social", "nightlife", "experience"],
    },
}


NAME_PREFIXES = {
    "food": ["Local", "Heritage", "Traditional", "Street", "Hidden"],
    "historical": ["Royal", "Ancient", "Historic", "Heritage", "Old"],
    "museum": ["City", "National", "Modern", "Cultural", "Art"],
    "nature": ["Green", "Lakeside", "Riverside", "Hill", "Nature"],
    "shopping": ["Central", "Local", "Grand", "Old Town", "Craft"],
    "activity": ["Urban", "Adventure", "Cultural", "Family", "Discovery"],
    "neighborhood": ["Old", "Heritage", "Local", "Cultural", "Artisan"],
    "entertainment": ["City", "Royal", "Grand", "Open Air", "Metro"],
}


def generate_poi_name(category, destination, index):
    prefix = random.choice(NAME_PREFIXES[category])
    subcategory = random.choice(POI_TYPES[category]["subcategories"])

    return f"{prefix} {subcategory.replace('_', ' ').title()} {destination} {index}"


def generate_pois(n_pois=500):
    rows = []

    destinations = list(DESTINATIONS.keys())
    categories = list(POI_TYPES.keys())

    for i in range(n_pois):
        poi_id = f"P{i + 1:04d}"

        destination = random.choice(destinations)
        category = random.choice(categories)

        poi_type = POI_TYPES[category]
        subcategory = random.choice(poi_type["subcategories"])

        lat_min, lat_max = DESTINATIONS[destination]["lat_range"]
        lon_min, lon_max = DESTINATIONS[destination]["lon_range"]

        latitude = round(random.uniform(lat_min, lat_max), 6)
        longitude = round(random.uniform(lon_min, lon_max), 6)

        tags = random.sample(
            poi_type["tags"],
            k=random.randint(2, min(4, len(poi_type["tags"])))
        )

        price_level = random.randint(1, 4)

        rating = round(
            min(5.0, max(2.5, np.random.normal(4.1, 0.45))),
            2
        )

        review_count = int(np.random.lognormal(mean=5.5, sigma=1.2))

        # Popularity is intentionally not perfectly correlated with rating.
        popularity = round(
            min(
                1.0,
                max(
                    0.01,
                    0.45 * (review_count / 1000)
                    + 0.25 * (rating / 5)
                    + random.uniform(0, 0.35)
                ),
            ),
            3,
        )

        expected_duration = random.choice(
            [30, 45, 60, 90, 120, 150, 180]
        )

        opening_hour = random.choice([8, 9, 10, 11])
        closing_hour = random.choice([17, 18, 19, 20, 21, 22])

        reservation_required = random.random() < 0.25
        family_friendly = random.random() < 0.65
        public_transport_access = random.random() < 0.75

        # Lower values represent more local / less tourist-heavy experiences.
        tourist_level = round(random.uniform(0.05, 1.0), 3)

        description = (
            f"A {subcategory.replace('_', ' ')} in {destination} "
            f"focused on {', '.join(tags)}."
        )

        rows.append(
            {
                "poi_id": poi_id,
                "name": generate_poi_name(
                    category,
                    destination,
                    i + 1,
                ),
                "destination": destination,
                "category": category,
                "subcategory": subcategory,
                "description": description,
                "tags": "|".join(tags),
                "latitude": latitude,
                "longitude": longitude,
                "price_level": price_level,
                "rating": rating,
                "review_count": review_count,
                "popularity": popularity,
                "expected_duration": expected_duration,
                "opening_time": f"{opening_hour:02d}:00",
                "closing_time": f"{closing_hour:02d}:00",
                "reservation_required": reservation_required,
                "family_friendly": family_friendly,
                "public_transport_access": public_transport_access,
                "tourist_level": tourist_level,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pois = generate_pois(500)

    output_path = OUTPUT_DIR / "pois.csv"
    pois.to_csv(output_path, index=False)

    print(f"Generated {len(pois)} POIs")
    print(f"Saved to: {output_path}")
    print("\nCategory distribution:")
    print(pois["category"].value_counts())

    print("\nDestination distribution:")
    print(pois["destination"].value_counts())

    print("\nFirst 5 rows:")
    print(pois.head())