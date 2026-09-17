import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRanker
import joblib

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_DIR.mkdir(exist_ok=True)


FEATURE_COLUMNS = [
    "interest_match",
    "preference_match",
    "budget_match",
    "mobility_match",
    "party_match",
    "poi_rating",
    "poi_review_count",
    "poi_popularity",
    "tourist_level",
    "expected_duration",
    "trip_duration",
    "geographic_distance",
    "total_interactions",
    "total_behavior_strength",
    "avg_poi_rating",
    "avg_poi_popularity",
    "avg_price_level",
    "avg_tourist_level",
    "category_affinity"
]


def prepare_training_data():

    data = pd.read_csv(
        DATA_DIR / "training_data_with_behavior.csv"
    )

    # XGBoost ranking requires each traveler's rows
    # to be grouped together.
    data = data.sort_values(
        ["traveler_id", "relevance_label"],
        ascending=[True, False]
    ).reset_index(drop=True)

    X = data[FEATURE_COLUMNS].copy()
    y = (data["relevance_label"].mul(10).round().astype(int))

    # Make sure all numeric features are valid.
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Number of POI pairs belonging to each traveler.
    groups = (
        data.groupby("traveler_id", sort=False)
        .size()
        .tolist()
    )

    return X, y, groups, data


def train_model():

    X, y, groups, data = prepare_training_data()

    print("Training ranking model...")
    print(f"Training rows: {len(X)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Travelers/groups: {len(groups)}")

    model = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        n_estimators=250,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42
    )

    model.fit(
        X,
        y,
        group=groups
    )

    model_path = MODEL_DIR / "poi_ranker.json"

    model.save_model(model_path)

    # Feature importance
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        DATA_DIR / "feature_importance.csv",
        index=False
    )

    print("\nModel trained successfully.")
    print(f"Saved model to: {model_path}")

    print("\nTop features:")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    train_model()