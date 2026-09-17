# Data

This folder contains the synthetic POI, traveler, interaction, feature, and train/test datasets generated and used by the ranking pipeline.

* `pois.csv` - Synthetic POI catalog with location, category, rating, popularity, price, and contextual attributes.
* `travelers.csv` - Synthetic traveler profiles containing destination, interests, budget, party type, mobility, and preferences.
* `interactions.csv` - Chronological synthetic traveler-POI interactions used to model behavioral signals.
* `train_interactions.csv` - Training portion of the temporal interaction data.
* `test_interactions.csv` - Held-out interaction data used for evaluation.
* `training_data.csv` - Traveler-POI training pairs with engineered relevance labels.
* `training_data_with_behavior.csv` - Training pairs enriched with historical behavioral features.
* `traveler_poi_features.csv` - Pairwise traveler-POI features used for ranking.
* `traveler_behavior_stats.csv` - Aggregated behavioral statistics for each traveler.
* `traveler_category_affinity.csv` - Traveler preferences/affinity scores by POI category.
* `feature_importance.csv` - Feature importance produced by the trained ranking model.
