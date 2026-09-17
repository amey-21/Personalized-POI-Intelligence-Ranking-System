# Source Code

This folder contains the complete data-generation, feature-engineering, candidate-generation, ranking, scoring, evaluation, and recommendation pipeline.

* `generate_data.py` - Generates the synthetic POI catalog.
* `generate_travelers.py` - Generates synthetic traveler profiles.
* `generate_interactions.py` - Generates chronological traveler-POI interactions.
* `feature_engineering.py` - Builds traveler-POI features for ranking.
* `create_training_data.py` - Creates labeled training data and temporal splits.
* `behavior_features.py` - Extracts historical traveler behavior features.
* `merge_behavior_features.py` - Combines behavioral features with ranking features.
* `candidate_generation.py` - Retrieves a diverse candidate set using multiple signals.
* `train_ranker.py` - Trains the XGBoost learning-to-rank model.
* `rank_recommendations.py` - Generates personalized POI rankings.
* `cold_start.py` - Handles travelers with insufficient historical interaction data.
* `context_scoring.py` - Calculates contextual compatibility and final recommendation utility.
* `diversity_reranking.py` - Reduces excessive repetition of POI categories.
* `explain_recommendations.py` - Generates human-readable recommendation reasons.
* `baseline.py` - Provides the popularity-based baseline.
* `evaluate.py` - Evaluates Precision@K, Recall@K, and NDCG@K.
* `evaluate_extended.py` - Evaluates coverage, long-tail discovery, diversity, and constraint compatibility.
* `run_scenarios.py` - Runs example traveler scenarios and displays recommendations.
* `split_data.py` - Splits interaction data for model development.
