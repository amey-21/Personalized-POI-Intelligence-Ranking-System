# Technical Documentation

## 1. Problem Formulation

The system ranks Points of Interest (POIs) for a traveler based on destination, trip context, explicit preferences, and historical interactions.

This is formulated as a personalized ranking problem. The output is a ranked list of POIs with relevance, contextual compatibility, and confidence signals for downstream itinerary planning.

The system does not generate itineraries, routes, bookings, or travel plans.

## 2. Architecture

```text
Traveler Context + POI Catalog + Historical Interactions
                         ↓
                  Feature Engineering
                         ↓
                  Candidate Generation
                         ↓
              Personalized XGBoost Ranker
                         ↓
                 Context / Constraint Scoring
                         ↓
                  Diversity Reranking
                         ↓
              Ranked POIs + Explanations
```

## 3. Data

The project uses synthetic data:

* 500 POIs across Pune, Mumbai, Goa, Bengaluru, and Delhi
* 100 travelers
* 8,659 chronological interactions

POIs contain category, location, price, rating, popularity, duration, accessibility, and tourist-level information.

Traveler data contains destination, trip duration, interests, budget, party type, mobility, and explicit preferences.

Interactions include view, click, save, share, navigate, visit, booking, and dismiss.

Interaction strength increases from weak signals such as views to strong signals such as visits and bookings, while dismiss represents negative feedback.

## 4. Feature Engineering

The ranking model uses 19 features covering:

* Interest and preference matching
* Budget, mobility, and party compatibility
* POI rating, review count, popularity, and tourist level
* Expected duration and trip duration
* Geographic distance
* Traveler interaction statistics
* Historical category affinity

Explicit preferences and historical behavior are treated as separate signals.

## 5. Candidate Generation

Candidate generation reduces the destination POI catalog before ranking.

Candidates are retrieved using multiple signals:

* Interest matching
* Geographic proximity
* Historical category affinity
* Popularity
* Long-tail POIs

The candidate sources are combined to avoid relying only on popular POIs and to improve local/less-popular discovery.

## 6. Ranking Model

An XGBoost learning-to-rank model is used with the `rank:ndcg` objective.

XGBoost was selected because it handles heterogeneous tabular features effectively while providing a relatively simple and interpretable ranking solution without unnecessary neural-model complexity.

Training uses traveler-level groups and a chronological train/test split.

The interaction relevance signals are converted into non-negative relevance grades for the ranking objective.

## 7. Context and Final Scoring

Preference relevance and practical compatibility are scored separately.

Context compatibility considers:

* Budget
* Mobility
* Party type
* Geographic distance

The final utility combines:

```text
Learned preference
+ Explicit preference
+ Context compatibility
```

The relative contribution of learned and explicit preference changes according to the amount of historical interaction data available.

A diversity-aware reranking step then penalizes repeated POI categories.

The final output contains signals such as:

```text
poi_id
ranking_score
preference_score
context_compatibility
final_utility
diversity_adjusted_score
confidence
```

## 8. Cold Start

Travelers with insufficient interaction history use a separate content/context-based scoring strategy.

It uses:

* Traveler interests
* Explicit preferences
* Budget
* Geographic distance
* POI quality

Historical behavioral features are not used for cold-start travelers.

This allows recommendations to be generated even when there is little or no interaction history.

## 9. Evaluation

The personalized ranker is evaluated on a temporal held-out interaction set against a popularity baseline.

| Metric                   | Personalized | Popularity |
| ------------------------ | -----------: | ---------: |
| Precision@10             |       0.1408 |     0.0724 |
| Recall@10                |       0.2191 |     0.1083 |
| NDCG@10                  |       0.2497 |     0.0767 |
| Catalog Coverage         |       0.6960 |     0.1000 |
| Long-tail Rate           |       0.4090 |     0.0000 |
| Constraint Compatibility |       0.8856 |     0.7762 |
| Category Diversity       |       0.2630 |     0.5640 |

The personalized model improves ranking metrics, coverage, long-tail discovery, and constraint compatibility compared with popularity ranking.

The lower category-diversity score reflects greater concentration on categories relevant to the traveler.

## 10. Example Scenarios

The implementation includes multiple traveler scenarios covering different interests, preferences, budgets, and travel contexts.

Each recommendation provides:

* POI name and category
* Ranking/final score
* Preference relevance
* Context compatibility
* Confidence
* Brief explanation

Example scenarios can be reproduced using:

```bash
python src/run_scenarios.py
```

## 11. Limitations

The interaction dataset is synthetic, so evaluation results should not be interpreted as real-world recommendation performance.

Unobserved traveler–POI pairs are treated as zero relevance during training. In a production system, non-interaction should generally be treated as unknown rather than automatically negative.

## 12. Production Considerations

A production implementation would require:

* Online feature updates
* Periodic model retraining
* Fresh interaction signals
* Recommendation monitoring
* Feedback loops
* Scalable candidate retrieval and model serving
* Monitoring for ranking quality, coverage, and constraint violations
