# Personalized POI Intelligence & Ranking System

An ML pipeline that generates and ranks Points of Interest (POIs) for travelers based on trip context, explicit preferences, and historical interactions.

The system produces ranked POIs for downstream itinerary planning. It does not generate itineraries or handle booking/navigation.

## Pipeline

```text
Traveler + POI Catalog + Interactions
              ↓
       Feature Engineering
              ↓
       Candidate Generation
              ↓
     Personalized Ranking
        (XGBoost Ranker)
              ↓
      Context Scoring
              ↓
       Diversity Reranking
              ↓
   Ranked POIs + Explanations
```

## Key Features

* Synthetic POI, traveler, and interaction datasets
* Content and behavioral feature engineering
* Candidate generation using interests, geography, behavior, popularity, and long-tail POIs
* Personalized learning-to-rank using XGBoost
* Cold-start recommendations using content and context signals
* Budget, mobility, party, and geographic compatibility scoring
* Diversity-aware reranking
* Recommendation explanations
* Offline evaluation against a popularity baseline

## Dataset

The project uses synthetic data:

* 500 POIs across Pune, Mumbai, Goa, Bengaluru, and Delhi
* 100 travelers
* 8,659 chronological interactions
* Interaction signals include view, click, save, share, navigate, visit, booking, and dismiss

Interaction strengths range from negative feedback (`dismiss`) to strong positive signals (`booking`).

## Ranking

The personalized ranker uses XGBoost's learning-to-rank objective.

Features include:

* Interest and preference matching
* Budget, mobility, and party compatibility
* POI rating, popularity, and review count
* Geographic distance
* Traveler behavioral statistics
* Category affinity

For travelers without sufficient history, the system falls back to a content/context-based cold-start model.

## Context & Diversity

Preference relevance and practical compatibility are scored separately.

Context considers:

* Budget
* Mobility
* Party type
* Geographic distance

The final utility combines learned preference, explicit preference, and contextual compatibility. A diversity-aware reranking step reduces excessive repetition of the same POI category.

## Evaluation

Evaluated on a temporal held-out interaction set.

| Metric       | Personalized | Popularity |
| ------------ | -----------: | ---------: |
| Precision@10 |       0.1408 |     0.0724 |
| Recall@10    |       0.2191 |     0.1083 |
| NDCG@10      |       0.2497 |     0.0767 |

Additional evaluation:

| Metric                   | Personalized | Popularity |
| ------------------------ | -----------: | ---------: |
| Catalog Coverage         |       0.6960 |     0.1000 |
| Long-tail Rate           |       0.4090 |     0.0000 |
| Constraint Compatibility |       0.8856 |     0.7762 |
| Category Diversity       |       0.2630 |     0.5640 |

The experiments use synthetic interactions, so the results should be interpreted as evaluation of the implemented methodology rather than real-world recommendation performance.

## Project Structure

```text
poi-intelligence/
├── data/
├── notebooks/
├── src/
├── tests/
├── models/
├── README.md
└── requirements.txt
```

## Setup

```bash
git clone <repository-url>
cd poi-intelligence

pip install -r requirements.txt
```

Generate the data:

```bash
python src/generate_data.py
python src/generate_travelers.py
python src/generate_interactions.py
```

Build features and training data:

```bash
python src/feature_engineering.py
python src/create_training_data.py
python src/behavior_features.py
python src/merge_behavior_features.py
```

Train the ranker:

```bash
python src/train_ranker.py
```

Run scenarios:

```bash
python src/run_scenarios.py
```

Run evaluation:

```bash
python src/evaluate.py
python src/evaluate_extended.py
```

## Limitations

* Interaction data is synthetic.
* Unobserved traveler–POI pairs are treated as zero relevance during training.
* Candidate generation and ranking are currently offline.
* Production deployment would require online feature updates, monitoring, retraining, and a feedback loop.


