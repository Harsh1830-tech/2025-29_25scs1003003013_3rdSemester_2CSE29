# 🎬 Movie Recommendation System

A movie recommendation engine built in Python implementing three classic recommender-system techniques:

- **Content-Based Filtering** — recommends movies with similar genres using TF-IDF + cosine similarity
- **Collaborative Filtering** — recommends movies based on the rating patterns of similar users (user-user CF)
- **Hybrid Model** — blends both approaches to overcome the weaknesses of each on its own

Built as a compact, dependency-light project to demonstrate core recommender-system concepts end to end: data modeling, vectorization, similarity metrics, and a simple CLI.

---

## Why two techniques instead of one?

| Approach | Strength | Weakness |
|---|---|---|
| **Content-Based** | Works for brand-new movies with zero ratings ("cold start" for items). Recommendations are explainable — "because you liked X". | Can get repetitive — only ever recommends within the same genres. |
| **Collaborative Filtering** | Captures real audience taste patterns that genre metadata misses entirely. | Struggles with new users/movies that have no ratings yet. |
| **Hybrid** | Combines both signals into one weighted score. | Slightly more compute; needs tuning of the blend weight. |

---

## Project Structure

```
movie-recommender/
├── data/
│   ├── movies.csv        # 40 movies with titles + pipe-separated genres
│   └── ratings.csv       # Simulated ratings from 25 users
├── src/
│   ├── content_based.py         # TF-IDF + cosine similarity on genres
│   ├── collaborative_filtering.py  # User-user CF on the ratings matrix
│   ├── hybrid.py                # Weighted blend of both models
│   └── cli.py                   # Command-line interface
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Content-Based Filtering (`content_based.py`)
Each movie's genres (e.g. `Action|Sci-Fi|Thriller`) are vectorized with **TF-IDF**, treating each genre as a "word." Cosine similarity between vectors gives a similarity score between any two movies. A user's "taste profile" is simply the average vector of the movies they liked.

### 2. Collaborative Filtering (`collaborative_filtering.py`)
Ratings are arranged into a **user × movie matrix**. Cosine similarity between rows finds the *k* most similar users to a target user. Predicted ratings for unseen movies are a similarity-weighted average of what those neighbors rated.

### 3. Hybrid Model (`hybrid.py`)
For a target user: collaborative filtering surfaces candidates from similar users' behavior, content-based filtering re-scores those (and adds genre-adjacent candidates) using the user's own top-rated movies. Final score:

```
hybrid_score = cf_weight * cf_score + (1 - cf_weight) * cb_score
```

`cf_weight` defaults to `0.6` and is easy to tune in `HybridRecommender.__init__`.

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/movie-recommender.git
cd movie-recommender

# Install dependencies
pip install -r requirements.txt
```

### Run via CLI

```bash
# Movies similar to a given title (content-based)
python src/cli.py similar --title "The Dark Knight" --top_n 5

# Recommend based on a list of movies you liked (content-based)
python src/cli.py profile --titles "Inception" "The Matrix" --top_n 5

# Recommend for a specific user based on similar users (collaborative filtering)
python src/cli.py user --user_id 1 --top_n 5

# Hybrid recommendation for a user
python src/cli.py hybrid --user_id 1 --top_n 5
```

### Or use it as a library

```python
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.hybrid import HybridRecommender

cb = ContentBasedRecommender("data/movies.csv")
print(cb.recommend_similar("Interstellar", top_n=5))

cf = CollaborativeFilteringRecommender("data/ratings.csv", "data/movies.csv")
print(cf.recommend_for_user(user_id=1, top_n=5))

hybrid = HybridRecommender("data/ratings.csv", "data/movies.csv")
print(hybrid.recommend_for_user(user_id=1, top_n=5))
```

---

## Sample Output

```
$ python src/cli.py similar --title "Inception" --top_n 3
 movieId        title                  genres  similarity_score
       7   The Matrix           Action|Sci-Fi             0.849
      13 The Avengers Action|Adventure|Sci-Fi             0.721
      14     Iron Man Action|Adventure|Sci-Fi             0.721
```

---

## Dataset

This repo ships with a small, self-contained sample dataset (40 movies, 25 simulated users, ~370 ratings) so the project runs immediately with zero setup. To scale it up, drop in the [MovieLens dataset](https://grouplens.org/datasets/movielens/) — the code expects the same `movieId, title, genres` / `userId, movieId, rating` schema, so no code changes are needed, just point the paths in `src/cli.py` at the new CSVs.

---

## Possible Extensions

- Matrix factorization (SVD / ALS) for more scalable collaborative filtering
- Item-based (rather than user-based) collaborative filtering
- Evaluation metrics: precision@k, recall@k, RMSE on held-out ratings
- A small Streamlit/Flask front end for interactive demos
- Swap TF-IDF for movie overviews/plot summaries (true NLP-based content filtering)

---

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `NumPy`

---

## License

MIT — free to use, modify, and build on.
