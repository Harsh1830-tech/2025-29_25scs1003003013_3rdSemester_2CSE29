"""
Hybrid Recommender
--------------------
Combines content-based and collaborative filtering:
- Collaborative filtering finds movies liked by similar users.
- Content-based filtering re-ranks/expands those using genre similarity
  to the user's own highly-rated movies.

This mitigates weaknesses of each approach on its own:
- Pure collaborative filtering struggles with new users/items (cold start).
- Pure content-based filtering can be too narrow (keeps suggesting the
  same genre) and ignores what similar users actually enjoyed.
"""

import pandas as pd

from content_based import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender


class HybridRecommender:
    def __init__(self, ratings_path: str, movies_path: str, cf_weight: float = 0.6):
        self.cb = ContentBasedRecommender(movies_path)
        self.cf = CollaborativeFilteringRecommender(ratings_path, movies_path)
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        self.cf_weight = cf_weight  # how much to trust CF vs content-based

    def recommend_for_user(self, user_id: int, top_n: int = 5) -> pd.DataFrame:
        # 1. Get the user's top-rated movies to build a content "taste profile"
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        if user_ratings.empty:
            raise ValueError(f"User {user_id} has no ratings in dataset.")

        top_liked = (
            user_ratings.sort_values("rating", ascending=False)
            .head(3)
            .merge(self.movies, on="movieId")["title"]
            .tolist()
        )

        # 2. Collaborative filtering candidates (from similar users' behavior)
        cf_recs = self.cf.recommend_for_user(user_id, top_n=top_n * 2)
        cf_recs["cf_score"] = cf_recs["predicted_rating"] / 5.0  # normalize to 0-1

        # 3. Content-based candidates (from genre similarity to liked movies)
        cb_recs = self.cb.recommend_for_profile(top_liked, top_n=top_n * 2)
        cb_recs = cb_recs.rename(columns={"similarity_score": "cb_score"})

        # 4. Merge both candidate sets and blend scores
        merged = pd.merge(
            cf_recs[["movieId", "title", "cf_score"]],
            cb_recs[["movieId", "cb_score"]],
            on="movieId",
            how="outer",
        )
        # Fill in missing titles from movies table (for CB-only rows)
        merged = merged.merge(self.movies[["movieId", "title"]], on="movieId", how="left", suffixes=("", "_lookup"))
        merged["title"] = merged["title"].fillna(merged["title_lookup"])
        merged = merged.drop(columns=["title_lookup"])

        merged["cf_score"] = merged["cf_score"].fillna(0)
        merged["cb_score"] = merged["cb_score"].fillna(0)

        merged["hybrid_score"] = (
            self.cf_weight * merged["cf_score"] + (1 - self.cf_weight) * merged["cb_score"]
        )

        result = merged.sort_values("hybrid_score", ascending=False).head(top_n)
        result["hybrid_score"] = result["hybrid_score"].round(3)
        return result[["movieId", "title", "hybrid_score"]].reset_index(drop=True)


if __name__ == "__main__":
    rec = HybridRecommender("data/ratings.csv", "data/movies.csv")
    print("Hybrid recommendations for User 1:")
    print(rec.recommend_for_user(1, top_n=5))
