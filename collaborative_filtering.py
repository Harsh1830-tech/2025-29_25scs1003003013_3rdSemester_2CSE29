"""
Collaborative Filtering Recommender
-------------------------------------
Recommends movies based on the ratings patterns of *similar users*
(user-based collaborative filtering), using cosine similarity over
a user-item ratings matrix.

Why this approach:
- Captures taste patterns that content/genre metadata can't see
  (e.g. two movies with different genres but the same audience).
- No need to hand-craft item features.
- Classic technique behind early Netflix/Amazon-style recommenders.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilteringRecommender:
    def __init__(self, ratings_path: str, movies_path: str):
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)

        # Build user-item matrix: rows = users, columns = movies, values = ratings
        self.user_item_matrix = self.ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        ).fillna(0)

        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

    def _movie_title(self, movie_id: int) -> str:
        row = self.movies.loc[self.movies["movieId"] == movie_id, "title"]
        return row.values[0] if len(row) else f"Movie {movie_id}"

    def recommend_for_user(self, user_id: int, top_n: int = 5, k_neighbors: int = 5) -> pd.DataFrame:
        """
        Recommend movies for a user by:
        1. Finding the k most similar users ("neighbors").
        2. Computing a weighted average of their ratings for movies
           the target user hasn't rated yet.
        3. Returning the top-N highest predicted-rating movies.
        """
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found in dataset.")

        # Most similar users, excluding the user themselves
        similar_users = (
            self.user_similarity_df[user_id]
            .drop(index=user_id)
            .sort_values(ascending=False)
            .head(k_neighbors)
        )

        neighbor_ratings = self.user_item_matrix.loc[similar_users.index]
        weights = similar_users.values

        # Weighted average rating per movie across neighbors
        weighted_sum = neighbor_ratings.T.dot(weights)
        weight_totals = np.array(
            [weights[neighbor_ratings[col] > 0].sum() if (neighbor_ratings[col] > 0).any() else 0
             for col in neighbor_ratings.columns]
        )
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            predicted_scores = np.where(weight_totals > 0, weighted_sum / weight_totals, 0)

        predictions = pd.Series(predicted_scores, index=neighbor_ratings.columns)

        # Exclude movies the target user already rated
        already_rated = self.user_item_matrix.loc[user_id]
        already_rated_ids = already_rated[already_rated > 0].index
        predictions = predictions.drop(index=already_rated_ids, errors="ignore")

        top_predictions = predictions.sort_values(ascending=False).head(top_n)

        result = pd.DataFrame({
            "movieId": top_predictions.index,
            "title": [self._movie_title(mid) for mid in top_predictions.index],
            "predicted_rating": top_predictions.values.round(2),
        })
        return result.reset_index(drop=True)

    def similar_users(self, user_id: int, top_n: int = 5) -> pd.Series:
        """Return the top-N users most similar in taste to the given user."""
        return (
            self.user_similarity_df[user_id]
            .drop(index=user_id)
            .sort_values(ascending=False)
            .head(top_n)
            .round(3)
        )


if __name__ == "__main__":
    rec = CollaborativeFilteringRecommender("data/ratings.csv", "data/movies.csv")

    user_id = 1
    print(f"Users most similar to User {user_id}:")
    print(rec.similar_users(user_id))

    print(f"\nRecommendations for User {user_id}:")
    print(rec.recommend_for_user(user_id, top_n=5))
