"""
Content-Based Filtering Recommender
------------------------------------
Recommends movies similar to a given movie (or a user's liked movies)
based on genre similarity, using TF-IDF + cosine similarity.

Why this approach:
- Works even for brand-new movies with no ratings yet (no "cold start"
  problem for items).
- Easy to explain: "recommended because it shares genres with X".
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, movies_path: str):
        self.movies = pd.read_csv(movies_path)
        # Genres are pipe-separated e.g. "Action|Sci-Fi" -> turn into
        # space-separated tokens so TF-IDF treats each genre as a "word".
        self.movies["genres_clean"] = (
            self.movies["genres"].fillna("").str.replace("|", " ", regex=False)
        )

        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies["genres_clean"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        self.title_to_index = {
            title: idx for idx, title in enumerate(self.movies["title"])
        }

    def recommend_similar(self, title: str, top_n: int = 5) -> pd.DataFrame:
        """Recommend movies similar to a given movie title."""
        if title not in self.title_to_index:
            raise ValueError(f"Movie '{title}' not found in dataset.")

        idx = self.title_to_index[title]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # Skip index 0 result since it will always be the movie itself
        scores = [s for s in scores if s[0] != idx][:top_n]

        result_indices = [s[0] for s in scores]
        result_scores = [round(s[1], 3) for s in scores]

        result = self.movies.iloc[result_indices][["movieId", "title", "genres"]].copy()
        result["similarity_score"] = result_scores
        return result.reset_index(drop=True)

    def recommend_for_profile(self, liked_titles: list, top_n: int = 5) -> pd.DataFrame:
        """
        Recommend movies for a user based on a list of movies they liked,
        by averaging the TF-IDF vectors of their liked movies (a simple
        'user profile') and finding the closest unwatched movies.
        """
        liked_indices = [
            self.title_to_index[t] for t in liked_titles if t in self.title_to_index
        ]
        if not liked_indices:
            raise ValueError("None of the liked titles were found in dataset.")

        import numpy as np
        profile_vector = np.asarray(self.tfidf_matrix[liked_indices].mean(axis=0))
        sims = cosine_similarity(profile_vector, self.tfidf_matrix)[0]

        scored = [
            (i, sims[i]) for i in range(len(self.movies)) if i not in liked_indices
        ]
        scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]

        result_indices = [s[0] for s in scored]
        result_scores = [round(s[1], 3) for s in scored]

        result = self.movies.iloc[result_indices][["movieId", "title", "genres"]].copy()
        result["similarity_score"] = result_scores
        return result.reset_index(drop=True)


if __name__ == "__main__":
    rec = ContentBasedRecommender("data/movies.csv")

    print("Movies similar to 'The Dark Knight':")
    print(rec.recommend_similar("The Dark Knight", top_n=5))

    print("\nRecommendations based on liking Inception + The Matrix:")
    print(rec.recommend_for_profile(["Inception", "The Matrix"], top_n=5))
