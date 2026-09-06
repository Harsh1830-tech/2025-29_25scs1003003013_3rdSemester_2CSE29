"""
Command-line interface for the Movie Recommendation System.

Usage examples:
    python src/cli.py similar --title "The Dark Knight"
    python src/cli.py profile --titles "Inception" "The Matrix"
    python src/cli.py user --user_id 1
    python src/cli.py hybrid --user_id 1
"""

import argparse

from content_based import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender
from hybrid import HybridRecommender

MOVIES_PATH = "data/movies.csv"
RATINGS_PATH = "data/ratings.csv"


def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_similar = subparsers.add_parser("similar", help="Content-based: movies similar to a title")
    p_similar.add_argument("--title", required=True)
    p_similar.add_argument("--top_n", type=int, default=5)

    p_profile = subparsers.add_parser("profile", help="Content-based: recommend from liked titles")
    p_profile.add_argument("--titles", nargs="+", required=True)
    p_profile.add_argument("--top_n", type=int, default=5)

    p_user = subparsers.add_parser("user", help="Collaborative filtering: recommend for a user id")
    p_user.add_argument("--user_id", type=int, required=True)
    p_user.add_argument("--top_n", type=int, default=5)

    p_hybrid = subparsers.add_parser("hybrid", help="Hybrid recommendation for a user id")
    p_hybrid.add_argument("--user_id", type=int, required=True)
    p_hybrid.add_argument("--top_n", type=int, default=5)

    args = parser.parse_args()

    if args.command == "similar":
        rec = ContentBasedRecommender(MOVIES_PATH)
        print(rec.recommend_similar(args.title, top_n=args.top_n).to_string(index=False))

    elif args.command == "profile":
        rec = ContentBasedRecommender(MOVIES_PATH)
        print(rec.recommend_for_profile(args.titles, top_n=args.top_n).to_string(index=False))

    elif args.command == "user":
        rec = CollaborativeFilteringRecommender(RATINGS_PATH, MOVIES_PATH)
        print(rec.recommend_for_user(args.user_id, top_n=args.top_n).to_string(index=False))

    elif args.command == "hybrid":
        rec = HybridRecommender(RATINGS_PATH, MOVIES_PATH)
        print(rec.recommend_for_user(args.user_id, top_n=args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
