# Authors: Siddarth Makkuni skm179, Dhruv Patel dp1379, Melania Labadze ML1854

"""
Automated tests for movie_recommender.py

Run with:
    python test_movie_recommender.py

Each test prints PASS or FAIL with details.
"""

import sys
import os

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from movie_recommender import (
    load_movies,
    load_ratings,
    top_n_movies,
    top_n_movies_in_genre,
    top_n_genres,
    user_top_genre,
    recommend_movies,
)

# ---------------------------------------------------------------------------
# Minimal inline dataset so the tests are self-contained.
# We also support loading from files if they exist next to this script.
# ---------------------------------------------------------------------------

MOVIES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "movies.txt")
RATINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ratings.txt")

# Inline dataset (used when files are absent)
INLINE_MOVIES = [
    {"genre": "Adventure", "movie_id": "1",  "movie_name": "Toy Story (1995)"},
    {"genre": "Adventure", "movie_id": "2",  "movie_name": "Jumanji (1995)"},
    {"genre": "Adventure", "movie_id": "8",  "movie_name": "Tom and Huck (1995)"},
    {"genre": "Comedy",    "movie_id": "3",  "movie_name": "Grumpier Old Men (1995)"},
    {"genre": "Comedy",    "movie_id": "4",  "movie_name": "Waiting to Exhale (1995)"},
    {"genre": "Comedy",    "movie_id": "5",  "movie_name": "Father of the Bride Part II (1995)"},
    {"genre": "Action",    "movie_id": "6",  "movie_name": "Heat (1995)"},
    {"genre": "Action",    "movie_id": "9",  "movie_name": "Sudden Death (1995)"},
    {"genre": "Action",    "movie_id": "10", "movie_name": "GoldenEye (1995)"},
    {"genre": "Drama",     "movie_id": "11", "movie_name": "Richard III (1995)"},
    {"genre": "Drama",     "movie_id": "12", "movie_name": "Dead Man (1995)"},
    {"genre": "Thriller",  "movie_id": "13", "movie_name": "Copycat (1995)"},
    {"genre": "Thriller",  "movie_id": "14", "movie_name": "Shanghai Triad (1995)"},
]

INLINE_RATINGS = [
    # Adventure
    {"movie_name": "Toy Story (1995)",    "rating": 4.5, "user_id": "user1"},
    {"movie_name": "Toy Story (1995)",    "rating": 5.0, "user_id": "user2"},
    {"movie_name": "Toy Story (1995)",    "rating": 3.5, "user_id": "user3"},
    {"movie_name": "Jumanji (1995)",      "rating": 3.0, "user_id": "user1"},
    {"movie_name": "Jumanji (1995)",      "rating": 2.5, "user_id": "user2"},
    {"movie_name": "Tom and Huck (1995)", "rating": 4.0, "user_id": "user3"},
    # Comedy
    {"movie_name": "Grumpier Old Men (1995)",             "rating": 2.0, "user_id": "user1"},
    {"movie_name": "Grumpier Old Men (1995)",             "rating": 3.0, "user_id": "user2"},
    {"movie_name": "Waiting to Exhale (1995)",            "rating": 4.5, "user_id": "user1"},
    {"movie_name": "Waiting to Exhale (1995)",            "rating": 5.0, "user_id": "user3"},
    {"movie_name": "Father of the Bride Part II (1995)",  "rating": 3.5, "user_id": "user2"},
    # Action
    {"movie_name": "Heat (1995)",         "rating": 4.0, "user_id": "user1"},
    {"movie_name": "Heat (1995)",         "rating": 4.5, "user_id": "user2"},
    {"movie_name": "Sudden Death (1995)", "rating": 2.5, "user_id": "user3"},
    {"movie_name": "GoldenEye (1995)",    "rating": 3.5, "user_id": "user1"},
    {"movie_name": "GoldenEye (1995)",    "rating": 4.0, "user_id": "user3"},
    # Drama
    {"movie_name": "Richard III (1995)", "rating": 4.5, "user_id": "user2"},
    {"movie_name": "Richard III (1995)", "rating": 5.0, "user_id": "user1"},
    {"movie_name": "Dead Man (1995)",    "rating": 3.0, "user_id": "user3"},
    # Thriller
    {"movie_name": "Copycat (1995)",        "rating": 3.5, "user_id": "user2"},
    {"movie_name": "Shanghai Triad (1995)", "rating": 4.0, "user_id": "user1"},
    {"movie_name": "Shanghai Triad (1995)", "rating": 4.5, "user_id": "user3"},
]


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0


def check(test_name: str, got, expected, exact: bool = True) -> None:
    """Compare got vs expected and print PASS/FAIL."""
    global PASS, FAIL
    ok = (got == expected) if exact else (str(got) == str(expected))
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
        print(f"  [{status}] {test_name}")
    else:
        FAIL += 1
        print(f"  [{status}] {test_name}")
        print(f"         Expected: {expected}")
        print(f"         Got:      {got}")


def approx_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


def check_float(test_name: str, got: float, expected: float, tol: float = 1e-9) -> None:
    global PASS, FAIL
    ok = approx_equal(got, expected, tol)
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
        print(f"  [{status}] {test_name}")
    else:
        FAIL += 1
        print(f"  [{status}] {test_name}")
        print(f"         Expected: {expected}")
        print(f"         Got:      {got}")


# ---------------------------------------------------------------------------
# Load data (from files if available, else inline)
# ---------------------------------------------------------------------------

def get_data():
    if os.path.exists(MOVIES_FILE) and os.path.exists(RATINGS_FILE):
        movies = load_movies(MOVIES_FILE)
        ratings = load_ratings(RATINGS_FILE)
        print(f"  Using files: {MOVIES_FILE}, {RATINGS_FILE}")
    else:
        movies = INLINE_MOVIES
        ratings = INLINE_RATINGS
        print("  Using inline data (files not found)")
    return movies, ratings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_movies():
    print("\n--- test_load_movies ---")
    if not os.path.exists(MOVIES_FILE):
        print("  (skipped – movies.txt not found)")
        return
    movies = load_movies(MOVIES_FILE)
    check("loads non-empty list", len(movies) > 0, True)
    check("first record has genre key", "genre" in movies[0], True)
    check("first record has movie_id key", "movie_id" in movies[0], True)
    check("first record has movie_name key", "movie_name" in movies[0], True)


def test_load_ratings():
    print("\n--- test_load_ratings ---")
    if not os.path.exists(RATINGS_FILE):
        print("  (skipped – ratings.txt not found)")
        return
    ratings = load_ratings(RATINGS_FILE)
    check("loads non-empty list", len(ratings) > 0, True)
    check("rating is float", isinstance(ratings[0]["rating"], float), True)


def test_load_bad_file():
    print("\n--- test_load_bad_file ---")
    result = load_movies("nonexistent_file_xyz.txt")
    check("returns empty list for missing file", result, [])


def test_top_n_movies(movies, ratings):
    print("\n--- test_top_n_movies ---")
    results = top_n_movies(movies, ratings, 3)
    check("returns 3 results", len(results), 3)
    # Scores should be non-increasing
    scores = [s for _, s in results]
    check("scores descending", scores, sorted(scores, reverse=True))
    # Top movie by inspection: Richard III avg=(4.5+5.0)/2=4.75
    check("top movie is Richard III (1995)", results[0][0], "Richard III (1995)")
    check_float("top movie avg rating", results[0][1], 4.75)

    # N larger than dataset → return all rated movies
    all_results = top_n_movies(movies, ratings, 9999)
    check("n > total returns all rated movies", len(all_results) <= len(movies), True)

    # N = 0 → empty
    check("n=0 returns empty", top_n_movies(movies, ratings, 0), [])


def test_top_n_movies_in_genre(movies, ratings):
    print("\n--- test_top_n_movies_in_genre ---")
    results = top_n_movies_in_genre(movies, ratings, "Adventure", 2)
    check("returns 2 results for Adventure", len(results), 2)
    names = [n for n, _ in results]
    # Toy Story avg=4.333..., Tom and Huck avg=4.0, Jumanji avg=2.75
    check("top Adventure movie is Toy Story (1995)", names[0], "Toy Story (1995)")

    # Case-insensitive genre matching
    results_lower = top_n_movies_in_genre(movies, ratings, "adventure", 2)
    check("genre match is case-insensitive", results_lower, results)

    # Unknown genre → empty
    check("unknown genre returns empty", top_n_movies_in_genre(movies, ratings, "Sci-Fi", 5), [])


def test_top_n_genres(movies, ratings):
    print("\n--- test_top_n_genres ---")
    results = top_n_genres(movies, ratings, 3)
    check("returns 3 genres", len(results), 3)
    genre_scores = {g: s for g, s in results}
    # Drama: only Richard III (4.75) and Dead Man (3.0) → avg of avgs = (4.75+3.0)/2 = 3.875
    check("Drama in top 3", "Drama" in genre_scores, True)

    # N=0 → empty
    check("n=0 returns empty", top_n_genres(movies, ratings, 0), [])


def test_user_top_genre(movies, ratings):
    print("\n--- test_user_top_genre ---")
    # user1 Comedy ratings: Grumpier Old Men=2.0, Waiting to Exhale=4.5 → avg 3.25
    # user1 Adventure ratings: Toy Story=4.5, Jumanji=3.0 → avg 3.75
    # user1 Action ratings: Heat=4.0, GoldenEye=3.5 → avg 3.75
    # user1 Drama ratings: Richard III=5.0 → avg 5.0
    # user1 Thriller ratings: Shanghai Triad=4.0 → avg 4.0
    # Expected top genre: Drama
    result = user_top_genre(movies, ratings, "user1")
    check("user1 top genre is Drama", result, "Drama")

    # Non-existent user → None
    check("unknown user returns None", user_top_genre(movies, ratings, "ghost"), None)


def test_recommend_movies(movies, ratings):
    print("\n--- test_recommend_movies ---")
    # user1's top genre is Drama
    # Drama movies: Richard III (rated by user1), Dead Man (not rated by user1)
    # Only Dead Man is unrated by user1
    results = recommend_movies(movies, ratings, "user1")
    names = [n for n, _ in results]
    check("returns at most 3", len(results) <= 3, True)
    check("Dead Man (1995) recommended for user1", "Dead Man (1995)" in names, True)
    check("Richard III not recommended (already rated)", "Richard III (1995)" not in names, True)

    # Unknown user → empty
    check("unknown user returns empty", recommend_movies(movies, ratings, "ghost"), [])


def test_edge_cases(movies, ratings):
    print("\n--- test_edge_cases ---")
    # Empty movies list
    check("top_n_movies empty movies", top_n_movies([], ratings, 5), [])
    # Empty ratings list
    check("top_n_movies empty ratings", top_n_movies(movies, [], 5), [])
    # Both empty
    check("top_n_genres both empty", top_n_genres([], [], 5), [])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  Movie Recommender – Automated Tests")
    print("=" * 50)

    movies, ratings = get_data()

    test_load_movies()
    test_load_ratings()
    test_load_bad_file()
    test_top_n_movies(movies, ratings)
    test_top_n_movies_in_genre(movies, ratings)
    test_top_n_genres(movies, ratings)
    test_user_top_genre(movies, ratings)
    test_recommend_movies(movies, ratings)
    test_edge_cases(movies, ratings)

    print("\n" + "=" * 50)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("=" * 50)
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
