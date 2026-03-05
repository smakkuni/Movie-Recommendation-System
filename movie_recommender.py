# Authors: Siddarth Makkuni skm179, Dhruv Patel dp1379, Melania Labadze ML1854

"""
Movie Recommender System
Implements movie popularity rankings, genre analysis, and personalized recommendations.
"""

from collections import defaultdict


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_movies(filepath: str) -> list[dict]:
    """
    Load movies from a pipe-delimited text file.

    Each line format: movie_genre|movie_id|movie_name

    Args:
        filepath: Path to the movies file.

    Returns:
        A list of dicts with keys 'genre', 'movie_id', 'movie_name'.
        Returns an empty list if the file cannot be read.
    """
    movies = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) != 3:
                    continue  # skip malformed lines
                genre, movie_id, movie_name = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if not genre or not movie_id or not movie_name:
                    continue
                movies.append({
                    "genre": genre,
                    "movie_id": movie_id,
                    "movie_name": movie_name,
                })
    except FileNotFoundError:
        print(f"[Error] File not found: {filepath}")
    except OSError as e:
        print(f"[Error] Could not read {filepath}: {e}")
    return movies


def load_ratings(filepath: str) -> list[dict]:
    """
    Load ratings from a pipe-delimited text file.

    Each line format: movie_name|rating|user_id

    Args:
        filepath: Path to the ratings file.

    Returns:
        A list of dicts with keys 'movie_name', 'rating' (float), 'user_id'.
        Returns an empty list if the file cannot be read.
    """
    ratings = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) != 3:
                    continue
                movie_name, rating_str, user_id = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if not movie_name or not rating_str or not user_id:
                    continue
                try:
                    rating = float(rating_str)
                except ValueError:
                    continue  # skip lines with non-numeric ratings
                if not (0.0 <= rating <= 5.0):
                    continue  # skip out-of-range ratings
                ratings.append({
                    "movie_name": movie_name,
                    "rating": rating,
                    "user_id": user_id,
                })
    except FileNotFoundError:
        print(f"[Error] File not found: {filepath}")
    except OSError as e:
        print(f"[Error] Could not read {filepath}: {e}")
    return ratings


# ---------------------------------------------------------------------------
# Helper – build lookup structures
# ---------------------------------------------------------------------------

def _build_avg_ratings(ratings: list[dict]) -> dict[str, float]:
    """
    Compute the average rating for every movie that has at least one rating.

    Args:
        ratings: List of rating dicts (from load_ratings).

    Returns:
        Dict mapping movie_name (case-sensitive) → average rating (float).
    """
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for r in ratings:
        totals[r["movie_name"]] += r["rating"]
        counts[r["movie_name"]] += 1
    return {name: totals[name] / counts[name] for name in totals}


def _movie_name_to_genre(movies: list[dict]) -> dict[str, str]:
    """
    Build a mapping from movie_name → genre.

    Args:
        movies: List of movie dicts (from load_movies).

    Returns:
        Dict mapping movie_name → genre string.
    """
    return {m["movie_name"]: m["genre"] for m in movies}


# ---------------------------------------------------------------------------
# Feature 1 – Movie popularity (top n by average rating)
# ---------------------------------------------------------------------------

def top_n_movies(movies: list[dict], ratings: list[dict], n: int) -> list[tuple[str, float]]:
    """
    Return the top-n movies ranked by average rating (descending).

    Movies with no ratings are excluded. Ties are broken alphabetically by
    movie name.

    Args:
        movies:  List of movie dicts (from load_movies).
        ratings: List of rating dicts (from load_ratings).
        n:       Number of movies to return.

    Returns:
        List of (movie_name, avg_rating) tuples, length ≤ n.
    """
    avg = _build_avg_ratings(ratings)
    known_names = {m["movie_name"] for m in movies}
    ranked = sorted(
        [(name, score) for name, score in avg.items() if name in known_names],
        key=lambda x: (-x[1], x[0]),
    )
    return ranked[:n]


# ---------------------------------------------------------------------------
# Feature 2 – Movie popularity in genre (top n in a genre)
# ---------------------------------------------------------------------------

def top_n_movies_in_genre(
    movies: list[dict], ratings: list[dict], genre: str, n: int
) -> list[tuple[str, float]]:
    """
    Return the top-n movies within a specific genre, ranked by average rating.

    Comparison is case-insensitive for the genre argument but the original
    genre casing is preserved in the data structures; movie names are
    case-sensitive as stored.

    Args:
        movies:  List of movie dicts (from load_movies).
        ratings: List of rating dicts (from load_ratings).
        genre:   Genre name (case-insensitive match).
        n:       Number of movies to return.

    Returns:
        List of (movie_name, avg_rating) tuples, length ≤ n.
    """
    genre_lower = genre.strip().lower()
    genre_movies = {
        m["movie_name"]
        for m in movies
        if m["genre"].strip().lower() == genre_lower
    }
    avg = _build_avg_ratings(ratings)
    ranked = sorted(
        [(name, score) for name, score in avg.items() if name in genre_movies],
        key=lambda x: (-x[1], x[0]),
    )
    return ranked[:n]


# ---------------------------------------------------------------------------
# Feature 3 – Genre popularity (top n genres by avg of avg ratings)
# ---------------------------------------------------------------------------

def top_n_genres(
    movies: list[dict], ratings: list[dict], n: int
) -> list[tuple[str, float]]:
    """
    Return the top-n genres ranked by the average of their movies' average ratings.

    Only movies that have at least one rating contribute. Genres whose movies
    have no ratings at all are excluded. Ties are broken alphabetically.

    Args:
        movies:  List of movie dicts (from load_movies).
        ratings: List of rating dicts (from load_ratings).
        n:       Number of genres to return.

    Returns:
        List of (genre, avg_of_avgs) tuples, length ≤ n.
    """
    avg = _build_avg_ratings(ratings)
    name_to_genre = _movie_name_to_genre(movies)

    genre_totals: dict[str, float] = defaultdict(float)
    genre_counts: dict[str, int] = defaultdict(int)
    for movie_name, score in avg.items():
        if movie_name in name_to_genre:
            g = name_to_genre[movie_name]
            genre_totals[g] += score
            genre_counts[g] += 1

    ranked = sorted(
        [(g, genre_totals[g] / genre_counts[g]) for g in genre_totals],
        key=lambda x: (-x[1], x[0]),
    )
    return ranked[:n]


# ---------------------------------------------------------------------------
# Feature 4 – User preference for genre
# ---------------------------------------------------------------------------

def user_top_genre(
    movies: list[dict], ratings: list[dict], user_id: str
) -> str | None:
    """
    Return the genre most preferred by a user.

    Preference is measured as the average of the user's average ratings for
    each genre (i.e., for each genre the user has rated movies in, compute
    the mean of those ratings; the genre with the highest mean wins).
    Ties are broken alphabetically by genre name.

    Args:
        movies:   List of movie dicts (from load_movies).
        ratings:  List of rating dicts (from load_ratings).
        user_id:  The user's ID string (case-sensitive).

    Returns:
        The genre name string, or None if the user has no ratings.
    """
    name_to_genre = _movie_name_to_genre(movies)
    user_ratings = [r for r in ratings if r["user_id"] == user_id]
    if not user_ratings:
        return None

    genre_totals: dict[str, float] = defaultdict(float)
    genre_counts: dict[str, int] = defaultdict(int)
    for r in user_ratings:
        genre = name_to_genre.get(r["movie_name"])
        if genre is None:
            continue
        genre_totals[genre] += r["rating"]
        genre_counts[genre] += 1

    if not genre_counts:
        return None

    best_score = max(genre_totals[g] / genre_counts[g] for g in genre_totals)
    candidates = [
        g for g in genre_totals
        if genre_totals[g] / genre_counts[g] == best_score
    ]
    return sorted(candidates)[0]


# ---------------------------------------------------------------------------
# Feature 5 – Recommend movies
# ---------------------------------------------------------------------------

def recommend_movies(
    movies: list[dict], ratings: list[dict], user_id: str
) -> list[tuple[str, float]]:
    """
    Recommend the 3 most popular (by average rating) unrated movies from the
    user's top genre.

    Steps:
      1. Find the user's top genre via user_top_genre().
      2. Collect all movies in that genre the user has NOT yet rated.
      3. Return the top 3 by average rating (ties broken alphabetically).
         Movies with no ratings are included last (avg treated as -1 for
         sorting but not shown), sorted alphabetically among themselves.

    Args:
        movies:   List of movie dicts (from load_movies).
        ratings:  List of rating dicts (from load_ratings).
        user_id:  The user's ID string (case-sensitive).

    Returns:
        List of (movie_name, avg_rating) tuples (up to 3).
        avg_rating is None for movies that have no ratings.
        Returns an empty list if no top genre or no unrated movies found.
    """
    top_genre = user_top_genre(movies, ratings, user_id)
    if top_genre is None:
        return []

    genre_lower = top_genre.strip().lower()
    genre_movies = {
        m["movie_name"]
        for m in movies
        if m["genre"].strip().lower() == genre_lower
    }

    rated_by_user = {r["movie_name"] for r in ratings if r["user_id"] == user_id}
    unrated = genre_movies - rated_by_user

    avg = _build_avg_ratings(ratings)

    rated_unrated = [(name, avg[name]) for name in unrated if name in avg]
    unrated_unrated = [(name, None) for name in unrated if name not in avg]

    rated_unrated.sort(key=lambda x: (-x[1], x[0]))
    unrated_unrated.sort(key=lambda x: x[0])

    combined = rated_unrated + unrated_unrated
    return combined[:3]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_table(rows: list, headers: list[str]) -> None:
    """Pretty-print a list of tuples as a table."""
    col_widths = [len(h) for h in headers]
    str_rows = [[str(v) if v is not None else "N/A" for v in row] for row in rows]
    for row in str_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in str_rows:
        print(fmt.format(*row))


def _require_data(movies, ratings) -> bool:
    """Return True if both datasets are loaded; print a warning otherwise."""
    if not movies or not ratings:
        print("[!] Please load both movies and ratings files first (options 1 & 2).")
        return False
    return True


def main() -> None:
    """Entry point for the CLI movie recommender."""
    movies: list[dict] = []
    ratings: list[dict] = []

    menu = """
========================================
       Movie Recommender System
========================================
 1. Load movies file
 2. Load ratings file
 3. Top N movies (overall)
 4. Top N movies in a genre
 5. Top N genres
 6. User's top genre
 7. Recommend movies for a user
 0. Exit
----------------------------------------
Choice: """

    while True:
        choice = input(menu).strip()

        if choice == "0":
            print("Goodbye!")
            break

        elif choice == "1":
            path = input("Movies file path: ").strip()
            movies = load_movies(path)
            print(f"  Loaded {len(movies)} movies.")

        elif choice == "2":
            path = input("Ratings file path: ").strip()
            ratings = load_ratings(path)
            print(f"  Loaded {len(ratings)} ratings.")

        elif choice == "3":
            if not _require_data(movies, ratings):
                continue
            try:
                n = int(input("  How many movies (N)? ").strip())
            except ValueError:
                print("  Invalid number."); continue
            results = top_n_movies(movies, ratings, n)
            if results:
                _print_table([(i + 1, name, f"{score:.4f}") for i, (name, score) in enumerate(results)],
                             ["Rank", "Movie", "Avg Rating"])
            else:
                print("  No results.")

        elif choice == "4":
            if not _require_data(movies, ratings):
                continue
            genre = input("  Genre: ").strip()
            try:
                n = int(input("  How many movies (N)? ").strip())
            except ValueError:
                print("  Invalid number."); continue
            results = top_n_movies_in_genre(movies, ratings, genre, n)
            if results:
                _print_table([(i + 1, name, f"{score:.4f}") for i, (name, score) in enumerate(results)],
                             ["Rank", "Movie", "Avg Rating"])
            else:
                print("  No results.")

        elif choice == "5":
            if not _require_data(movies, ratings):
                continue
            try:
                n = int(input("  How many genres (N)? ").strip())
            except ValueError:
                print("  Invalid number."); continue
            results = top_n_genres(movies, ratings, n)
            if results:
                _print_table([(i + 1, g, f"{score:.4f}") for i, (g, score) in enumerate(results)],
                             ["Rank", "Genre", "Avg of Avgs"])
            else:
                print("  No results.")

        elif choice == "6":
            if not _require_data(movies, ratings):
                continue
            user_id = input("  User ID: ").strip()
            result = user_top_genre(movies, ratings, user_id)
            if result:
                print(f"  Top genre for user '{user_id}': {result}")
            else:
                print(f"  No data found for user '{user_id}'.")

        elif choice == "7":
            if not _require_data(movies, ratings):
                continue
            user_id = input("  User ID: ").strip()
            results = recommend_movies(movies, ratings, user_id)
            if results:
                _print_table(
                    [(i + 1, name, f"{score:.4f}" if score is not None else "N/A")
                     for i, (name, score) in enumerate(results)],
                    ["Rank", "Movie", "Avg Rating"],
                )
            else:
                print(f"  No recommendations available for user '{user_id}'.")

        else:
            print("  Unknown option. Please try again.")


if __name__ == "__main__":
    main()
