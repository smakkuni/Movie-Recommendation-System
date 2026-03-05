# Authors: Siddarth Makkuni skm179, Dhruv Patel dp1379, Melania Labadze ML1854

"""
Movie Recommender System
Implements movie popularity rankings, genre analysis, and personalized recommendations.
"""

from collections import defaultdict

def load_movies(filepath: str) -> list[dict]:
    """
    Load movies from a text file.
    Format: movie_genre|movie_id|movie_name
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
                    continue
                genre, movie_id, movie_name = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if not genre or not movie_id or not movie_name:
                    continue
                movies.append({"genre": genre, "movie_id": movie_id, "movie_name": movie_name})
    except FileNotFoundError:
        print(f"[Error] File not found: {filepath}")
    except OSError as e:
        print(f"[Error] Could not read {filepath}: {e}")
    return movies


def load_ratings(filepath: str) -> list[dict]:
    """
    Load ratings from a text file.
    Format: movie_name|rating|user_id
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
                    continue
                if not (0.0 <= rating <= 5.0):
                    continue
                ratings.append({"movie_name": movie_name, "rating": rating, "user_id": user_id})
    except FileNotFoundError:
        print(f"[Error] File not found: {filepath}")
    except OSError as e:
        print(f"[Error] Could not read {filepath}: {e}")
    return ratings


def _build_avg_ratings(ratings: list[dict]) -> dict[str, float]:
    """
    Compute the average rating for every movie.
    """
    totals = defaultdict(float)
    counts = defaultdict(int)
    for r in ratings:
        totals[r["movie_name"]] += r["rating"]
        counts[r["movie_name"]] += 1
    return {name: totals[name] / counts[name] for name in totals}


def _movie_name_to_genre(movies: list[dict]) -> dict[str, str]:
    """
    Build a mapping from movie_name to genre.
    """
    return {m["movie_name"]: m["genre"] for m in movies}


def top_n_movies(movies: list[dict], ratings: list[dict], n: int) -> list[tuple[str, float]]:
    """
    Return the top-n movies ranked by average rating.
    """
    avg = _build_avg_ratings(ratings)
    known_names = {m["movie_name"] for m in movies}
    ranked = sorted(
        [(name, score) for name, score in avg.items() if name in known_names],
        key=lambda x: (-x[1], x[0]),
    )
    return ranked[:n]


def top_n_movies_in_genre(movies: list[dict], ratings: list[dict], genre: str, n: int) -> list[tuple[str, float]]:
    """
    Return the top-n movies in a specific genre.
    """
    genre_lower = genre.strip().lower()
    genre_movies = {m["movie_name"] for m in movies if m["genre"].strip().lower() == genre_lower}
    avg = _build_avg_ratings(ratings)

    ranked = sorted(
        [(name, score) for name, score in avg.items() if name in genre_movies],
        key=lambda x: (-x[1], x[0]),
    )
    return ranked[:n]


def top_n_genres(movies: list[dict], ratings: list[dict], n: int) -> list[tuple[str, float]]:
    """
    Return the top-n genres ranked by the average of movie averages.
    """
    avg = _build_avg_ratings(ratings)
    name_to_genre = _movie_name_to_genre(movies)

    genre_totals = defaultdict(float)
    genre_counts = defaultdict(int)

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


def user_top_genre(movies: list[dict], ratings: list[dict], user_id: str) -> str | None:
    """
    Return the genre most preferred by a user.
    """
    name_to_genre = _movie_name_to_genre(movies)
    user_ratings = [r for r in ratings if r["user_id"] == user_id]

    if not user_ratings:
        return None

    genre_totals = defaultdict(float)
    genre_counts = defaultdict(int)

    for r in user_ratings:
        genre = name_to_genre.get(r["movie_name"])
        if genre is None:
            continue
        genre_totals[genre] += r["rating"]
        genre_counts[genre] += 1

    if not genre_counts:
        return None

    best_score = max(genre_totals[g] / genre_counts[g] for g in genre_totals)
    candidates = [g for g in genre_totals if genre_totals[g] / genre_counts[g] == best_score]

    return sorted(candidates)[0]


def recommend_movies(movies: list[dict], ratings: list[dict], user_id: str) -> list[tuple[str, float]]:
    """
    Recommend up to 3 movies from the user's top genre that they have not rated.
    """
    top_genre = user_top_genre(movies, ratings, user_id)

    if top_genre is None:
        return []

    genre_lower = top_genre.strip().lower()
    genre_movies = {m["movie_name"] for m in movies if m["genre"].strip().lower() == genre_lower}

    rated_by_user = {r["movie_name"] for r in ratings if r["user_id"] == user_id}
    unrated = genre_movies - rated_by_user

    avg = _build_avg_ratings(ratings)

    rated_unrated = [(name, avg[name]) for name in unrated if name in avg]
    unrated_unrated = [(name, None) for name in unrated if name not in avg]

    rated_unrated.sort(key=lambda x: (-x[1], x[0]))
    unrated_unrated.sort(key=lambda x: x[0])

    return (rated_unrated + unrated_unrated)[:3]


def _print_table(rows: list, headers: list[str]) -> None:
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
    if not movies or not ratings:
        print("[!] Please load both movies and ratings files first.")
        return False
    return True


def main() -> None:
    movies = []
    ratings = []

    menu = """
Movie Recommender System
1. Load movies file
2. Load ratings file
3. Top N movies
4. Top N movies in a genre
5. Top N genres
6. User's top genre
7. Recommend movies
0. Exit
Choice: """

    while True:
        choice = input(menu).strip()

        if choice == "0":
            break

        elif choice == "1":
            path = input("Movies file path: ").strip()
            movies = load_movies(path)
            print(f"Loaded {len(movies)} movies")

        elif choice == "2":
            path = input("Ratings file path: ").strip()
            ratings = load_ratings(path)
            print(f"Loaded {len(ratings)} ratings")

        elif choice == "3":
            if not _require_data(movies, ratings):
                continue
            n = int(input("How many movies? ").strip())
            results = top_n_movies(movies, ratings, n)
            _print_table([(i + 1, name, f"{score:.4f}") for i, (name, score) in enumerate(results)],
                         ["Rank", "Movie", "Avg Rating"])

        elif choice == "4":
            if not _require_data(movies, ratings):
                continue
            genre = input("Genre: ").strip()
            n = int(input("How many movies? ").strip())
            results = top_n_movies_in_genre(movies, ratings, genre, n)
            _print_table([(i + 1, name, f"{score:.4f}") for i, (name, score) in enumerate(results)],
                         ["Rank", "Movie", "Avg Rating"])

        elif choice == "5":
            if not _require_data(movies, ratings):
                continue
            n = int(input("How many genres? ").strip())
            results = top_n_genres(movies, ratings, n)
            _print_table([(i + 1, g, f"{score:.4f}") for i, (g, score) in enumerate(results)],
                         ["Rank", "Genre", "Avg Rating"])

        elif choice == "6":
            if not _require_data(movies, ratings):
                continue
            user_id = input("User ID: ").strip()
            result = user_top_genre(movies, ratings, user_id)
            print(result if result else "No data for that user")

        elif choice == "7":
            if not _require_data(movies, ratings):
                continue
            user_id = input("User ID: ").strip()
            results = recommend_movies(movies, ratings, user_id)
            _print_table([(i + 1, name, f"{score:.4f}" if score else "N/A")
                          for i, (name, score) in enumerate(results)],
                         ["Rank", "Movie", "Avg Rating"])
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()
