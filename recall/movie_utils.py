from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MOVIES_PATH = BASE_DIR / "train_data" / "movies.dat"


def load_movie_titles(movies_path=MOVIES_PATH, movies=None):
    if movies is None:
        movies = load_mysql_movies_if_configured()

    if movies is not None:
        return {int(m["movie_id"]): m["title"] for m in movies}

    movie_titles = {}
    with movies_path.open("r", encoding="latin-1") as f:
        for line in f:
            movie_id, title, _ = line.strip().split("::")
            movie_titles[int(movie_id)] = title
    return movie_titles


def add_movie_titles(recommendations, movies_path=MOVIES_PATH, movies=None):
    movie_titles = load_movie_titles(movies_path, movies=movies)
    for item in recommendations:
        item["title"] = movie_titles.get(item["movie_id"], "")
    return recommendations


def load_mysql_movies_if_configured():
    from database.dataset_repository import load_mysql_dataset

    dataset = load_mysql_dataset(split="train")

    if dataset is None:
        return None

    return dataset["movies"]


def print_recommendations(recommendations):
    for rank, item in enumerate(recommendations, start=1):
        print(f"{rank}. movie_id={item['movie_id']} score={item['score']:.4f} title={item.get('title', '')}")
