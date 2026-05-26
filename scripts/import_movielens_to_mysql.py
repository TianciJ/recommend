import argparse
from pathlib import Path

try:
    from scripts.bootstrap import add_project_root_to_path
except ModuleNotFoundError:
    from bootstrap import add_project_root_to_path


add_project_root_to_path()

from database import MysqlDatasetRepository
from database import UserProfileRepository


BASE_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Import MovieLens dat files into MySQL.")
    parser.add_argument("--train-dir", default=str(BASE_DIR / "train_data"))
    parser.add_argument("--test-dir", default=str(BASE_DIR / "test_data"))
    return parser.parse_args()


def read_users(users_path):
    users = []

    with users_path.open("r", encoding="utf-8") as users_file:
        for line in users_file:
            user_id, gender, age, occupation, zip_code = line.strip().split("::")
            users.append(
                {
                    "user_id": int(user_id),
                    "username": f"movielens_{user_id}",
                    "gender": gender,
                    "age": int(age),
                    "occupation": int(occupation),
                    "zip_code": zip_code,
                }
            )

    return users


def read_movies(movies_path):
    movies = []

    with movies_path.open("r", encoding="latin-1") as movies_file:
        for line in movies_file:
            movie_id, title, genres = line.strip().split("::")
            movies.append(
                {
                    "movie_id": int(movie_id),
                    "title": title,
                    "genres": genres.split("|"),
                }
            )

    return movies


def read_ratings(ratings_path):
    ratings = []

    with ratings_path.open("r", encoding="utf-8") as ratings_file:
        for line in ratings_file:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            ratings.append(
                {
                    "user_id": int(user_id),
                    "movie_id": int(movie_id),
                    "rating": int(rating),
                    "timestamp": int(timestamp),
                }
            )

    return ratings


def main():
    args = parse_args()
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    dataset_repository = MysqlDatasetRepository()
    user_repository = UserProfileRepository()
    dataset_repository.initialize_schema()

    users = read_users(train_dir / "users.dat")
    movies = read_movies(train_dir / "movies.dat")
    train_ratings = read_ratings(train_dir / "ratings.dat")
    test_ratings = read_ratings(test_dir / "ratings.dat")

    user_repository.upsert_users(users)
    dataset_repository.upsert_movies(movies)
    dataset_repository.upsert_ratings(train_ratings, split="train")
    dataset_repository.upsert_ratings(test_ratings, split="test")

    print(
        f"Imported users={len(users)} movies={len(movies)} "
        f"train_ratings={len(train_ratings)} test_ratings={len(test_ratings)}"
    )


if __name__ == "__main__":
    main()
