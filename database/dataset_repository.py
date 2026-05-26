from .mysql_client import create_mysql_connection
from .mysql_client import get_mysql_config_from_env
from .user_repository import CREATE_USERS_TABLE_SQL
from .user_repository import UserProfileRepository


CREATE_MOVIES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS movies (
    movie_id BIGINT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    genres VARCHAR(255) NOT NULL
);
"""


CREATE_RATINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ratings (
    user_id BIGINT NOT NULL,
    movie_id BIGINT NOT NULL,
    rating INT NOT NULL,
    rating_timestamp BIGINT NOT NULL,
    split_name VARCHAR(16) NOT NULL DEFAULT 'train',
    PRIMARY KEY (user_id, movie_id, rating_timestamp, split_name),
    INDEX idx_ratings_user_split (user_id, split_name),
    INDEX idx_ratings_movie_split (movie_id, split_name)
);
"""


class MysqlDatasetRepository:
    def __init__(self, connection_factory=create_mysql_connection):
        self.connection_factory = connection_factory

    def initialize_schema(self):
        self._execute(
            lambda cursor: [
                cursor.execute(CREATE_USERS_TABLE_SQL),
                cursor.execute(CREATE_MOVIES_TABLE_SQL),
                cursor.execute(CREATE_RATINGS_TABLE_SQL),
            ]
        )

    def list_movies(self):
        rows = self._fetchall(
            """
            SELECT movie_id, title, genres
            FROM movies
            ORDER BY movie_id
            """
        )

        return [to_movie(row) for row in rows]

    def list_ratings(self, split="train"):
        rows = self._fetchall(
            """
            SELECT user_id, movie_id, rating, rating_timestamp
            FROM ratings
            WHERE split_name = %s
            ORDER BY user_id, movie_id, rating_timestamp
            """,
            (split,),
        )

        return [to_rating(row) for row in rows]

    def upsert_movies(self, movies):
        params = [to_upsert_movie_params(movie) for movie in movies]
        if not params:
            return

        self._execute(
            lambda cursor: cursor.executemany(
                """
                INSERT INTO movies (movie_id, title, genres)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    title = VALUES(title),
                    genres = VALUES(genres)
                """,
                params,
            )
        )

    def upsert_ratings(self, ratings, split="train"):
        params = [to_upsert_rating_params(rating, split) for rating in ratings]
        if not params:
            return

        self._execute(
            lambda cursor: cursor.executemany(
                """
                INSERT INTO ratings (
                    user_id, movie_id, rating, rating_timestamp, split_name
                )
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    rating = VALUES(rating)
                """,
                params,
            )
        )

    def _fetchall(self, sql, params=None):
        return self._execute(lambda cursor: fetchall(cursor, sql, params))

    def _execute(self, operation):
        connection = self.connection_factory()

        try:
            with connection.cursor() as cursor:
                return operation(cursor)
        finally:
            connection.close()


def fetchall(cursor, sql, params=None):
    cursor.execute(sql, params)
    return cursor.fetchall()


def to_movie(row):
    return {
        "movie_id": int(row["movie_id"]),
        "title": row["title"],
        "genres": str(row["genres"]).split("|"),
    }


def to_rating(row):
    return {
        "user_id": int(row["user_id"]),
        "movie_id": int(row["movie_id"]),
        "rating": int(row["rating"]),
        "timestamp": int(row["rating_timestamp"]),
    }


def to_upsert_movie_params(movie):
    return (int(movie["movie_id"]), movie["title"], "|".join(movie["genres"]))


def to_upsert_rating_params(rating, split):
    return (
        int(rating["user_id"]),
        int(rating["movie_id"]),
        int(rating["rating"]),
        int(rating["timestamp"]),
        split,
    )


def load_mysql_dataset(split="train"):
    if get_mysql_config_from_env() is None:
        return None

    user_repository = UserProfileRepository()
    dataset_repository = MysqlDatasetRepository()

    return {
        "users": user_repository.list_users(),
        "movies": dataset_repository.list_movies(),
        "ratings": dataset_repository.list_ratings(split=split),
    }
