import unittest

from database.dataset_repository import CREATE_MOVIES_TABLE_SQL
from database.dataset_repository import CREATE_RATINGS_TABLE_SQL
from database.dataset_repository import MysqlDatasetRepository


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = []
        self.executemany_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def executemany(self, sql, params):
        self.executemany_calls.append((sql, params))

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_instance = cursor
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def close(self):
        self.closed = True


class MysqlDatasetRepositoryTest(unittest.TestCase):
    def test_initialize_schema_creates_movies_and_ratings_tables(self):
        cursor = FakeCursor()
        connection = FakeConnection(cursor)
        repository = MysqlDatasetRepository(connection_factory=lambda: connection)

        repository.initialize_schema()

        executed_sql = "\n".join(sql for sql, params in cursor.executed)
        self.assertIn(CREATE_MOVIES_TABLE_SQL.strip(), executed_sql)
        self.assertIn(CREATE_RATINGS_TABLE_SQL.strip(), executed_sql)
        self.assertTrue(connection.closed)

    def test_list_movies_returns_movie_rows_with_genre_lists(self):
        cursor = FakeCursor(
            rows=[
                {"movie_id": 10, "title": "Movie A", "genres": "Drama|Comedy"},
                {"movie_id": 20, "title": "Movie B", "genres": "Action"},
            ]
        )
        connection = FakeConnection(cursor)
        repository = MysqlDatasetRepository(connection_factory=lambda: connection)

        movies = repository.list_movies()

        self.assertEqual(
            movies,
            [
                {"movie_id": 10, "title": "Movie A", "genres": ["Drama", "Comedy"]},
                {"movie_id": 20, "title": "Movie B", "genres": ["Action"]},
            ],
        )

    def test_list_ratings_returns_rating_rows(self):
        cursor = FakeCursor(
            rows=[
                {"user_id": 1, "movie_id": 10, "rating": 5, "rating_timestamp": 100},
                {"user_id": 2, "movie_id": 20, "rating": 3, "rating_timestamp": 200},
            ]
        )
        connection = FakeConnection(cursor)
        repository = MysqlDatasetRepository(connection_factory=lambda: connection)

        ratings = repository.list_ratings(split="train")

        self.assertEqual(
            ratings,
            [
                {"user_id": 1, "movie_id": 10, "rating": 5, "timestamp": 100},
                {"user_id": 2, "movie_id": 20, "rating": 3, "timestamp": 200},
            ],
        )
        self.assertEqual(cursor.executed[0][1], ("train",))

    def test_import_methods_batch_rows(self):
        cursor = FakeCursor()
        connection = FakeConnection(cursor)
        repository = MysqlDatasetRepository(connection_factory=lambda: connection)

        repository.upsert_movies(
            [{"movie_id": 10, "title": "Movie A", "genres": ["Drama", "Comedy"]}]
        )
        repository.upsert_ratings(
            [{"user_id": 1, "movie_id": 10, "rating": 5, "timestamp": 100}],
            split="train",
        )

        self.assertEqual(cursor.executemany_calls[0][1], [(10, "Movie A", "Drama|Comedy")])
        self.assertEqual(cursor.executemany_calls[1][1], [(1, 10, 5, 100, "train")])
        self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
