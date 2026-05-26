import unittest

from database.mysql_client import MysqlConfig
from database.mysql_client import get_mysql_config_from_env
from database.user_repository import UserProfileRepository
from database.user_repository import normalize_age_to_movielens_bucket


class MysqlClientConfigTest(unittest.TestCase):
    def test_get_mysql_config_from_env_returns_none_without_credentials(self):
        env = {
            "MYSQL_HOST": "localhost",
            "MYSQL_PORT": "3306",
            "MYSQL_DATABASE": "recommend",
        }

        config = get_mysql_config_from_env(env)

        self.assertIsNone(config)

    def test_get_mysql_config_from_env_reads_database_settings(self):
        env = {
            "MYSQL_HOST": "127.0.0.1",
            "MYSQL_PORT": "3307",
            "MYSQL_USER": "root",
            "MYSQL_PASSWORD": "secret",
            "MYSQL_DATABASE": "recommend",
            "MYSQL_CONNECT_TIMEOUT": "5",
        }

        config = get_mysql_config_from_env(env)

        self.assertEqual(
            config,
            MysqlConfig(
                host="127.0.0.1",
                port=3307,
                user="root",
                password="secret",
                database="recommend",
                connect_timeout=5,
            ),
        )


class FakeCursor:
    def __init__(self, row=None, rows=None, lastrowid=900001):
        self.row = row
        self.rows = rows or []
        self.lastrowid = lastrowid
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.row

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


class UserProfileRepositoryTest(unittest.TestCase):
    def test_normalize_age_to_movielens_bucket(self):
        self.assertEqual(normalize_age_to_movielens_bucket(17), 1)
        self.assertEqual(normalize_age_to_movielens_bucket(18), 18)
        self.assertEqual(normalize_age_to_movielens_bucket(24), 18)
        self.assertEqual(normalize_age_to_movielens_bucket(25), 25)
        self.assertEqual(normalize_age_to_movielens_bucket(34), 25)
        self.assertEqual(normalize_age_to_movielens_bucket(35), 35)
        self.assertEqual(normalize_age_to_movielens_bucket(44), 35)
        self.assertEqual(normalize_age_to_movielens_bucket(45), 45)
        self.assertEqual(normalize_age_to_movielens_bucket(49), 45)
        self.assertEqual(normalize_age_to_movielens_bucket(50), 50)
        self.assertEqual(normalize_age_to_movielens_bucket(55), 50)
        self.assertEqual(normalize_age_to_movielens_bucket(56), 56)

    def test_create_user_inserts_normalized_profile_and_returns_user_id(self):
        cursor = FakeCursor(lastrowid=900123)
        connection = FakeConnection(cursor)
        repository = UserProfileRepository(connection_factory=lambda: connection)

        user_id = repository.create_user(username="alice", age=23, occupation=4)

        self.assertEqual(user_id, 900123)
        self.assertTrue(connection.closed)
        sql, params = cursor.executed[0]
        self.assertIn("INSERT INTO users", sql)
        self.assertEqual(params, ("alice", 18, 4))

    def test_get_user_profile_returns_standard_profile_dict(self):
        cursor = FakeCursor(row={"user_id": 900123, "age": 25, "occupation": 4})
        connection = FakeConnection(cursor)
        repository = UserProfileRepository(connection_factory=lambda: connection)

        profile = repository.get_user_profile(900123)

        self.assertEqual(
            profile,
            {
                "user_id": 900123,
                "age": 25,
                "occupation": 4,
            },
        )
        self.assertTrue(connection.closed)

    def test_get_user_profile_returns_none_when_user_is_missing(self):
        cursor = FakeCursor(row=None)
        connection = FakeConnection(cursor)
        repository = UserProfileRepository(connection_factory=lambda: connection)

        profile = repository.get_user_profile(900999)

        self.assertIsNone(profile)
        self.assertTrue(connection.closed)

    def test_list_user_profiles_returns_profile_mapping_for_cold_start_segments(self):
        cursor = FakeCursor(
            rows=[
                {"user_id": 1, "age": 25, "occupation": 4},
                {"user_id": 2, "age": 35, "occupation": 7},
            ]
        )
        connection = FakeConnection(cursor)
        repository = UserProfileRepository(connection_factory=lambda: connection)

        profiles = repository.list_user_profiles()

        self.assertEqual(
            profiles,
            {
                1: {"age": "25", "occupation": "4"},
                2: {"age": "35", "occupation": "7"},
            },
        )
        self.assertTrue(connection.closed)

    def test_create_user_rejects_invalid_profile(self):
        repository = UserProfileRepository(connection_factory=lambda: FakeConnection(FakeCursor()))

        with self.assertRaises(ValueError):
            repository.create_user(username="", age=25, occupation=4)

        with self.assertRaises(ValueError):
            repository.create_user(username="alice", age=0, occupation=4)

        with self.assertRaises(ValueError):
            repository.create_user(username="alice", age=25, occupation=21)


class UserProfileScriptImportTest(unittest.TestCase):
    def test_schema_script_can_be_imported(self):
        from scripts.init_mysql_schema import main

        self.assertTrue(callable(main))

    def test_register_script_can_be_imported(self):
        from scripts.register_user import main

        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
