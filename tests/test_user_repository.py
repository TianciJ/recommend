import unittest

from database.mysql_client import MysqlConfig
from database.mysql_client import get_mysql_config_from_env


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


if __name__ == "__main__":
    unittest.main()
