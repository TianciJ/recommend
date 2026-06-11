import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MysqlConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    connect_timeout: int


def get_mysql_config_from_env(env=None):
    env = env or os.environ

    user = env.get("MYSQL_USER")
    password = env.get("MYSQL_PASSWORD")

    if not user or password is None:
        return None

    return MysqlConfig(
        host=env.get("MYSQL_HOST", "localhost"),
        port=int(env.get("MYSQL_PORT", "3306")),
        user=user,
        password=password,
        database=env.get("MYSQL_DATABASE", "recommend"),
        connect_timeout=int(env.get("MYSQL_CONNECT_TIMEOUT", "3")),
    )


def create_mysql_connection(config=None):
    config = config or get_mysql_config_from_env()

    if config is None:
        raise RuntimeError("MySQL is not configured. Set MYSQL_USER and MYSQL_PASSWORD.")

    try:
        import pymysql
    except ImportError as error:
        raise RuntimeError("Missing dependency pymysql. Install it with: pip install pymysql") from error

    return pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
        connect_timeout=config.connect_timeout,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )
