# MySQL 连接工具
# 从环境变量读取连接配置，返回 pymysql 连接对象
# 需要设置的环境变量：MYSQL_USER、MYSQL_PASSWORD（必填），
#                    MYSQL_HOST、MYSQL_PORT、MYSQL_DATABASE（可选，有默认值）
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
    # 读取环境变量；user 和 password 缺一不可，否则返回 None 表示未配置
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
    # 若未传入 config，自动从环境变量读取；pymysql 未安装时给出友好提示
    config = config or get_mysql_config_from_env()

    if config is None:
        raise RuntimeError("MySQL 未配置，请设置环境变量 MYSQL_USER 和 MYSQL_PASSWORD。")

    try:
        import pymysql
    except ImportError as error:
        raise RuntimeError("缺少依赖 pymysql，请执行 pip install pymysql 安装。") from error

    return pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
        connect_timeout=config.connect_timeout,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,  # 查询结果以字典形式返回，方便按列名访问
        autocommit=True,
    )
