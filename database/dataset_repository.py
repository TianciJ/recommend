# 数据集 Repository
# 负责 movies 和 ratings 两张表的读写，以及组合加载完整数据集的便捷函数
from .mysql_client import create_mysql_connection, get_mysql_config_from_env
from .user_repository import CREATE_USERS_TABLE_SQL, UserProfileRepository


CREATE_MOVIES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS movies (
    movie_id BIGINT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    genres VARCHAR(255) NOT NULL
);
"""

# split_name 区分训练集和测试集，索引覆盖常用查询模式
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
        # 建立三张表（users/movies/ratings），如果已存在则跳过
        self._execute(
            lambda cursor: [
                cursor.execute(CREATE_USERS_TABLE_SQL),
                cursor.execute(CREATE_MOVIES_TABLE_SQL),
                cursor.execute(CREATE_RATINGS_TABLE_SQL),
            ]
        )

    def list_movies(self):
        # 读取全量电影数据，genres 以 | 分隔的字符串存储，读出时拆成列表
        rows = self._fetchall("SELECT movie_id, title, genres FROM movies ORDER BY movie_id")
        return [to_movie(row) for row in rows]

    def list_ratings(self, split="train"):
        # 按 split_name 读取评分数据（train/test）
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
        # 批量写入电影数据；遇到相同 movie_id 时更新 title 和 genres
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

    def add_rating(self, user_id, movie_id, rating, timestamp, split="train"):
        # 写入单条用户评分；已有则更新分数
        self._execute(
            lambda cursor: cursor.execute(
                """
                INSERT INTO ratings (user_id, movie_id, rating, rating_timestamp, split_name)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE rating = VALUES(rating)
                """,
                (int(user_id), int(movie_id), int(rating), int(timestamp), split),
            )
        )

    def upsert_ratings(self, ratings, split="train"):
        # 批量写入评分数据；遇到相同主键时更新 rating 分数
        params = [to_upsert_rating_params(rating, split) for rating in ratings]
        if not params:
            return
        self._execute(
            lambda cursor: cursor.executemany(
                """
                INSERT INTO ratings (user_id, movie_id, rating, rating_timestamp, split_name)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE rating = VALUES(rating)
                """,
                params,
            )
        )

    def _fetchall(self, sql, params=None):
        return self._execute(lambda cursor: fetchall(cursor, sql, params))

    def _execute(self, operation):
        # 每次操作新建连接，用完即关（生产环境建议替换为连接池）
        connection = self.connection_factory()
        try:
            with connection.cursor() as cursor:
                return operation(cursor)
        finally:
            connection.close()


# ---------- SQL 执行工具 ----------

def fetchall(cursor, sql, params=None):
    cursor.execute(sql, params)
    return cursor.fetchall()


# ---------- 行数据转换 ----------

def to_movie(row):
    # genres 存储为 "Action|Comedy|Drama"，读出时拆成列表
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
    return (int(rating["user_id"]), int(rating["movie_id"]), int(rating["rating"]), int(rating["timestamp"]), split)


# ---------- 便捷加载函数 ----------

def load_mysql_dataset(split="train"):
    # 若 MySQL 未配置则返回 None，各模块检测到 None 后自动回退到 .dat 文件
    if get_mysql_config_from_env() is None:
        return None

    user_repository = UserProfileRepository()
    dataset_repository = MysqlDatasetRepository()

    return {
        "users": user_repository.list_users(),
        "movies": dataset_repository.list_movies(),
        "ratings": dataset_repository.list_ratings(split=split),
    }
