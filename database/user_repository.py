# 用户画像 Repository
# 负责 users 表的 CRUD：新建用户、查询画像、批量 upsert（导入 MovieLens 数据时用）
# user_id 从 900001 开始自增，避免与 MovieLens 原有的 1-6040 用户 id 冲突
from .mysql_client import create_mysql_connection


CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(64) NOT NULL UNIQUE,
    gender VARCHAR(8) NOT NULL DEFAULT 'U',
    age INT NOT NULL,
    occupation INT NOT NULL,
    zip_code VARCHAR(32) NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) AUTO_INCREMENT = 900001;
"""


class UserProfileRepository:
    def __init__(self, connection_factory=create_mysql_connection):
        # connection_factory 可注入 mock，方便测试
        self.connection_factory = connection_factory

    def initialize_schema(self):
        # 建表（如果不存在）
        self._execute(lambda cursor: cursor.execute(CREATE_USERS_TABLE_SQL))

    def create_user(self, username, age, occupation):
        # 校验并规范化输入，写入数据库，返回新分配的 user_id
        username = validate_username(username)
        age = normalize_age_to_movielens_bucket(age)
        occupation = validate_occupation(occupation)
        return self._execute(lambda cursor: insert_user(cursor, username, age, occupation))

    def get_user_by_username(self, username):
        # 按用户名查询，返回 {user_id, username} 或 None
        row = self._fetchone(
            "SELECT user_id, username FROM users WHERE username = %s",
            (str(username),),
        )
        return {"user_id": int(row["user_id"]), "username": row["username"]} if row else None

    def get_user_profile(self, user_id):
        # 按 user_id 查询，返回 {user_id, age, occupation} 或 None
        row = self._fetchone(
            "SELECT user_id, age, occupation FROM users WHERE user_id = %s",
            (int(user_id),),
        )
        return to_profile(row) if row else None

    def list_user_profiles(self):
        # 返回全量用户画像，格式为 {user_id: {"age": str, "occupation": str}}
        # 供冷启动推荐器构建分群统计用
        rows = self._fetchall("SELECT user_id, age, occupation FROM users")
        return {
            int(row["user_id"]): {"age": str(row["age"]), "occupation": str(row["occupation"])}
            for row in rows
        }

    def list_users(self):
        # 返回完整用户列表（含 username/gender/zip_code），供训练数据导出用
        rows = self._fetchall(
            "SELECT user_id, username, gender, age, occupation, zip_code FROM users ORDER BY user_id"
        )
        return [to_user(row) for row in rows]

    def upsert_users(self, users):
        # 批量写入用户数据；遇到相同 user_id 时更新字段（用于导入 MovieLens .dat 数据）
        params = [to_upsert_user_params(user) for user in users]
        if not params:
            return
        self._execute(
            lambda cursor: cursor.executemany(
                """
                INSERT INTO users (user_id, username, gender, age, occupation, zip_code)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    username = VALUES(username),
                    gender = VALUES(gender),
                    age = VALUES(age),
                    occupation = VALUES(occupation),
                    zip_code = VALUES(zip_code)
                """,
                params,
            )
        )

    # --- 内部执行方法：每次操作新建并归还连接 ---

    def _fetchone(self, sql, params=None):
        return self._execute(lambda cursor: fetchone(cursor, sql, params))

    def _fetchall(self, sql, params=None):
        return self._execute(lambda cursor: fetchall(cursor, sql, params))

    def _execute(self, operation):
        connection = self.connection_factory()
        try:
            with connection.cursor() as cursor:
                return operation(cursor)
        finally:
            connection.close()


# ---------- SQL 执行工具 ----------

def insert_user(cursor, username, age, occupation):
    cursor.execute(
        "INSERT INTO users (username, age, occupation) VALUES (%s, %s, %s)",
        (username, age, occupation),
    )
    return int(cursor.lastrowid)


def fetchone(cursor, sql, params=None):
    cursor.execute(sql, params)
    return cursor.fetchone()


def fetchall(cursor, sql, params=None):
    cursor.execute(sql, params)
    return cursor.fetchall()


# ---------- 行数据转换 ----------

def to_profile(row):
    return {"user_id": int(row["user_id"]), "age": int(row["age"]), "occupation": int(row["occupation"])}


def to_user(row):
    return {
        "user_id": int(row["user_id"]),
        "username": row["username"],
        "gender": row["gender"],
        "age": int(row["age"]),
        "occupation": int(row["occupation"]),
        "zip_code": row["zip_code"],
    }


def to_upsert_user_params(user):
    return (
        int(user["user_id"]),
        validate_username(user["username"]),
        validate_gender(user.get("gender", "U")),
        normalize_age_to_movielens_bucket(user["age"]),
        validate_occupation(user["occupation"]),
        str(user.get("zip_code", "")),
    )


# ---------- 输入校验与规范化 ----------

def validate_username(username):
    # 非空字符串，最长 64 字符
    if username is None:
        raise ValueError("用户名不能为空")
    normalized = str(username).strip()
    if not normalized:
        raise ValueError("用户名不能为空")
    if len(normalized) > 64:
        raise ValueError("用户名最长 64 个字符")
    return normalized


def validate_occupation(occupation):
    # MovieLens 职业编码范围 0-20
    occupation = int(occupation)
    if occupation < 0 or occupation > 20:
        raise ValueError("职业编码须在 0-20 之间")
    return occupation


def validate_gender(gender):
    normalized = str(gender or "U").strip() or "U"
    if len(normalized) > 8:
        raise ValueError("性别字段最长 8 个字符")
    return normalized


def normalize_age_to_movielens_bucket(age):
    # 将真实年龄映射到 MovieLens 的 7 个年龄段编码
    # 段划分：<18, 18-24, 25-34, 35-44, 45-49, 50-55, 56+
    age = int(age)
    if age <= 0:
        raise ValueError("年龄必须为正整数")
    if age < 18:  return 1
    if age < 25:  return 18
    if age < 35:  return 25
    if age < 45:  return 35
    if age < 50:  return 45
    if age < 56:  return 50
    return 56
