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
        self.connection_factory = connection_factory

    def initialize_schema(self):
        self._execute(lambda cursor: cursor.execute(CREATE_USERS_TABLE_SQL))

    def create_user(self, username, age, occupation):
        username = validate_username(username)
        age = normalize_age_to_movielens_bucket(age)
        occupation = validate_occupation(occupation)

        return self._execute(
            lambda cursor: insert_user(cursor, username, age, occupation)
        )

    def get_user_profile(self, user_id):
        row = self._fetchone(
            """
            SELECT user_id, age, occupation
            FROM users
            WHERE user_id = %s
            """,
            (int(user_id),),
        )

        if row is None:
            return None

        return to_profile(row)

    def list_user_profiles(self):
        rows = self._fetchall(
            """
            SELECT user_id, age, occupation
            FROM users
            """
        )

        return {
            int(row["user_id"]): {
                "age": str(row["age"]),
                "occupation": str(row["occupation"]),
            }
            for row in rows
        }

    def list_users(self):
        rows = self._fetchall(
            """
            SELECT user_id, username, gender, age, occupation, zip_code
            FROM users
            ORDER BY user_id
            """
        )

        return [to_user(row) for row in rows]

    def upsert_users(self, users):
        params = [to_upsert_user_params(user) for user in users]
        if not params:
            return

        self._execute(
            lambda cursor: cursor.executemany(
                """
                INSERT INTO users (
                    user_id, username, gender, age, occupation, zip_code
                )
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


def insert_user(cursor, username, age, occupation):
    cursor.execute(
        """
        INSERT INTO users (username, age, occupation)
        VALUES (%s, %s, %s)
        """,
        (username, age, occupation),
    )
    return int(cursor.lastrowid)


def fetchone(cursor, sql, params=None):
    cursor.execute(sql, params)
    return cursor.fetchone()


def fetchall(cursor, sql, params=None):
    cursor.execute(sql, params)
    return cursor.fetchall()


def to_profile(row):
    return {
        "user_id": int(row["user_id"]),
        "age": int(row["age"]),
        "occupation": int(row["occupation"]),
    }


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


def validate_username(username):
    if username is None:
        raise ValueError("username is required")

    normalized_username = str(username).strip()

    if not normalized_username:
        raise ValueError("username is required")

    if len(normalized_username) > 64:
        raise ValueError("username must be at most 64 characters")

    return normalized_username


def validate_occupation(occupation):
    occupation = int(occupation)

    if occupation < 0 or occupation > 20:
        raise ValueError("occupation must be between 0 and 20")

    return occupation


def validate_gender(gender):
    normalized_gender = str(gender or "U").strip() or "U"

    if len(normalized_gender) > 8:
        raise ValueError("gender must be at most 8 characters")

    return normalized_gender


def normalize_age_to_movielens_bucket(age):
    age = int(age)

    if age <= 0:
        raise ValueError("age must be positive")

    if age < 18:
        return 1

    if age < 25:
        return 18

    if age < 35:
        return 25

    if age < 45:
        return 35

    if age < 50:
        return 45

    if age < 56:
        return 50

    return 56
