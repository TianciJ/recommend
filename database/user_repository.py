from .mysql_client import create_mysql_connection


CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(64) NOT NULL UNIQUE,
    age INT NOT NULL,
    occupation INT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) AUTO_INCREMENT = 900001;
"""


class UserProfileRepository:
    def __init__(self, connection_factory=create_mysql_connection):
        self.connection_factory = connection_factory

    def initialize_schema(self):
        connection = self.connection_factory()

        try:
            with connection.cursor() as cursor:
                cursor.execute(CREATE_USERS_TABLE_SQL)
        finally:
            connection.close()

    def create_user(self, username, age, occupation):
        username = validate_username(username)
        age = normalize_age_to_movielens_bucket(age)
        occupation = validate_occupation(occupation)
        connection = self.connection_factory()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (username, age, occupation)
                    VALUES (%s, %s, %s)
                    """,
                    (username, age, occupation),
                )
                return int(cursor.lastrowid)
        finally:
            connection.close()

    def get_user_profile(self, user_id):
        connection = self.connection_factory()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT user_id, age, occupation
                    FROM users
                    WHERE user_id = %s
                    """,
                    (int(user_id),),
                )
                row = cursor.fetchone()
        finally:
            connection.close()

        if row is None:
            return None

        return {
            "user_id": int(row["user_id"]),
            "age": int(row["age"]),
            "occupation": int(row["occupation"]),
        }

    def list_user_profiles(self):
        connection = self.connection_factory()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT user_id, age, occupation
                    FROM users
                    """
                )
                rows = cursor.fetchall()
        finally:
            connection.close()

        return {
            int(row["user_id"]): {
                "age": str(row["age"]),
                "occupation": str(row["occupation"]),
            }
            for row in rows
        }


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
