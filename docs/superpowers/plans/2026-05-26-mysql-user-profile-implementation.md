# MySQL User Profile Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MySQL-backed registered-user profile storage so cold-start recommendations can automatically read `age` and `occupation` by `user_id`.

**Architecture:** Add a focused `database` package with MySQL config/client helpers and a `UserProfileRepository`. Keep MySQL optional in normal recommendation flow: if no database config exists, or lookup fails, the current cold-start/global fallback behavior continues. Add small scripts for schema initialization and command-line user registration instead of introducing a web backend.

**Tech Stack:** Python stdlib, `unittest`, optional `pymysql`, MySQL 8-compatible SQL.

---

## File Structure

Create:

- `database/__init__.py`: exports database helpers used by scripts and pipeline.
- `database/mysql_client.py`: reads MySQL environment variables and creates PyMySQL connections.
- `database/user_repository.py`: owns `users` table SQL, profile validation, age bucketing, create/read operations.
- `scripts/init_mysql_schema.py`: creates the MySQL `users` table.
- `scripts/register_user.py`: simulates user registration from CLI arguments.
- `tests/test_user_repository.py`: unit tests for repository behavior using fake connections.

Modify:

- `recommender_pipeline.py`: inject optional `user_profile_repository`, auto-fill cold-start profile fields from MySQL repository.
- `tests/test_pipeline_timing.py`: add fake repository tests for pipeline cold-start profile resolution.
- `README.md`: document MySQL setup, schema init, registration, and recommendation flow.

Do not modify:

- Recall, rough-rank, fine-rank model training or inference logic.
- Existing data files in `train_data/`, `test_data/`, or `data/`.

---

### Task 1: MySQL Config And Connection Helper

**Files:**
- Create: `database/__init__.py`
- Create: `database/mysql_client.py`
- Test: `tests/test_user_repository.py`

- [ ] **Step 1: Write failing tests for MySQL config parsing**

Create `tests/test_user_repository.py` with the initial config tests:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_user_repository
```

Expected:

```text
ImportError: No module named 'database'
```

- [ ] **Step 3: Add database package exports**

Create `database/__init__.py`:

```python
from .mysql_client import MysqlConfig
from .mysql_client import create_mysql_connection
from .mysql_client import get_mysql_config_from_env

__all__ = [
    "MysqlConfig",
    "create_mysql_connection",
    "get_mysql_config_from_env",
]
```

- [ ] **Step 4: Add MySQL config and lazy connection helper**

Create `database/mysql_client.py`:

```python
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
    env = env if env is not None else os.environ

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
```

- [ ] **Step 5: Run tests to verify Task 1 passes**

Run:

```powershell
python -m unittest tests.test_user_repository
```

Expected:

```text
Ran 2 tests
OK
```

- [ ] **Step 6: Commit Task 1**

Run:

```powershell
git add database/__init__.py database/mysql_client.py tests/test_user_repository.py
git commit -m "feat: add mysql config helper"
```

---

### Task 2: User Repository With Validation And Age Bucketing

**Files:**
- Modify: `database/__init__.py`
- Create: `database/user_repository.py`
- Modify: `tests/test_user_repository.py`

- [ ] **Step 1: Add failing tests for repository behavior**

Append these tests to `tests/test_user_repository.py` above the `if __name__ == "__main__":` block:

```python
from database.user_repository import UserProfileRepository
from database.user_repository import normalize_age_to_movielens_bucket


class FakeCursor:
    def __init__(self, row=None, lastrowid=900001):
        self.row = row
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

    def test_create_user_rejects_invalid_profile(self):
        repository = UserProfileRepository(connection_factory=lambda: FakeConnection(FakeCursor()))

        with self.assertRaises(ValueError):
            repository.create_user(username="", age=25, occupation=4)

        with self.assertRaises(ValueError):
            repository.create_user(username="alice", age=0, occupation=4)

        with self.assertRaises(ValueError):
            repository.create_user(username="alice", age=25, occupation=21)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_user_repository
```

Expected:

```text
ImportError: No module named 'database.user_repository'
```

- [ ] **Step 3: Implement user repository**

Create `database/user_repository.py`:

```python
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
```

- [ ] **Step 4: Export repository from database package**

Modify `database/__init__.py`:

```python
from .mysql_client import MysqlConfig
from .mysql_client import create_mysql_connection
from .mysql_client import get_mysql_config_from_env
from .user_repository import CREATE_USERS_TABLE_SQL
from .user_repository import UserProfileRepository
from .user_repository import normalize_age_to_movielens_bucket

__all__ = [
    "CREATE_USERS_TABLE_SQL",
    "MysqlConfig",
    "UserProfileRepository",
    "create_mysql_connection",
    "get_mysql_config_from_env",
    "normalize_age_to_movielens_bucket",
]
```

- [ ] **Step 5: Run tests to verify Task 2 passes**

Run:

```powershell
python -m unittest tests.test_user_repository
```

Expected:

```text
Ran 7 tests
OK
```

- [ ] **Step 6: Commit Task 2**

Run:

```powershell
git add database/__init__.py database/user_repository.py tests/test_user_repository.py
git commit -m "feat: add user profile repository"
```

---

### Task 3: MySQL Schema And Registration Scripts

**Files:**
- Create: `scripts/init_mysql_schema.py`
- Create: `scripts/register_user.py`
- Modify: `tests/test_user_repository.py`

- [ ] **Step 1: Add import smoke tests for scripts**

Append these tests to `tests/test_user_repository.py` above the `if __name__ == "__main__":` block:

```python
class UserProfileScriptImportTest(unittest.TestCase):
    def test_schema_script_can_be_imported(self):
        from scripts.init_mysql_schema import main

        self.assertTrue(callable(main))

    def test_register_script_can_be_imported(self):
        from scripts.register_user import main

        self.assertTrue(callable(main))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_user_repository
```

Expected:

```text
ModuleNotFoundError: No module named 'scripts'
```

- [ ] **Step 3: Create schema init script**

Create `scripts/init_mysql_schema.py`:

```python
from database import UserProfileRepository


def main():
    repository = UserProfileRepository()
    repository.initialize_schema()
    print("MySQL users table is ready.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create registration script**

Create `scripts/register_user.py`:

```python
import argparse
import sys

from database import UserProfileRepository


def parse_args():
    parser = argparse.ArgumentParser(description="Register a user profile for cold-start recommendation.")
    parser.add_argument("--username", required=True)
    parser.add_argument("--age", required=True, type=int)
    parser.add_argument("--occupation", required=True, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    repository = UserProfileRepository()

    try:
        user_id = repository.create_user(
            username=args.username,
            age=args.age,
            occupation=args.occupation,
        )
    except Exception as error:
        print(f"Failed to register user: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(
        f"Registered user_id={user_id} "
        f"username={args.username} "
        f"age={args.age} "
        f"occupation={args.occupation}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify Task 3 passes**

Run:

```powershell
python -m unittest tests.test_user_repository
```

Expected:

```text
Ran 9 tests
OK
```

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
git add scripts/init_mysql_schema.py scripts/register_user.py tests/test_user_repository.py
git commit -m "feat: add mysql user registration scripts"
```

---

### Task 4: Pipeline Cold-Start Profile Lookup

**Files:**
- Modify: `recommender_pipeline.py`
- Modify: `tests/test_pipeline_timing.py`

- [ ] **Step 1: Add failing pipeline tests for repository profile lookup**

Modify the import and helper area in `tests/test_pipeline_timing.py`.

Add fake repositories after `FakeColdStartRecommender`:

```python
class FakeUserProfileRepository:
    def __init__(self, profile):
        self.profile = profile
        self.requested_user_ids = []

    def get_user_profile(self, user_id):
        self.requested_user_ids.append(user_id)
        return self.profile


class RaisingUserProfileRepository:
    def get_user_profile(self, user_id):
        raise RuntimeError("database unavailable")
```

Change `build_cold_start_pipeline` to accept a repository:

```python
    def build_cold_start_pipeline(self, user_profile_repository=None):
        pipeline = RecommenderPipeline.__new__(RecommenderPipeline)
        pipeline.recaller = FakeEmptyRecaller()
        pipeline.rough_ranker = RankerShouldNotRun()
        pipeline.fine_ranker = RankerShouldNotRun()
        pipeline.reranker = FakeReranker()
        pipeline.cold_start_recommender = FakeColdStartRecommender()
        pipeline.user_profile_repository = user_profile_repository
        return pipeline
```

Append tests above `if __name__ == "__main__":`:

```python
    def test_recommend_uses_repository_profile_for_cold_start(self):
        repository = FakeUserProfileRepository(
            profile={"user_id": 900001, "age": 35, "occupation": 7}
        )
        pipeline = self.build_cold_start_pipeline(user_profile_repository=repository)

        recommendations = pipeline.recommend(user_id=900001, top_k=2)

        self.assertEqual(repository.requested_user_ids, [900001])
        self.assertEqual(recommendations[0]["age"], 35)
        self.assertEqual(recommendations[0]["occupation"], 7)

    def test_recommend_prefers_explicit_profile_over_repository_profile(self):
        repository = FakeUserProfileRepository(
            profile={"user_id": 900001, "age": 35, "occupation": 7}
        )
        pipeline = self.build_cold_start_pipeline(user_profile_repository=repository)

        recommendations = pipeline.recommend(
            user_id=900001,
            age=25,
            occupation=4,
            top_k=2,
        )

        self.assertEqual(recommendations[0]["age"], 25)
        self.assertEqual(recommendations[0]["occupation"], 4)

    def test_recommend_keeps_cold_start_when_repository_returns_none(self):
        repository = FakeUserProfileRepository(profile=None)
        pipeline = self.build_cold_start_pipeline(user_profile_repository=repository)

        recommendations = pipeline.recommend(user_id=900001, top_k=2)

        self.assertEqual(len(recommendations), 2)
        self.assertIsNone(recommendations[0]["age"])
        self.assertIsNone(recommendations[0]["occupation"])

    def test_recommend_keeps_cold_start_when_repository_raises(self):
        pipeline = self.build_cold_start_pipeline(
            user_profile_repository=RaisingUserProfileRepository()
        )

        recommendations = pipeline.recommend(user_id=900001, top_k=2)

        self.assertEqual(len(recommendations), 2)
        self.assertIsNone(recommendations[0]["age"])
        self.assertIsNone(recommendations[0]["occupation"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_pipeline_timing
```

Expected:

```text
FAIL: test_recommend_uses_repository_profile_for_cold_start
```

The first failing assertion should show that `age` and `occupation` stayed `None`.

- [ ] **Step 3: Add repository injection and profile resolution to pipeline**

Modify `RecommenderPipeline.__init__` in `recommender_pipeline.py`:

```python
    def __init__(self, user_profile_repository=None):
        self.recaller = build_recaller()
        self.rough_ranker = build_rough_ranker()
        self.fine_ranker = build_fine_ranker()
        self.cold_start_recommender = build_cold_start_recommender()
        self.user_profile_repository = (
            user_profile_repository
            if user_profile_repository is not None
            else build_user_profile_repository()
        )
        self.reranker = Reranker()
```

Modify `cold_start`:

```python
    def cold_start(self, user_id, age=None, occupation=None, top_k=20):
        age, occupation = self.resolve_cold_start_profile(
            user_id=user_id,
            age=age,
            occupation=occupation,
        )
        return self.cold_start_recommender.recommend(
            user_id=user_id,
            age=age,
            occupation=occupation,
            top_k=top_k,
        )
```

Add this method under `cold_start`:

```python
    def resolve_cold_start_profile(self, user_id, age=None, occupation=None):
        resolved_age = age
        resolved_occupation = occupation

        if resolved_age is not None and resolved_occupation is not None:
            return resolved_age, resolved_occupation

        if self.user_profile_repository is None:
            return resolved_age, resolved_occupation

        try:
            profile = self.user_profile_repository.get_user_profile(user_id)
        except Exception as error:
            print(f"读取用户画像失败，继续使用冷启动兜底: {error}")
            return resolved_age, resolved_occupation

        if profile is None:
            return resolved_age, resolved_occupation

        if resolved_age is None:
            resolved_age = profile.get("age")

        if resolved_occupation is None:
            resolved_occupation = profile.get("occupation")

        return resolved_age, resolved_occupation
```

Add this builder near the other `build_*` functions:

```python
def build_user_profile_repository():
    from database.mysql_client import get_mysql_config_from_env

    if get_mysql_config_from_env() is None:
        return None

    from database import UserProfileRepository

    return UserProfileRepository()
```

- [ ] **Step 4: Run pipeline tests to verify Task 4 passes**

Run:

```powershell
python -m unittest tests.test_pipeline_timing
```

Expected:

```text
Ran 9 tests
OK
```

- [ ] **Step 5: Run all tests**

Run:

```powershell
python -m unittest discover -s tests
```

Expected:

```text
OK
```

- [ ] **Step 6: Commit Task 4**

Run:

```powershell
git add recommender_pipeline.py tests/test_pipeline_timing.py
git commit -m "feat: load cold start profile from repository"
```

---

### Task 5: README MySQL Usage Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add MySQL dependency and environment section**

Add a README section near the existing cold-start or command sections:

```markdown
## MySQL 用户画像接入

冷启动 V1 支持从 MySQL 读取新注册用户画像。当前阶段不引入复杂后端，只通过脚本模拟注册。

安装依赖：

```powershell
pip install pymysql
```

配置环境变量：

```powershell
$env:MYSQL_HOST="localhost"
$env:MYSQL_PORT="3306"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="your_password"
$env:MYSQL_DATABASE="recommend"
```

如果没有配置 `MYSQL_USER` 或 `MYSQL_PASSWORD`，推荐系统会跳过数据库读取，继续使用现有冷启动兜底逻辑。
```

- [ ] **Step 2: Add schema and registration commands**

Add this command block below the MySQL environment section:

```markdown
初始化用户表：

```powershell
python scripts/init_mysql_schema.py
```

模拟注册用户：

```powershell
python scripts/register_user.py --username alice --age 25 --occupation 4
```

脚本会输出新注册用户的 `user_id`。推荐时可以直接使用该 `user_id`：

```python
from recommender_pipeline import RecommenderPipeline

pipeline = RecommenderPipeline()
recommendations = pipeline.recommend(user_id=900001, top_k=20)
```

当该用户无法进入双塔召回时，pipeline 会自动从 MySQL 读取 `age` 和 `occupation`，再进入冷启动推荐。
```

- [ ] **Step 3: Review README for command consistency**

Run:

```powershell
rg -n "MySQL|register_user|init_mysql_schema|pymysql" README.md
```

Expected:

```text
README.md contains the dependency, environment variables, init script, and registration script.
```

- [ ] **Step 4: Commit Task 5**

Run:

```powershell
git add README.md
git commit -m "docs: document mysql user profile flow"
```

---

### Task 6: Manual Verification Commands

**Files:**
- No code changes expected.

- [ ] **Step 1: Run default unit tests**

Run:

```powershell
python -m unittest discover -s tests
```

Expected:

```text
OK
```

- [ ] **Step 2: Run pipeline without MySQL env**

Open a fresh PowerShell without MySQL environment variables, or temporarily remove them:

```powershell
Remove-Item Env:MYSQL_USER -ErrorAction SilentlyContinue
Remove-Item Env:MYSQL_PASSWORD -ErrorAction SilentlyContinue
python -m recommender_pipeline
```

Expected:

```text
The pipeline prints recommendations and does not fail because MySQL is unconfigured.
```

- [ ] **Step 3: Run MySQL integration flow manually**

Only run this step if local MySQL is available and the `recommend` database exists:

```powershell
$env:MYSQL_HOST="localhost"
$env:MYSQL_PORT="3306"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="your_password"
$env:MYSQL_DATABASE="recommend"

python scripts/init_mysql_schema.py
python scripts/register_user.py --username alice --age 25 --occupation 4
```

Expected:

```text
MySQL users table is ready.
Registered user_id=<new id> username=alice age=25 occupation=4
```

- [ ] **Step 4: Verify registered-user recommendation from Python**

Replace `<new id>` with the user ID printed by the registration script:

```powershell
python -c "from recommender_pipeline import RecommenderPipeline; p=RecommenderPipeline(); recs=p.recommend(user_id=<new id>, top_k=5); print(len(recs)); print(recs[0].get('recall_source'), recs[0].get('cold_start_source'))"
```

Expected:

```text
5
cold_start age_occupation
```

If the exact segment has too little data, the second line may be:

```text
cold_start age
```

or:

```text
cold_start occupation
```

This is acceptable because the repository lookup succeeded and cold-start fallback used the available profile.

---

### Task 7: Final Review And Cleanup

**Files:**
- Review: `database/mysql_client.py`
- Review: `database/user_repository.py`
- Review: `recommender_pipeline.py`
- Review: `scripts/init_mysql_schema.py`
- Review: `scripts/register_user.py`
- Review: `README.md`

- [ ] **Step 1: Check git status**

Run:

```powershell
git status --short
```

Expected:

```text
No uncommitted implementation changes, or only intentional files remain.
```

- [ ] **Step 2: Inspect recent commits**

Run:

```powershell
git log --oneline -5
```

Expected:

```text
Recent commits include mysql config helper, user repository, scripts, pipeline lookup, and README documentation.
```

- [ ] **Step 3: Run final test suite**

Run:

```powershell
python -m unittest discover -s tests
```

Expected:

```text
OK
```

- [ ] **Step 4: Summarize final behavior**

Use this summary in the final response:

```text
MySQL-backed registered-user profiles are now optional inputs to cold-start recommendation. New users can be inserted through scripts, pipeline can auto-fill age/occupation from MySQL, and missing/unavailable database state falls back to existing cold-start behavior without breaking old-user recommendations.
```

---

## Self-Review

Spec coverage:

- MySQL `users` table is covered by Task 2 and Task 3.
- Environment-variable configuration is covered by Task 1 and README in Task 5.
- Script-based registration is covered by Task 3.
- Pipeline automatic profile lookup is covered by Task 4.
- Database-unavailable fallback is covered by Task 4 tests.
- README usage commands are covered by Task 5.
- Manual MySQL verification is covered by Task 6.

Placeholder scan:

- This plan contains no banned placeholder markers.
- All task steps contain concrete files, commands, expected outputs, or code snippets.
- All new functions referenced in tests are defined in implementation steps.

Type consistency:

- Repository profile shape is consistently `{"user_id": int, "age": int, "occupation": int}`.
- Pipeline injection name is consistently `user_profile_repository`.
- MySQL helper names are consistently `MysqlConfig`, `get_mysql_config_from_env`, and `create_mysql_connection`.
