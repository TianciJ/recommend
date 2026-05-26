# MySQL Data Source

The project can run from MySQL instead of depending on MovieLens `.dat` files at
runtime or retraining time. The `.dat` files are only needed as a one-time import
source when bootstrapping a local database.

## Install

```powershell
pip install pymysql
```

## Configure

```powershell
$env:MYSQL_HOST="localhost"
$env:MYSQL_PORT="3306"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="your_password"
$env:MYSQL_DATABASE="recommend"
```

## Initialize Tables

```powershell
python scripts/init_mysql_schema.py
```

## Import Existing MovieLens Data

```powershell
python scripts/import_movielens_to_mysql.py --train-dir train_data --test-dir test_data
```

## Register A New User

```powershell
python scripts/register_user.py --username alice --age 25 --occupation 4
```

## Run From MySQL

After MySQL is configured and imported, these commands read users, movies, and
ratings from MySQL first:

```powershell
python -m recommender_pipeline
python -m recall.two_tower --mode train
python -m rough_rank.train_rough_rank
python -m fine_rank.train_mmoe_ranker
python -m recall.evaluate
```

If MySQL variables are not configured, the code keeps the old `.dat` fallback for
local compatibility.
