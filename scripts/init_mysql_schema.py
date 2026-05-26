from database import MysqlDatasetRepository


def main():
    repository = MysqlDatasetRepository()
    repository.initialize_schema()
    print("MySQL tables are ready.")


if __name__ == "__main__":
    main()
