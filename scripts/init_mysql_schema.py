try:
    from scripts.bootstrap import add_project_root_to_path
except ModuleNotFoundError:
    from bootstrap import add_project_root_to_path


add_project_root_to_path()

from database import MysqlDatasetRepository


def main():
    repository = MysqlDatasetRepository()
    repository.initialize_schema()
    print("MySQL tables are ready.")


if __name__ == "__main__":
    main()
