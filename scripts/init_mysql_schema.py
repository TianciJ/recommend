from database import UserProfileRepository


def main():
    repository = UserProfileRepository()
    repository.initialize_schema()
    print("MySQL users table is ready.")


if __name__ == "__main__":
    main()
