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
