from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MOVIES_PATH = BASE_DIR / "train_data" / "movies.dat"


def load_movie_titles(movies_path=MOVIES_PATH, movies=None):
    # 保存 movie_id 和电影标题的对应关系，格式是 movie_id: title
    movie_titles = {}

    if movies is None:
        movies = load_mysql_movies_if_configured()

    if movies is not None:
        for movie in movies:
            movie_titles[int(movie["movie_id"])] = movie["title"]
        return movie_titles

    with movies_path.open("r", encoding="latin-1") as movies_file:
        for line in movies_file:
            # 每行格式是 MovieID::Title::Genres，这里只需要前两个字段
            movie_id, title, genres = line.strip().split("::")
            movie_titles[int(movie_id)] = title

    return movie_titles


def add_movie_titles(recommendations, movies_path=MOVIES_PATH, movies=None):
    # 给召回结果补上电影标题，方便查看
    movie_titles = load_movie_titles(movies_path, movies=movies)

    for item in recommendations:
        item["title"] = movie_titles.get(item["movie_id"], "")

    return recommendations


def load_mysql_movies_if_configured():
    from database.dataset_repository import load_mysql_dataset

    dataset = load_mysql_dataset(split="train")

    if dataset is None:
        return None

    return dataset["movies"]


def print_recommendations(recommendations):
    # 按固定格式打印推荐结果
    for rank, item in enumerate(recommendations, start=1):
        print(
            f"{rank}. movie_id={item['movie_id']} "
            f"score={item['score']:.4f} "
            f"title={item.get('title', '')}"
        )
