import argparse
from functools import cmp_to_key
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RATINGS_PATH = BASE_DIR / "data" / "ratings.dat"
MOVIES_PATH = BASE_DIR / "data" / "movies.dat"


def load_movie_titles(movies_path=MOVIES_PATH):
    # 保存 movie_id 和电影标题的对应关系，格式是 movie_id: title
    movie_titles = {}

    with movies_path.open("r", encoding="latin-1") as movies_file:
        for line in movies_file:
            # 每行格式是 MovieID::Title::Genres，这里只需要前两个字段
            movie_id, title, genres = line.strip().split("::")

            # 取出来的 movie_id 是字符串，转成整数后保存到字典里
            movie_titles[int(movie_id)] = title

    return movie_titles


def compare_movie(a, b):
    # 如果 a 的平均分更高，a 应该排在 b 前面
    if a["average_rating"] > b["average_rating"]:
        return -1

    # 如果 a 的平均分更低，a 应该排在 b 后面
    if a["average_rating"] < b["average_rating"]:
        return 1

    # 走到这里说明平均分一样，再比较评分人数
    if a["rating_count"] > b["rating_count"]:
        return -1

    if a["rating_count"] < b["rating_count"]:
        return 1

    # 平均分和评分人数都一样，就认为它们排序优先级相同
    return 0


def calculate_average_ratings(ratings_path=RATINGS_PATH):
    # 记录每部电影收到的评分总和
    rating_sum = {}

    # 记录每部电影被评分了多少次
    rating_count = {}

    with ratings_path.open("r", encoding="utf-8") as ratings_file:
        for line in ratings_file:
            # 每行格式是 UserID::MovieID::Rating::Timestamp
            user_id, movie_id, rating, timestamp = line.strip().split("::")

            # 把字符串转成后面方便计算的数字
            movie_id = int(movie_id)
            rating = float(rating)

            # 如果这部电影第一次出现，先给它一个初始值
            if movie_id not in rating_sum:
                rating_sum[movie_id] = 0
                rating_count[movie_id] = 0

            # 把评分加到这部电影的评分总和里
            rating_sum[movie_id] += rating

            # 这部电影的评分次数加 1
            rating_count[movie_id] += 1

    # 把每部电影的平均分算出来，放进列表
    average_ratings = []
    for movie_id in rating_sum:
        average_rating = rating_sum[movie_id] / rating_count[movie_id]
        average_ratings.append(
            {
                "movie_id": movie_id,
                "average_rating": average_rating,
                "rating_count": rating_count[movie_id],
            }
        )

    # 按平均分从高到低排序；平均分一样时，评分人数多的排前面
    average_ratings.sort(key=cmp_to_key(compare_movie))

    return average_ratings


def recall_by_average_rating(top_k=20, include_title=True):
    # 先计算所有电影的平均分，并取前 top_k 个
    recommendations = calculate_average_ratings()[:top_k]

    # 如果不需要电影标题，直接返回 movie_id、平均分、评分人数
    if not include_title:
        return recommendations

    # 读取电影标题，方便看召回结果
    movie_titles = load_movie_titles()

    # 给每条召回结果补上电影标题
    for item in recommendations:
        item["title"] = movie_titles.get(item["movie_id"], "")

    return recommendations


def main():
    # 解析命令行参数，比如：python recall.py --top-k 10
    parser = argparse.ArgumentParser(description="按电影平均评分做简单召回")
    parser.add_argument("--top-k", type=int, default=20, help="返回的电影数量")
    args = parser.parse_args()

    # 调用召回函数，拿到结果
    recommendations = recall_by_average_rating(top_k=args.top_k)

    # 把结果打印出来，方便直接在终端查看
    for rank, item in enumerate(recommendations, start=1):
        print(
            f"{rank}. movie_id={item['movie_id']} "
            f"avg={item['average_rating']:.4f} "
            f"count={item['rating_count']} "
            f"title={item['title']}"
        )


if __name__ == "__main__":
    main()
