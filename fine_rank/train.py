# 精排（Fine Rank）训练入口
# 职责：在召回、粗排之后对候选集做精细排序，同时优化三个任务（喜欢/高分/评分回归）
# 流程：构建特征信息 → 加载样本 → 用召回/粗排模型打分 → 训练 MMoE 精排模型 → 保存 checkpoint
import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .model import MMoERanker, evaluate, get_device, train_one_epoch
from recall.two_tower import build_model_from_checkpoint as build_recall_model
from recall.two_tower import load_movies_from_dat, load_mysql_dataset_if_configured, load_ratings_from_dat, load_users_from_dat
from rough_rank.inference import build_dense_features, build_model_from_checkpoint as build_rough_model
from utils import load_checkpoint

# ---------- 路径常量 ----------
BASE_DIR = Path(__file__).resolve().parent.parent   # 项目根目录
TRAIN_DIR = BASE_DIR / "train_data"                 # 训练数据目录
TEST_DIR = BASE_DIR / "test_data"                   # 测试数据目录
TRAIN_RATINGS_PATH = TRAIN_DIR / "ratings.dat"      # 训练集评分文件
TEST_RATINGS_PATH = TEST_DIR / "ratings.dat"        # 测试集评分文件
USERS_PATH = TRAIN_DIR / "users.dat"                # 用户属性文件
MOVIES_PATH = TRAIN_DIR / "movies.dat"              # 电影属性文件

# 依赖的上游模型路径：召回双塔、粗排三塔
RECALL_MODEL_PATH = BASE_DIR / "models" / "recall" / "two_tower.pt"
ROUGH_MODEL_PATH = BASE_DIR / "models" / "rough_rank" / "three_tower.pt"
# 精排模型输出目录及最终模型路径
FINE_RANK_MODEL_DIR = BASE_DIR / "models" / "fine_rank"
FINE_RANK_MODEL_PATH = FINE_RANK_MODEL_DIR / "mmoe.pt"


# ---------- 数据集 ----------

class MMoEDataset(Dataset):
    # 将样本列表包装为 PyTorch Dataset，供 DataLoader 批量读取
    # samples: list[dict]，每条包含用户/电影特征、上游打分和三个标签
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # 将每个字段转为 tensor，类型与模型 Embedding/损失函数要求对齐
        return {
            "user_id":           torch.tensor(s["user_id"],           dtype=torch.long),
            "gender":            torch.tensor(s["gender"],            dtype=torch.long),
            "age":               torch.tensor(s["age"],               dtype=torch.long),
            "occupation":        torch.tensor(s["occupation"],        dtype=torch.long),
            "movie_id":          torch.tensor(s["movie_id"],          dtype=torch.long),
            "genres":            torch.tensor(s["genres"],            dtype=torch.long),
            "recall_score":      torch.tensor(s["recall_score"],      dtype=torch.float),  # 召回模型输出的相似度分
            "coarse_score":      torch.tensor(s["coarse_score"],      dtype=torch.float),  # 粗排模型输出的排序分
            "like_label":        torch.tensor(s["like_label"],        dtype=torch.float),  # 是否喜欢（rating >= 4）
            "high_rating_label": torch.tensor(s["high_rating_label"], dtype=torch.float),  # 是否高分（rating == 5）
            "rating_label":      torch.tensor(s["rating_label"],      dtype=torch.float),  # 归一化评分（rating / 5）
        }


# ---------- 特征构建 ----------

def load_user_features(users_path=USERS_PATH, users=None):
    # 将用户原始属性（性别/年龄/职业）映射为连续整数索引，供 Embedding 层使用
    # users: 可直接传入已加载的用户列表，避免重复读文件（MySQL 数据源场景）
    # 返回：user_features（uid -> 各属性索引），以及三个属性的 str->int 映射表
    if users is None:
        users = load_users_from_dat(users_path)

    gender_to_index, age_to_index, occupation_to_index = {}, {}, {}
    user_features = {}

    for user in users:
        uid = int(user["user_id"])
        gender, age, occ = str(user["gender"]), str(user["age"]), str(user["occupation"])

        # 首次出现时分配新索引，保证索引从 0 开始连续递增
        if gender not in gender_to_index:
            gender_to_index[gender] = len(gender_to_index)
        if age not in age_to_index:
            age_to_index[age] = len(age_to_index)
        if occ not in occupation_to_index:
            occupation_to_index[occ] = len(occupation_to_index)

        user_features[uid] = {
            "gender":     gender_to_index[gender],
            "age":        age_to_index[age],
            "occupation": occupation_to_index[occ],
        }

    return user_features, gender_to_index, age_to_index, occupation_to_index


def load_movie_features(movies_path=MOVIES_PATH, movies=None):
    # 将电影的多标签 genre 转为 padding 对齐的索引列表，供 Embedding + attention pooling 使用
    # 用索引序列而非 one-hot 向量，是因为精排模型用 genre Embedding 做 attention pooling
    # 返回：movie_features（mid -> genres 索引列表），genre 映射表，以及 genres 最大长度
    if movies is None:
        movies = load_movies_from_dat(movies_path)

    genre_to_index, movie_features = {}, {}
    max_genre_length = 0

    for movie in movies:
        mid = int(movie["movie_id"])
        idxs = []
        for g in movie["genres"]:
            if g not in genre_to_index:
                genre_to_index[g] = len(genre_to_index) + 1  # 0 留给 padding，实际索引从 1 开始
            idxs.append(genre_to_index[g])
        max_genre_length = max(max_genre_length, len(idxs))  # 记录最长 genre 列表，用于后续补 padding
        movie_features[mid] = {"genres": idxs}

    # 统一补 0（padding）到相同长度，使 DataLoader 能正确 batch
    for mid in movie_features:
        g = movie_features[mid]["genres"]
        movie_features[mid]["genres"] = g + [0] * (max_genre_length - len(g))

    return movie_features, genre_to_index, max_genre_length


def build_feature_info(users=None, movies=None, ratings=None):
    # 汇总所有特征元信息，供模型构建和样本加载共享
    # 优先使用外部传入数据（MySQL 数据源），否则回退到本地 .dat 文件
    # 返回一个包含特征映射表、Embedding 词表大小等信息的字典，作为 checkpoint 的一部分保存
    if users is None and movies is None and ratings is None:
        mysql_dataset = load_mysql_dataset_if_configured(split="train")
        if mysql_dataset is not None:
            users, movies, ratings = mysql_dataset["users"], mysql_dataset["movies"], mysql_dataset["ratings"]

    if ratings is None:
        ratings = load_ratings_from_dat(TRAIN_RATINGS_PATH)

    user_features, gender_to_index, age_to_index, occupation_to_index = load_user_features(users=users)
    movie_features, genre_to_index, max_genre_length = load_movie_features(movies=movies)

    # 从评分记录中收集出现过的 user/movie，建立原始 id -> 连续整数索引 的双向映射
    user_id_to_index, movie_id_to_index, index_to_movie_id = {}, {}, {}
    for row in ratings:
        uid, mid = int(row["user_id"]), int(row["movie_id"])
        if uid not in user_id_to_index:
            user_id_to_index[uid] = len(user_id_to_index)
        if mid not in movie_id_to_index:
            idx = len(movie_id_to_index)
            movie_id_to_index[mid] = idx
            index_to_movie_id[idx] = mid  # 反向映射，推理时还原真实 movie_id

    return {
        "user_features":     user_features,
        "movie_features":    movie_features,
        "user_id_to_index":  user_id_to_index,
        "movie_id_to_index": movie_id_to_index,
        "index_to_movie_id": index_to_movie_id,
        "gender_count":      len(gender_to_index),
        "age_count":         len(age_to_index),
        "occupation_count":  len(occupation_to_index),
        "genre_count":       len(genre_to_index) + 1,  # +1 是因为 0 被保留为 padding
        "max_genre_length":  max_genre_length,
    }


def load_base_samples(ratings_path, feature_info, skip_unknown=True, ratings=None):
    # 从评分文件加载样本，构造三个监督标签（like / high_rating / rating 回归）
    # skip_unknown=True：跳过训练集中未见过的 user/movie，避免越界 Embedding 查找
    # 此时 recall_score / coarse_score 尚未填充，由 attach_model_scores 后续补充
    if ratings is None:
        split = "test" if ratings_path == TEST_RATINGS_PATH else "train"
        mysql_dataset = load_mysql_dataset_if_configured(split=split)
        ratings = mysql_dataset["ratings"] if mysql_dataset else load_ratings_from_dat(ratings_path)

    uf = feature_info["user_features"]
    mf = feature_info["movie_features"]
    uid_to_idx = feature_info["user_id_to_index"]
    mid_to_idx = feature_info["movie_id_to_index"]

    samples = []
    for row in ratings:
        uid, mid, r = int(row["user_id"]), int(row["movie_id"]), int(row["rating"])
        if skip_unknown and (uid not in uid_to_idx or mid not in mid_to_idx):
            continue
        samples.append({
            "raw_user_id":       uid,          # 保留原始 id，供上游打分器查找
            "raw_movie_id":      mid,
            "user_id":           uid_to_idx[uid],
            "gender":            uf[uid]["gender"],
            "age":               uf[uid]["age"],
            "occupation":        uf[uid]["occupation"],
            "movie_id":          mid_to_idx[mid],
            "genres":            mf[mid]["genres"],
            "like_label":        1 if r >= 4 else 0,   # 二分类：喜欢/不喜欢
            "high_rating_label": 1 if r == 5 else 0,   # 二分类：满分/非满分
            "rating_label":      r / 5,                # 回归目标，归一化到 [0,1]
        })
    return samples


# ---------- 上游模型打分 ----------

def _collect_scorable_rows(batch_samples, fi):
    """从 batch 中找出 user/movie 均在上游模型训练集内的样本，返回行数据和对应下标。
    不可打分的样本保留位置（用默认分 0.0 填充），避免输出长度与输入不一致。"""
    uid_to_idx = fi["user_id_to_index"]
    mid_to_idx = fi["movie_id_to_index"]
    uf = fi["user_features"]
    mf = fi["movie_features"]
    rows, positions = [], []
    for pos, s in enumerate(batch_samples):
        uid, mid = s["raw_user_id"], s["raw_movie_id"]
        if uid in uid_to_idx and mid in mid_to_idx:
            rows.append((uid, mid, uf[uid], mf[mid]))
            positions.append(pos)
    return rows, positions


def _score_all_samples(model, fi, device, build_inputs_fn, samples, batch_size):
    """分批对所有样本打分，不可打分的位置填 0.0。
    分批是为了避免一次性把所有样本塞入 GPU 导致 OOM。"""
    scores = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            rows, positions = _collect_scorable_rows(chunk, fi)
            batch_scores = [0.0] * len(chunk)
            if rows:
                outputs = model(*build_inputs_fn(rows, fi, device)).cpu().tolist()
                for pos, score in zip(positions, outputs):
                    batch_scores[pos] = score
            scores.extend(batch_scores)
    return scores


class RecallScorer:
    """封装召回双塔模型，给精排样本附加 recall_score（余弦相似度）。"""

    def __init__(self, device):
        cp = load_checkpoint(RECALL_MODEL_PATH, device)
        self.fi = cp["feature_info"]
        self.model = build_recall_model(cp, device)
        self.device = device

    def score_samples(self, samples, batch_size):
        return _score_all_samples(self.model, self.fi, self.device, self._build_inputs, samples, batch_size)

    def _build_inputs(self, rows, fi, device):
        # avg_rating 和 rating_count 做归一化，使数值范围与 Embedding 输出对齐
        uid_to_idx, mid_to_idx = fi["user_id_to_index"], fi["movie_id_to_index"]
        max_count = fi.get("max_rating_count", 1) or 1  # 防止除以 0
        t = lambda vals, dtype: torch.tensor(vals, dtype=dtype, device=device)
        return (
            t([uid_to_idx[r[0]] for r in rows], torch.long),
            t([r[2]["gender_index"] for r in rows], torch.long),
            t([r[2]["age_index"] for r in rows], torch.long),
            t([r[2]["occupation_index"] for r in rows], torch.long),
            t([mid_to_idx[r[1]] for r in rows], torch.long),
            t([r[3]["genre_vector"] for r in rows], torch.float),
            # 用户行为特征：平均评分（归一化）+ log 评分次数（归一化），反映活跃度和口味稳定性
            t([[r[2]["avg_rating"] / 5.0, math.log1p(r[2]["rating_count"]) / math.log1p(max_count)] for r in rows], torch.float),
        )


class CoarseScorer:
    """封装粗排三塔模型，给精排样本附加 coarse_score。
    粗排分作为精排的输入特征之一，帮助模型感知上游排序信号。"""

    def __init__(self, device):
        cp = load_checkpoint(ROUGH_MODEL_PATH, device)
        self.fi = cp["feature_info"]
        self.model = build_rough_model(cp, device)
        self.device = device

    def score_samples(self, samples, batch_size):
        return _score_all_samples(self.model, self.fi, self.device, self._build_inputs, samples, batch_size)

    def _build_inputs(self, rows, fi, device):
        uid_to_idx, mid_to_idx = fi["user_id_to_index"], fi["movie_id_to_index"]
        rs = fi["rating_stats"]
        t = lambda vals, dtype: torch.tensor(vals, dtype=dtype, device=device)
        return (
            t([uid_to_idx[r[0]] for r in rows], torch.long),
            t([r[2]["gender_index"] for r in rows], torch.long),
            t([r[2]["age_index"] for r in rows], torch.long),
            t([r[2]["occupation_index"] for r in rows], torch.long),
            t([mid_to_idx[r[1]] for r in rows], torch.long),
            t([r[3]["genre_vector"] for r in rows], torch.float),
            t([build_dense_features(r[0], r[1], rs) for r in rows], torch.float),  # 5 维统计特征
        )


def attach_model_scores(samples, recall_scorer, coarse_scorer, batch_size):
    # 原地为每条样本补充 recall_score 和 coarse_score
    # 两个打分器独立调用，互不依赖，分开批量推理后再合并写回
    recall_scores = recall_scorer.score_samples(samples, batch_size)
    coarse_scores = coarse_scorer.score_samples(samples, batch_size)
    for s, rs, cs in zip(samples, recall_scores, coarse_scores):
        s["recall_score"] = rs
        s["coarse_score"] = cs


# ---------- 模型构建与保存 ----------

def build_model(feature_info, device):
    # 根据 feature_info 中的词表大小初始化 MMoERanker，确保 Embedding 维度与数据匹配
    fi = feature_info
    return MMoERanker(
        user_count=len(fi["user_id_to_index"]),
        movie_count=len(fi["movie_id_to_index"]),
        gender_count=fi["gender_count"],
        age_count=fi["age_count"],
        occupation_count=fi["occupation_count"],
        genre_count=fi["genre_count"],
    ).to(device)


def save_checkpoint(model, feature_info, model_path, extra_info=None):
    # 保存模型权重和 feature_info 到同一个 checkpoint，推理时无需重新构建特征映射
    # extra_info 用于记录 epoch、训练指标等调试信息，不影响模型加载
    FINE_RANK_MODEL_DIR.mkdir(exist_ok=True)
    cp = {"model_state_dict": model.state_dict(), "feature_info": feature_info}
    if extra_info is not None:
        cp["extra_info"] = extra_info
    torch.save(cp, model_path)


# ---------- 训练主流程 ----------

def train_model(epochs=3, batch_size=1024, score_batch_size=4096, learning_rate=0.001):
    # 完整训练流程：特征构建 → 样本加载 → 上游打分 → 模型训练 → 逐 epoch 保存
    # score_batch_size 通常远大于 batch_size，因为打分阶段不需要反向传播，显存占用更小
    device = get_device()
    print(f"当前训练设备: {device}")

    # 构建特征元信息（词表、映射表），训练集和测试集共享同一份，避免特征不一致
    feature_info = build_feature_info()
    train_samples = load_base_samples(TRAIN_RATINGS_PATH, feature_info)
    test_samples = load_base_samples(TEST_RATINGS_PATH, feature_info)

    # 加载上游模型，准备打分
    recall_scorer = RecallScorer(device)
    coarse_scorer = CoarseScorer(device)

    print("正在生成训练集 recall_score 和 coarse_score...")
    attach_model_scores(train_samples, recall_scorer, coarse_scorer, score_batch_size)
    print("正在生成测试集 recall_score 和 coarse_score...")
    attach_model_scores(test_samples, recall_scorer, coarse_scorer, score_batch_size)

    # 训练集 shuffle，测试集不 shuffle，保证评估结果可复现
    train_loader = DataLoader(MMoEDataset(train_samples), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MMoEDataset(test_samples), batch_size=batch_size, shuffle=False)

    model = build_model(feature_info, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # train_one_epoch / evaluate 均定义在 model.py，返回包含各任务指标的字典
        tm = train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, device=device)
        vm = evaluate(model=model, dataloader=test_loader, device=device)
        print(f"epoch={epoch + 1} train_loss={tm['total_loss']:.4f} test_loss={vm['total_loss']:.4f} "
              f"test_like_acc={vm['like_accuracy']:.4f} test_high_acc={vm['high_rating_accuracy']:.4f}")

        # 每个 epoch 单独保存一份，方便后续选取最优 checkpoint
        save_checkpoint(model, feature_info, FINE_RANK_MODEL_DIR / f"mmoe_epoch_{epoch + 1}.pt",
                        extra_info={"epoch": epoch + 1, "train_metrics": tm, "test_metrics": vm})

    # 最后一个 epoch 的权重额外保存为固定路径，供推理模块直接加载
    save_checkpoint(model, feature_info, FINE_RANK_MODEL_PATH, extra_info={"epoch": epochs})
    print(f"模型已保存到: {FINE_RANK_MODEL_PATH}")


# ---------- 命令行入口 ----------

def main():
    # 解析命令行参数，所有超参数均有默认值，可按需覆盖
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",           type=int,   default=3)
    parser.add_argument("--batch-size",       type=int,   default=1024)
    parser.add_argument("--score-batch-size", type=int,   default=4096)
    parser.add_argument("--learning-rate",    type=float, default=0.001)
    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size, score_batch_size=args.score_batch_size, learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
