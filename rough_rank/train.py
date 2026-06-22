# 粗排模块训练入口
# 负责加载用户、电影、评分数据，构建特征，训练三塔粗排模型
# 训练完成后将模型权重和特征元信息保存到 models/rough_rank/ 目录

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .model import ThreeTowerRoughRankModel
from recall.two_tower import load_movies_from_dat, load_mysql_dataset_if_configured, load_ratings_from_dat, load_users_from_dat

# 项目根目录及各数据目录路径
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "train_data"
TEST_DIR = BASE_DIR / "test_data"
TRAIN_RATINGS_PATH = TRAIN_DIR / "ratings.dat"
TEST_RATINGS_PATH = TEST_DIR / "ratings.dat"
USERS_PATH = TRAIN_DIR / "users.dat"
MOVIES_PATH = TRAIN_DIR / "movies.dat"
MODEL_DIR = BASE_DIR / "models" / "rough_rank"
MODEL_PATH = MODEL_DIR / "three_tower.pt"

# 稠密特征维度：user_avg_rating, user_count, movie_avg_rating, movie_count, recall_score
DENSE_FEATURE_DIM = 5  # user_avg_rating, user_count, movie_avg_rating, movie_count, recall_score


# 返回可用的训练设备，优先使用 GPU
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 粗排训练数据集，封装样本列表，供 DataLoader 按批次读取
# samples: 由 load_samples 构建的字典列表，每条包含用户/电影特征和标签
class RoughRankDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples  # 原始样本列表

    # 返回样本总数
    def __len__(self):
        return len(self.samples)

    # 将第 idx 条样本的各字段转为 Tensor 并返回
    # 整数索引类特征用 long，浮点特征用 float，标签用 float（BCEWithLogitsLoss 要求）
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "user_index": torch.tensor(s["user_index"], dtype=torch.long),
            "gender_index": torch.tensor(s["gender_index"], dtype=torch.long),
            "age_index": torch.tensor(s["age_index"], dtype=torch.long),
            "occupation_index": torch.tensor(s["occupation_index"], dtype=torch.long),
            "movie_index": torch.tensor(s["movie_index"], dtype=torch.long),
            "genre_vector": torch.tensor(s["genre_vector"], dtype=torch.float),
            "dense_features": torch.tensor(s["dense_features"], dtype=torch.float),
            "label": torch.tensor(s["label"], dtype=torch.float),
        }


# 加载用户画像特征，并为性别、年龄、职业建立字符串到整数索引的映射
# users_path: users.dat 文件路径（当 users 为 None 时从文件读取）
# users: 可直接传入已加载的用户列表，避免重复 IO（MySQL 数据源场景使用）
# 返回：user_features 字典（uid -> 各索引）以及三个类别映射字典
def load_user_features(users_path=USERS_PATH, users=None):
    if users is None:
        users = load_users_from_dat(users_path)

    gender_to_index, age_to_index, occupation_to_index = {}, {}, {}
    user_features = {}

    for user in users:
        uid = int(user["user_id"])
        gender, age, occ = str(user["gender"]), str(user["age"]), str(user["occupation"])

        # 首次出现则分配新索引，保证索引连续且从 0 开始
        if gender not in gender_to_index:
            gender_to_index[gender] = len(gender_to_index)
        if age not in age_to_index:
            age_to_index[age] = len(age_to_index)
        if occ not in occupation_to_index:
            occupation_to_index[occ] = len(occupation_to_index)

        user_features[uid] = {
            "gender_index": gender_to_index[gender],
            "age_index": age_to_index[age],
            "occupation_index": occupation_to_index[occ],
        }

    return user_features, gender_to_index, age_to_index, occupation_to_index


# 加载电影特征，将每部电影的 genre 列表转为多热（multi-hot）向量
# movies_path: movies.dat 文件路径（当 movies 为 None 时从文件读取）
# movies: 可直接传入已加载的电影列表
# 返回：movie_features 字典（mid -> genre_vector）以及 genre 到索引的映射
def load_movie_features(movies_path=MOVIES_PATH, movies=None):
    if movies is None:
        movies = load_movies_from_dat(movies_path)

    genre_to_index, movie_genres = {}, {}
    for movie in movies:
        mid = int(movie["movie_id"])
        genre_list = list(movie["genres"])
        # 遍历所有 genre，动态扩充索引表
        for g in genre_list:
            if g not in genre_to_index:
                genre_to_index[g] = len(genre_to_index)
        movie_genres[mid] = genre_list

    genre_count = len(genre_to_index)
    movie_features = {}
    for mid, genre_list in movie_genres.items():
        # 构造长度为 genre_count 的 0/1 向量，有该类型则置 1
        vec = [0] * genre_count
        for g in genre_list:
            vec[genre_to_index[g]] = 1
        movie_features[mid] = {"genre_vector": vec}

    return movie_features, genre_to_index


# 统计训练集中每个用户和每部电影的评分均值与评分次数，用于构建稠密特征
# ratings_path: ratings.dat 文件路径
# ratings: 可直接传入已加载的评分列表
# 返回包含各统计量及归一化所需最大值的字典
def build_rating_stats(ratings_path=TRAIN_RATINGS_PATH, ratings=None):
    if ratings is None:
        ratings = load_ratings_from_dat(ratings_path)

    user_sum, user_count, movie_sum, movie_count = {}, {}, {}, {}
    for row in ratings:
        uid, mid, r = int(row["user_id"]), int(row["movie_id"]), int(row["rating"])
        user_sum[uid] = user_sum.get(uid, 0) + r
        user_count[uid] = user_count.get(uid, 0) + 1
        movie_sum[mid] = movie_sum.get(mid, 0) + r
        movie_count[mid] = movie_count.get(mid, 0) + 1

    return {
        "user_rating_sum": user_sum,
        "user_rating_count": user_count,
        "movie_rating_sum": movie_sum,
        "movie_rating_count": movie_count,
        "max_user_count": max(user_count.values()),    # 用于归一化用户评分次数
        "max_movie_count": max(movie_count.values()),  # 用于归一化电影评分次数
    }


# 为单条 (user_id, movie_id) 样本构建稠密特征向量（共 DENSE_FEATURE_DIM 维）
# user_id / movie_id: 原始 ID
# rating_stats: build_rating_stats 的返回值
# recall_score: 召回阶段给出的相似度分数，范围 [-1, 1]，默认 0（训练时无召回分）
# 所有特征归一化到 [0, 1]，方便模型稳定训练
def build_dense_features(user_id, movie_id, rating_stats, recall_score=0.0):
    u_count = rating_stats["user_rating_count"].get(user_id, 0)
    m_count = rating_stats["movie_rating_count"].get(movie_id, 0)
    # 冷启动用户/电影（无历史评分）默认均值取 3（中间值）
    u_avg = rating_stats["user_rating_sum"][user_id] / u_count if u_count > 0 else 3
    m_avg = rating_stats["movie_rating_sum"][movie_id] / m_count if m_count > 0 else 3
    return [
        u_avg / 5,                                      # 用户平均评分，归一化到 [0,1]
        u_count / rating_stats["max_user_count"],       # 用户活跃度，归一化到 [0,1]
        m_avg / 5,                                      # 电影平均评分，归一化到 [0,1]
        m_count / rating_stats["max_movie_count"],      # 电影热度，归一化到 [0,1]
        (float(recall_score) + 1.0) / 2.0,             # [-1,1] → [0,1]
    ]


# 汇总构建模型所需的全部特征元信息，包括各类别数量和 ID 到索引的映射
# 支持从 MySQL 或本地 .dat 文件加载数据（优先 MySQL）
# 评分为 3 的样本视为中性，跳过不参与训练，避免噪声标签
# 返回包含特征映射、统计量和维度信息的字典，供模型初始化和样本构建使用
def build_feature_info(users=None, movies=None, ratings=None):
    # 若未传入任何数据，优先尝试从 MySQL 加载训练集
    if users is None and movies is None and ratings is None:
        mysql_dataset = load_mysql_dataset_if_configured(split="train")
        if mysql_dataset is not None:
            users, movies, ratings = mysql_dataset["users"], mysql_dataset["movies"], mysql_dataset["ratings"]

    if ratings is None:
        ratings = load_ratings_from_dat(TRAIN_RATINGS_PATH)

    user_features, gender_to_index, age_to_index, occupation_to_index = load_user_features(users=users)
    movie_features, genre_to_index = load_movie_features(movies=movies)
    rating_stats = build_rating_stats(ratings=ratings)

    # 遍历评分记录，为出现过的用户和电影分配连续整数索引（供 Embedding 层使用）
    user_id_to_index, movie_id_to_index, index_to_movie_id = {}, {}, {}
    for row in ratings:
        uid, mid, r = int(row["user_id"]), int(row["movie_id"]), int(row["rating"])
        if r == 3:
            continue  # 跳过中性评分
        if uid not in user_id_to_index:
            user_id_to_index[uid] = len(user_id_to_index)
        if mid not in movie_id_to_index:
            idx = len(movie_id_to_index)
            movie_id_to_index[mid] = idx
            index_to_movie_id[idx] = mid  # 反向映射，推理时用于还原电影 ID

    return {
        "user_features": user_features,
        "movie_features": movie_features,
        "rating_stats": rating_stats,
        "user_id_to_index": user_id_to_index,
        "movie_id_to_index": movie_id_to_index,
        "index_to_movie_id": index_to_movie_id,
        "gender_count": len(gender_to_index),
        "age_count": len(age_to_index),
        "occupation_count": len(occupation_to_index),
        "genre_count": len(genre_to_index),
        "dense_feature_dim": DENSE_FEATURE_DIM,
    }


# 从评分文件（或传入的评分列表）构建训练/测试样本
# ratings_path: 数据文件路径，用于判断是训练集还是测试集（决定 MySQL split）
# feature_info: build_feature_info 的返回值，提供特征映射和统计量
# skip_unknown: 为 True 时跳过训练集中未出现的用户/电影（避免索引越界）
# ratings: 可直接传入已加载的评分列表
# 标签规则：评分 >= 4 为正样本（label=1），评分 <= 2 为负样本（label=0），评分 3 跳过
def load_samples(ratings_path, feature_info, skip_unknown=True, ratings=None):
    if ratings is None:
        # 根据路径判断加载训练集还是测试集
        split = "test" if ratings_path == TEST_RATINGS_PATH else "train"
        mysql_dataset = load_mysql_dataset_if_configured(split=split)
        ratings = mysql_dataset["ratings"] if mysql_dataset else load_ratings_from_dat(ratings_path)

    uf = feature_info["user_features"]
    mf = feature_info["movie_features"]
    rs = feature_info["rating_stats"]
    uid_to_idx = feature_info["user_id_to_index"]
    mid_to_idx = feature_info["movie_id_to_index"]

    samples = []
    for row in ratings:
        uid, mid, r = int(row["user_id"]), int(row["movie_id"]), int(row["rating"])
        if r == 3:
            continue  # 中性评分不参与训练
        # 测试集中可能出现训练集未见过的用户/电影，跳过以避免索引越界
        if skip_unknown and (uid not in uid_to_idx or mid not in mid_to_idx):
            continue
        samples.append({
            "user_index": uid_to_idx[uid],
            "gender_index": uf[uid]["gender_index"],
            "age_index": uf[uid]["age_index"],
            "occupation_index": uf[uid]["occupation_index"],
            "movie_index": mid_to_idx[mid],
            "genre_vector": mf[mid]["genre_vector"],
            "dense_features": build_dense_features(uid, mid, rs, recall_score=0.0),  # 训练时无召回分，默认 0
            "label": 1 if r >= 4 else 0,  # 二分类标签：喜欢 vs 不喜欢
        })
    return samples


# 将一个 batch 的所有 Tensor 移动到指定设备（CPU 或 GPU）
def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


# 根据 feature_info 中的各类别数量初始化三塔粗排模型并移动到目标设备
# feature_info: build_feature_info 的返回值，提供 Embedding 层所需的词表大小
def build_model(feature_info, device):
    fi = feature_info
    return ThreeTowerRoughRankModel(
        user_count=len(fi["user_id_to_index"]),
        movie_count=len(fi["movie_id_to_index"]),
        gender_count=fi["gender_count"],
        age_count=fi["age_count"],
        occupation_count=fi["occupation_count"],
        genre_count=fi["genre_count"],
        dense_feature_dim=fi["dense_feature_dim"],
    ).to(device)


# 执行一次前向传播，将 batch 中的各特征字段传入模型，返回原始 logit 分数
def run_batch(model, batch):
    return model(
        batch["user_index"], batch["gender_index"], batch["age_index"],
        batch["occupation_index"], batch["movie_index"], batch["genre_vector"], batch["dense_features"],
    )


# 在验证集上评估模型的损失和准确率
# 使用 torch.no_grad() 关闭梯度计算以节省显存和加速推理
# 评估完成后恢复 model.train() 状态，不影响后续训练
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = correct = total = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            score = run_batch(model, batch)
            total_loss += loss_fn(score, batch["label"]).item()
            # sigmoid 后以 0.5 为阈值转为二分类预测
            pred = (torch.sigmoid(score) >= 0.5).float()
            correct += (pred == batch["label"]).sum().item()
            total += batch["label"].shape[0]

    model.train()
    # 返回平均 loss 和整体准确率
    return total_loss / len(dataloader), correct / total


# 完整的训练流程：加载数据 → 构建模型 → 逐 epoch 训练 → 评估 → 保存权重
# epochs: 训练轮数
# batch_size: 每批样本数，影响显存占用和梯度更新频率
# learning_rate: Adam 优化器学习率
# 每个 epoch 结束后保存一份带 epoch 编号的检查点，方便回滚
# 全部训练完成后额外保存一份 three_tower.pt 作为最终推理模型
def train_model(epochs=3, batch_size=1024, learning_rate=0.001):
    device = get_device()
    print(f"当前训练设备: {device}")

    # 构建特征元信息，同时用于初始化模型和构建样本
    feature_info = build_feature_info()
    train_loader = DataLoader(RoughRankDataset(load_samples(TRAIN_RATINGS_PATH, feature_info)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(RoughRankDataset(load_samples(TEST_RATINGS_PATH, feature_info)), batch_size=batch_size, shuffle=False)

    model = build_model(feature_info, device)
    loss_fn = nn.BCEWithLogitsLoss()  # 二分类交叉熵，内置 sigmoid，数值更稳定
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    MODEL_DIR.mkdir(exist_ok=True)  # 确保模型保存目录存在

    for epoch in range(epochs):
        train_loss = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            score = run_batch(model, batch)
            loss = loss_fn(score, batch["label"])
            optimizer.zero_grad()   # 清空上一步梯度
            loss.backward()         # 反向传播计算梯度
            optimizer.step()        # 更新参数
            train_loss += loss.item()

        # 计算该 epoch 的平均训练损失
        train_loss /= len(train_loader)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} test_loss={test_loss:.4f} test_accuracy={test_acc:.4f}")

        # 保存当前 epoch 的检查点，包含模型权重、特征元信息和训练指标
        torch.save({"model_state_dict": model.state_dict(), "feature_info": feature_info,
                    "epoch": epoch + 1, "train_loss": train_loss, "test_loss": test_loss, "test_accuracy": test_acc},
                   MODEL_DIR / f"three_tower_epoch_{epoch + 1}.pt")

    # 保存最终模型，供推理模块直接加载
    torch.save({"model_state_dict": model.state_dict(), "feature_info": feature_info, "epoch": epochs}, MODEL_PATH)
    print(f"模型已保存到: {MODEL_PATH}")


# 命令行入口，解析超参数后调用 train_model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
