# MMoE 精排模型
# Multi-gate Mixture of Experts，同时优化三个任务：
#   like        - 用户是否喜欢（rating >= 4），二分类
#   high_rating - 是否给出高分（rating == 5），二分类
#   rating      - 归一化评分预测（rating / 5），回归
# 三个任务共享 4 个 Expert，每个任务有独立的 Gate 对 Expert 输出加权
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_device, move_batch_to_device


# ---------- 子模块 ----------

class Expert(nn.Module):
    """单个 Expert：将 input_dim 维输入压缩到 expert_dim 维表示。"""
    def __init__(self, input_dim, expert_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, expert_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class TaskTower(nn.Module):
    """单任务输出头：将 expert_dim 维表示映射到标量 logit。"""
    def __init__(self, expert_dim=64):
        super().__init__()
        self.tower = nn.Sequential(
            nn.Linear(expert_dim, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.tower(x).squeeze(1)


# ---------- 主模型 ----------

class MMoERanker(nn.Module):
    def __init__(self, user_count, movie_count, gender_count, age_count,
                 occupation_count, genre_count, num_experts=4, expert_dim=64):
        super().__init__()

        self.num_experts = num_experts
        self.expert_dim = expert_dim

        # --- 特征 Embedding ---
        self.user_embedding = nn.Embedding(user_count, 32)
        self.movie_embedding = nn.Embedding(movie_count, 32)
        self.gender_embedding = nn.Embedding(gender_count, 4)
        self.age_embedding = nn.Embedding(age_count, 8)
        self.occupation_embedding = nn.Embedding(occupation_count, 8)
        # genre 为多标签特征，0 留给 padding，padding_idx=0 保证 padding 位梯度不更新
        self.genre_embedding = nn.Embedding(genre_count, 16, padding_idx=0)
        self.genre_attention = nn.Linear(16, 1)  # 用于 attention pooling

        # 输入向量总维度: 32+32+4+8+8+16+1+1 = 102
        self.input_dim = 102

        # --- MMoE 层 ---
        # 4 个独立 Expert，每个都接收完整的 input_dim 向量
        self.experts = nn.ModuleList([
            Expert(input_dim=self.input_dim, expert_dim=expert_dim)
            for _ in range(num_experts)
        ])

        # 每个任务有独立的 Gate，学习对 Expert 的加权组合
        self.like_gate = nn.Linear(self.input_dim, num_experts)
        self.high_rating_gate = nn.Linear(self.input_dim, num_experts)
        self.rating_gate = nn.Linear(self.input_dim, num_experts)

        # 每个任务的输出头
        self.like_tower = TaskTower(expert_dim=expert_dim)
        self.high_rating_tower = TaskTower(expert_dim=expert_dim)
        self.rating_tower = TaskTower(expert_dim=expert_dim)

    def pool_genres(self, genres):
        # Attention Pooling：对多个 genre embedding 加权求和，比 mean pooling 更灵活
        genre_vectors = self.genre_embedding(genres)          # [batch, max_genres, 16]
        mask = (genres != 0).float().unsqueeze(-1)             # [batch, max_genres, 1]，padding 位为 0

        attention_scores = self.genre_attention(genre_vectors)            # [batch, max_genres, 1]
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)  # padding 位填极小值
        attention_weights = torch.softmax(attention_scores, dim=1)        # softmax 后 padding 权重趋近 0

        return (genre_vectors * attention_weights * mask).sum(dim=1)      # [batch, 16]

    def build_input(self, user_id, gender, age, occupation, movie_id, genres, recall_score, coarse_score):
        # 拼接所有特征，构建统一的 102 维输入向量
        recall_score = recall_score.float().view(-1, 1)
        coarse_score = coarse_score.float().view(-1, 1)
        return torch.cat([
            self.user_embedding(user_id),
            self.movie_embedding(movie_id),
            self.gender_embedding(gender),
            self.age_embedding(age),
            self.occupation_embedding(occupation),
            self.pool_genres(genres),
            recall_score,   # 来自双塔召回
            coarse_score,   # 来自三塔粗排
        ], dim=1)

    def apply_gate(self, expert_outputs, gate_weights):
        # 用 Gate 权重对所有 Expert 输出加权求和，得到该任务的表示向量
        # expert_outputs: [batch, num_experts, expert_dim]
        # gate_weights:   [batch, num_experts]
        return (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, user_id, gender, age, occupation, movie_id, genres, recall_score, coarse_score):
        x = self.build_input(user_id, gender, age, occupation, movie_id, genres, recall_score, coarse_score)

        # 所有 Expert 并行处理同一输入，堆叠成 [batch, num_experts, expert_dim]
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)

        # 每个任务用各自的 Gate 对 Expert 输出加权组合
        like_repr = self.apply_gate(expert_outputs, F.softmax(self.like_gate(x), dim=1))
        high_repr = self.apply_gate(expert_outputs, F.softmax(self.high_rating_gate(x), dim=1))
        rating_repr = self.apply_gate(expert_outputs, F.softmax(self.rating_gate(x), dim=1))

        return {
            "like_logit": self.like_tower(like_repr),
            "high_rating_logit": self.high_rating_tower(high_repr),
            "rating_pred": torch.sigmoid(self.rating_tower(rating_repr)),  # 归一化到 [0,1]
        }


# ---------- 损失函数 ----------

def compute_mmoe_loss(outputs, batch):
    # 三任务加权损失：like 权重最大（0.5），高分预测次之（0.3），评分回归最小（0.2）
    like_loss = nn.BCEWithLogitsLoss()(outputs["like_logit"], batch["like_label"].float())
    high_loss = nn.BCEWithLogitsLoss()(outputs["high_rating_logit"], batch["high_rating_label"].float())
    rating_loss = nn.MSELoss()(outputs["rating_pred"], batch["rating_label"].float())

    total_loss = 0.5 * like_loss + 0.3 * high_loss + 0.2 * rating_loss
    return {
        "total_loss": total_loss,
        "like_loss": like_loss,
        "high_loss": high_loss,
        "rating_loss": rating_loss,
    }


def forward_from_batch(model, batch):
    # 从 batch 字典中取出各字段，调用 model.forward()
    return model(
        user_id=batch["user_id"], gender=batch["gender"], age=batch["age"],
        occupation=batch["occupation"], movie_id=batch["movie_id"], genres=batch["genres"],
        recall_score=batch["recall_score"], coarse_score=batch["coarse_score"],
    )


# ---------- 训练与评估 ----------

_LOSS_KEYS = ("total", "like", "high", "rating")


def _accum_losses(sums, losses):
    sums["total"] += losses["total_loss"].item()
    sums["like"] += losses["like_loss"].item()
    sums["high"] += losses["high_loss"].item()
    sums["rating"] += losses["rating_loss"].item()


def _avg_losses(sums, n):
    return {f"{k}_loss": sums[k] / n for k in _LOSS_KEYS}


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    sums = dict.fromkeys(_LOSS_KEYS, 0)

    for i, batch in enumerate(dataloader, 1):
        batch = move_batch_to_device(batch, device)
        losses = compute_mmoe_loss(forward_from_batch(model, batch), batch)
        optimizer.zero_grad()
        losses["total_loss"].backward()
        optimizer.step()
        _accum_losses(sums, losses)

    return _avg_losses(sums, i)


def evaluate(model, dataloader, device):
    model.eval()
    sums = dict.fromkeys(_LOSS_KEYS, 0)
    like_correct = high_correct = sample_count = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader, 1):
            batch = move_batch_to_device(batch, device)
            outputs = forward_from_batch(model, batch)
            losses = compute_mmoe_loss(outputs, batch)

            # 二分类准确率：sigmoid > 0.5 预测为正
            like_pred = (torch.sigmoid(outputs["like_logit"]) >= 0.5).float()
            high_pred = (torch.sigmoid(outputs["high_rating_logit"]) >= 0.5).float()
            like_correct += (like_pred == batch["like_label"]).sum().item()
            high_correct += (high_pred == batch["high_rating_label"]).sum().item()
            sample_count += batch["like_label"].shape[0]
            _accum_losses(sums, losses)

    return {
        **_avg_losses(sums, i),
        "like_accuracy": like_correct / sample_count,
        "high_rating_accuracy": high_correct / sample_count,
    }


def inference_rank(model, batch, device, top_k=50, score_name="like"):
    # 对一个 batch 打分并返回 top_k 的下标和分数（供外部排序使用）
    model.eval()
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        outputs = forward_from_batch(model, batch)
        if score_name == "like":
            scores = torch.sigmoid(outputs["like_logit"])
        elif score_name == "high_rating":
            scores = torch.sigmoid(outputs["high_rating_logit"])
        elif score_name == "rating":
            scores = outputs["rating_pred"]
        else:
            raise ValueError("score_name 须为 like、high_rating 或 rating 之一")

        top_scores, top_indexes = torch.topk(scores, min(top_k, scores.shape[0]))

    return {"indexes": top_indexes.cpu().tolist(), "scores": top_scores.cpu().tolist()}
