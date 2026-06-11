# 三塔粗排模型
# 用户塔 + 物品塔 + Dense 统计特征塔，三路各输出 64 维，拼接后经 MLP 输出单个粗排分数
import torch
import torch.nn as nn


class ThreeTowerRoughRankModel(nn.Module):
    def __init__(
        self,
        user_count,
        movie_count,
        gender_count,
        age_count,
        occupation_count,
        genre_count,
        dense_feature_dim,  # Dense 塔的输入维度，目前为 5
    ):
        super().__init__()

        # --- 用户塔 ---
        # 输入: user_id(32) + gender(4) + age(8) + occupation(8) = 52 维
        self.user_embedding = nn.Embedding(user_count, 32)
        self.gender_embedding = nn.Embedding(gender_count, 4)
        self.age_embedding = nn.Embedding(age_count, 8)
        self.occupation_embedding = nn.Embedding(occupation_count, 8)
        self.user_tower = nn.Sequential(
            nn.Linear(52, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )

        # --- 物品塔 ---
        # 输入: movie_id(32) + genre one-hot -> 16 维 = 48 维
        self.movie_embedding = nn.Embedding(movie_count, 32)
        self.genre_layer = nn.Linear(genre_count, 16)
        self.movie_tower = nn.Sequential(
            nn.Linear(48, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )

        # --- Dense 特征塔 ---
        # 输入: user_avg_rating, user_count, movie_avg_rating, movie_count, recall_score
        self.dense_tower = nn.Sequential(
            nn.Linear(dense_feature_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

        # --- 融合层 ---
        # 三塔拼接后 192 维，输出单个粗排分数
        self.output_layer = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        user_index,
        gender_index,
        age_index,
        occupation_index,
        movie_index,
        genre_vector,
        dense_features,
    ):
        # 用户塔前向
        user_input = torch.cat([
            self.user_embedding(user_index),
            self.gender_embedding(gender_index),
            self.age_embedding(age_index),
            self.occupation_embedding(occupation_index),
        ], dim=1)
        user_vector = self.user_tower(user_input)

        # 物品塔前向
        movie_input = torch.cat([
            self.movie_embedding(movie_index),
            self.genre_layer(genre_vector),
        ], dim=1)
        movie_vector = self.movie_tower(movie_input)

        # Dense 塔前向
        dense_vector = self.dense_tower(dense_features)

        # 拼接三塔输出，输出粗排分数
        final_input = torch.cat([user_vector, movie_vector, dense_vector], dim=1)
        score = self.output_layer(final_input)

        return score.squeeze(1)
