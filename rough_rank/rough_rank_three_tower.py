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
        dense_feature_dim,
    ):
        super().__init__()

        # 用户塔输入：user_id_emb 32维 + gender_emb 4维 + age_emb 8维 + occupation_emb 8维
        self.user_embedding = nn.Embedding(user_count, 32)
        self.gender_embedding = nn.Embedding(gender_count, 4)
        self.age_embedding = nn.Embedding(age_count, 8)
        self.occupation_embedding = nn.Embedding(occupation_count, 8)

        # 用户塔 MLP: 52 -> 128 -> 64
        self.user_tower = nn.Sequential(
            nn.Linear(52, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # 物品塔输入：movie_id_emb 32维 + genres_emb 16维
        self.movie_embedding = nn.Embedding(movie_count, 32)
        self.genre_layer = nn.Linear(genre_count, 16)

        # 物品塔 MLP: 48 -> 128 -> 64
        self.movie_tower = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # 第三塔：放粗排常用的连续特征
        # 例如 recall_score、movie_avg_rating、movie_rating_count、user_avg_rating 等
        self.dense_tower = nn.Sequential(
            nn.Linear(dense_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # 三个塔各输出 64 维，拼接后是 192 维
        # 最后输出一个粗排分数
        self.output_layer = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
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
        user_id_vector = self.user_embedding(user_index)
        gender_vector = self.gender_embedding(gender_index)
        age_vector = self.age_embedding(age_index)
        occupation_vector = self.occupation_embedding(occupation_index)

        user_input = torch.cat(
            [user_id_vector, gender_vector, age_vector, occupation_vector], dim=1
        )
        user_vector = self.user_tower(user_input)

        movie_id_vector = self.movie_embedding(movie_index)
        genre_feature_vector = self.genre_layer(genre_vector)

        movie_input = torch.cat([movie_id_vector, genre_feature_vector], dim=1)
        movie_vector = self.movie_tower(movie_input)

        dense_vector = self.dense_tower(dense_features)

        final_input = torch.cat([user_vector, movie_vector, dense_vector], dim=1)
        score = self.output_layer(final_input)

        return score.squeeze(1)
