import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch, device):
    moved_batch = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value

    return moved_batch


class Expert(nn.Module):
    def __init__(self, input_dim, expert_dim=64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, expert_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class TaskTower(nn.Module):
    def __init__(self, expert_dim=64):
        super().__init__()

        self.tower = nn.Sequential(
            nn.Linear(expert_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.tower(x).squeeze(1)


class MMoERanker(nn.Module):
    def __init__(
        self,
        user_count,
        movie_count,
        gender_count,
        age_count,
        occupation_count,
        genre_count,
        num_experts=4,
        expert_dim=64,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.expert_dim = expert_dim

        self.user_embedding = nn.Embedding(user_count, 32)
        self.movie_embedding = nn.Embedding(movie_count, 32)
        self.gender_embedding = nn.Embedding(gender_count, 4)
        self.age_embedding = nn.Embedding(age_count, 8)
        self.occupation_embedding = nn.Embedding(occupation_count, 8)

        # genres 是多标签特征，0 留给 padding，所以 genre_count 需要包含 padding 位置
        self.genre_embedding = nn.Embedding(genre_count, 16, padding_idx=0)

        # 32 + 32 + 4 + 8 + 8 + 16 + recall_score 1 + coarse_score 1 = 102
        self.input_dim = 102

        self.experts = nn.ModuleList(
            [
                Expert(input_dim=self.input_dim, expert_dim=expert_dim)
                for _ in range(num_experts)
            ]
        )

        self.like_gate = nn.Linear(self.input_dim, num_experts)
        self.high_rating_gate = nn.Linear(self.input_dim, num_experts)
        self.rating_gate = nn.Linear(self.input_dim, num_experts)

        self.like_tower = TaskTower(expert_dim=expert_dim)
        self.high_rating_tower = TaskTower(expert_dim=expert_dim)
        self.rating_tower = TaskTower(expert_dim=expert_dim)

    def pool_genres(self, genres):
        genre_vectors = self.genre_embedding(genres)
        mask = (genres != 0).float().unsqueeze(-1)

        genre_sum = (genre_vectors * mask).sum(dim=1)
        genre_count = mask.sum(dim=1).clamp(min=1)

        return genre_sum / genre_count

    def build_input(
        self,
        user_id,
        gender,
        age,
        occupation,
        movie_id,
        genres,
        recall_score,
        coarse_score,
    ):
        user_vector = self.user_embedding(user_id)
        movie_vector = self.movie_embedding(movie_id)
        gender_vector = self.gender_embedding(gender)
        age_vector = self.age_embedding(age)
        occupation_vector = self.occupation_embedding(occupation)
        genre_vector = self.pool_genres(genres)

        recall_score = recall_score.float().view(-1, 1)
        coarse_score = coarse_score.float().view(-1, 1)

        return torch.cat(
            [
                user_vector,
                movie_vector,
                gender_vector,
                age_vector,
                occupation_vector,
                genre_vector,
                recall_score,
                coarse_score,
            ],
            dim=1,
        )

    def apply_gate(self, expert_outputs, gate_weights):
        # expert_outputs: [batch_size, num_experts, expert_dim]
        # gate_weights: [batch_size, num_experts]
        gate_weights = gate_weights.unsqueeze(-1)
        task_representation = (expert_outputs * gate_weights).sum(dim=1)
        return task_representation

    def forward(
        self,
        user_id,
        gender,
        age,
        occupation,
        movie_id,
        genres,
        recall_score,
        coarse_score,
    ):
        input_vector = self.build_input(
            user_id=user_id,
            gender=gender,
            age=age,
            occupation=occupation,
            movie_id=movie_id,
            genres=genres,
            recall_score=recall_score,
            coarse_score=coarse_score,
        )

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(input_vector))

        expert_outputs = torch.stack(expert_outputs, dim=1)

        like_gate_weights = F.softmax(self.like_gate(input_vector), dim=1)
        high_gate_weights = F.softmax(self.high_rating_gate(input_vector), dim=1)
        rating_gate_weights = F.softmax(self.rating_gate(input_vector), dim=1)

        like_representation = self.apply_gate(expert_outputs, like_gate_weights)
        high_representation = self.apply_gate(expert_outputs, high_gate_weights)
        rating_representation = self.apply_gate(expert_outputs, rating_gate_weights)

        like_logit = self.like_tower(like_representation)
        high_rating_logit = self.high_rating_tower(high_representation)

        rating_raw = self.rating_tower(rating_representation)
        rating_pred = torch.sigmoid(rating_raw)

        return {
            "like_logit": like_logit,
            "high_rating_logit": high_rating_logit,
            "rating_pred": rating_pred,
        }


def compute_mmoe_loss(outputs, batch):
    like_loss_fn = nn.BCEWithLogitsLoss()
    high_loss_fn = nn.BCEWithLogitsLoss()
    rating_loss_fn = nn.MSELoss()

    like_loss = like_loss_fn(outputs["like_logit"], batch["like_label"].float())
    high_loss = high_loss_fn(
        outputs["high_rating_logit"], batch["high_rating_label"].float()
    )
    rating_loss = rating_loss_fn(outputs["rating_pred"], batch["rating_label"].float())

    total_loss = 0.5 * like_loss + 0.3 * high_loss + 0.2 * rating_loss

    return {
        "total_loss": total_loss,
        "like_loss": like_loss,
        "high_loss": high_loss,
        "rating_loss": rating_loss,
    }


def forward_from_batch(model, batch):
    return model(
        user_id=batch["user_id"],
        gender=batch["gender"],
        age=batch["age"],
        occupation=batch["occupation"],
        movie_id=batch["movie_id"],
        genres=batch["genres"],
        recall_score=batch["recall_score"],
        coarse_score=batch["coarse_score"],
    )


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()

    total_loss_sum = 0
    like_loss_sum = 0
    high_loss_sum = 0
    rating_loss_sum = 0
    batch_count = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = forward_from_batch(model, batch)
        losses = compute_mmoe_loss(outputs, batch)

        optimizer.zero_grad()
        losses["total_loss"].backward()
        optimizer.step()

        total_loss_sum += losses["total_loss"].item()
        like_loss_sum += losses["like_loss"].item()
        high_loss_sum += losses["high_loss"].item()
        rating_loss_sum += losses["rating_loss"].item()
        batch_count += 1

    return {
        "total_loss": total_loss_sum / batch_count,
        "like_loss": like_loss_sum / batch_count,
        "high_loss": high_loss_sum / batch_count,
        "rating_loss": rating_loss_sum / batch_count,
    }


def evaluate(model, dataloader, device):
    model.eval()

    total_loss_sum = 0
    like_loss_sum = 0
    high_loss_sum = 0
    rating_loss_sum = 0
    like_correct = 0
    high_correct = 0
    sample_count = 0
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = forward_from_batch(model, batch)
            losses = compute_mmoe_loss(outputs, batch)

            like_prob = torch.sigmoid(outputs["like_logit"])
            high_prob = torch.sigmoid(outputs["high_rating_logit"])

            like_pred = (like_prob >= 0.5).float()
            high_pred = (high_prob >= 0.5).float()

            like_correct += (like_pred == batch["like_label"]).sum().item()
            high_correct += (high_pred == batch["high_rating_label"]).sum().item()
            sample_count += batch["like_label"].shape[0]

            total_loss_sum += losses["total_loss"].item()
            like_loss_sum += losses["like_loss"].item()
            high_loss_sum += losses["high_loss"].item()
            rating_loss_sum += losses["rating_loss"].item()
            batch_count += 1

    return {
        "total_loss": total_loss_sum / batch_count,
        "like_loss": like_loss_sum / batch_count,
        "high_loss": high_loss_sum / batch_count,
        "rating_loss": rating_loss_sum / batch_count,
        "like_accuracy": like_correct / sample_count,
        "high_rating_accuracy": high_correct / sample_count,
    }


def inference_rank(model, batch, device, top_k=50, score_name="like"):
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
            raise ValueError("score_name must be like, high_rating, or rating")

        top_k = min(top_k, scores.shape[0])
        top_scores, top_indexes = torch.topk(scores, top_k)

    return {
        "indexes": top_indexes.cpu().tolist(),
        "scores": top_scores.cpu().tolist(),
    }
