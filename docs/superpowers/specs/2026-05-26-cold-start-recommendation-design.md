# 新用户冷启动推荐设计

## 背景

当前推荐链路是：

```text
双塔召回 -> 三塔粗排 -> MMoE 精排 -> 重排
```

这条链路主要服务训练集中出现过的老用户。对于新注册用户，虽然系统可以拿到 `user_id`、`age`、`occupation` 等注册信息，但该用户没有历史行为，也没有出现在模型训练阶段的 `user_id_to_index` 中。

因此，现有模型会遇到两个问题：

1. 双塔召回中的 `user_id embedding` 无法处理新的 `user_id`。
2. 粗排和精排同样依赖训练时建立的用户索引，新用户会返回空结果。

第一版冷启动目标不是重训模型，而是在现有 pipeline 外侧增加一个规则型兜底链路，让新用户也能拿到可解释、稳定、多样化的推荐结果。

## 用户信息假设

第一版默认新注册用户至少有以下信息：

```text
user_id
age
occupation
```

其中：

1. `user_id` 用于标识请求、日志记录和后续行为沉淀。
2. `age` 和 `occupation` 用于画像分群推荐。
3. `gender`、注册问卷、偏好类型、实时点击等信息第一版暂不依赖。

## 目标

1. 新用户没有历史行为时，也能返回 TopK 推荐结果。
2. 不改变老用户原有推荐链路和推荐结果格式。
3. 使用 `age + occupation` 做画像分群冷启动。
4. 用电影热度、平均评分和分群偏好生成冷启动分数。
5. 保持推荐结果的类型多样性，避免单一类型刷屏。
6. 输出结果能标识来自冷启动链路，方便调试和评估。

## 非目标

1. 第一版不训练新的深度模型。
2. 第一版不解决新物品冷启动。
3. 第一版不接入数据库、Redis 或线上服务。
4. 第一版不做 bandit 探索、在线学习或实时画像更新。
5. 第一版不要求用户选择喜欢的电影类型。

## 常见方案对比

### 方案一：全局热门高分兜底

基于所有用户的评分数据，推荐评分高、评分人数多、正反馈多的电影。

优点：

1. 实现最简单。
2. 稳定性强。
3. 不依赖用户画像。

缺点：

1. 个性化弱。
2. 所有新用户看到的结果接近。

### 方案二：画像分群冷启动

根据新用户的 `age` 和 `occupation`，找到训练集中相同或相近画像用户喜欢的电影，再结合电影全局评分和热度排序。

优点：

1. 能利用注册信息做弱个性化。
2. 不需要训练新模型。
3. 和当前 MovieLens 数据结构匹配。
4. 结果容易解释。

缺点：

1. 分群数据过少时需要兜底。
2. 个性化粒度不如真实行为推荐。

### 方案三：画像 + 类型偏好问卷

在画像分群基础上，让新用户选择喜欢的电影类型，例如 `Action`、`Comedy`、`Drama`，再增强类型匹配分数。

优点：

1. 冷启动个性化更强。
2. 用户偏好更明确。

缺点：

1. 需要额外产品交互。
2. 第一版实现和评估更复杂。

## 推荐方案

第一版采用：

```text
age + occupation 画像分群冷启动
```

整体策略是：

```text
新用户请求
  -> 使用 age + occupation 获取分群候选
  -> 候选不足时依次使用 age、occupation、global 兜底
  -> 根据分群偏好、电影平均评分、电影热度计算冷启动分数
  -> 类型打散
  -> 返回 TopK
```

## 模块设计

新增一个冷启动推荐模块：

```text
ColdStartRecommender
```

建议文件位置：

```text
cold_start/
  __init__.py
  cold_start_recommender.py
```

核心职责：

1. 加载训练集用户信息、评分数据和电影信息。
2. 预计算电影全局统计。
3. 预计算 `age + occupation` 分群偏好。
4. 根据新用户画像生成冷启动推荐。
5. 保持输出字段与主推荐链路兼容。

## 离线统计数据

初始化时从以下文件读取数据：

```text
train_data/users.dat
train_data/ratings.dat
data/movies.dat
```

需要构建的统计表：

```text
movie_stats:
  movie_id -> {
    rating_sum,
    rating_count,
    avg_rating,
    positive_count
  }

segment_movie_stats:
  (age, occupation) -> {
    movie_id -> positive_count
  }

age_movie_stats:
  age -> {
    movie_id -> positive_count
  }

occupation_movie_stats:
  occupation -> {
    movie_id -> positive_count
  }

movie_info:
  movie_id -> {
    title,
    genres
  }
```

正反馈定义沿用当前项目习惯：

```text
rating >= 4 -> positive
```

## 冷启动打分

第一版使用可解释的规则分：

```text
cold_start_score =
  0.45 * segment_positive_score
  + 0.35 * movie_avg_rating_score
  + 0.20 * movie_popularity_score
```

其中：

```text
segment_positive_score = 当前画像分群中，该电影正反馈次数的归一化分数
movie_avg_rating_score = movie_avg_rating / 5
movie_popularity_score = log(1 + movie_rating_count) 的归一化分数
```

如果 `(age, occupation)` 分群候选不足，则按以下顺序补充候选：

```text
1. age + occupation 分群
2. age 分群
3. occupation 分群
4. global 热门高分
```

每个候选结果需要记录来源：

```text
cold_start_source = "age_occupation" | "age" | "occupation" | "global"
```

## 类型打散

冷启动结果也要经过类型打散，避免推荐列表被单一类型占满。

第一版可以复用当前 `Reranker` 的主类型打散逻辑：

```text
按 cold_start_score 从高到低遍历候选
尽量避免相邻两个电影的 primary_genre 相同
直到选满 top_k
```

## Pipeline 接入方式

老用户链路保持不变：

```text
双塔召回 -> 三塔粗排 -> MMoE 精排 -> 重排
```

新用户或召回为空时走冷启动：

```text
if user_id 不在召回模型训练用户中:
    使用 ColdStartRecommender
elif 双塔召回结果为空:
    使用 ColdStartRecommender
else:
    使用原推荐链路
```

推荐接口建议在 `recommend()` 中增加可选参数：

```python
pipeline.recommend(
    user_id=900001,
    age=25,
    occupation=4,
    top_k=20,
)
```

老用户仍然可以继续使用原调用方式：

```python
pipeline.recommend(user_id=15, top_k=20)
```

当用户是新用户，但没有传入 `age` 或 `occupation` 时，系统使用全局热门高分兜底。

## 返回格式

冷启动结果建议保持与主链路结果相近：

```python
{
    "item_id": 318,
    "movie_id": 318,
    "title": "Shawshank Redemption, The (1994)",
    "cold_start_score": 0.91,
    "recall_score": 0.91,
    "recall_source": "cold_start",
    "cold_start_source": "age_occupation",
    "rerank_primary_genre": "Drama",
}
```

说明：

1. `cold_start_score` 是冷启动规则分。
2. `recall_score` 可以临时复用冷启动分，保持下游显示兼容。
3. `recall_source` 标记为 `cold_start`。
4. `cold_start_source` 标记具体兜底来源。

## 错误处理

1. `age` 或 `occupation` 缺失时，不报错，使用全局热门高分推荐。
2. `age` 或 `occupation` 在训练集中未出现时，跳过对应分群，继续使用其他兜底。
3. 某个分群候选不足时，继续从下一级候选池补齐。
4. 如果所有统计数据为空，返回空列表，并在日志或返回字段中标记冷启动失败原因。

## 评估方式

第一版可以增加离线模拟评估：

```text
从测试集中选择一批用户
隐藏他们的历史行为
只使用这些用户的 age 和 occupation
调用冷启动推荐
用测试集中 rating >= 4 的电影作为正样本
计算 Precision@K / Recall@K / HitRate@K / MRR@K / NDCG@K
```

建议先评估：

```text
K = 10, 20
max_users = 1000
```

并与当前完整链路指标分开展示，因为冷启动只使用画像信息，不能直接和老用户完整链路做完全公平对比。

## 测试计划

需要覆盖：

1. 新用户传入 `age + occupation` 时返回非空推荐。
2. 新用户只传入 `age` 时可以走 age 兜底。
3. 新用户只传入 `occupation` 时可以走 occupation 兜底。
4. 新用户没有画像信息时可以走 global 兜底。
5. 冷启动结果包含 `cold_start_score`、`recall_source`、`cold_start_source`。
6. 老用户不受冷启动逻辑影响，仍走原推荐链路。
7. `recommend_with_timing()` 能记录冷启动阶段耗时和结果数量。

## 后续升级方向

### V2：加入类型偏好

支持新用户传入：

```python
preferred_genres=["Action", "Comedy"]
```

在冷启动分数中加入：

```text
genre_match_score
```

### V3：短期行为画像

当新用户产生少量点击或评分后，构建轻量实时画像：

```text
最近点击 genres
最近高分电影
最近浏览电影
```

再逐步从冷启动链路切换到主推荐链路。

### V4：模型侧支持新用户画像

改造模型输入，减少对 `user_id embedding` 的强依赖，让模型能通过 `age`、`occupation`、`gender` 等画像特征直接服务新用户。

这需要重新训练模型，不属于第一版范围。

## 验收标准

1. 对一个训练集中不存在的新 `user_id`，只要传入 `age` 和 `occupation`，系统可以返回 TopK 推荐。
2. 老用户原有推荐链路不受影响。
3. 冷启动结果字段可解释，能看出来自哪一种兜底来源。
4. 冷启动推荐结果经过类型打散。
5. 有单元测试覆盖主要兜底路径。
6. 有离线评估命令可以单独评估冷启动效果。
