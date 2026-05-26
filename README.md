# MovieLens 电影推荐系统

这是一个基于 MovieLens 数据集实现的电影推荐系统项目。当前项目已经包含完整的离线推荐链路：

```text
双塔召回 -> 三塔粗排 -> MMoE 精排 -> 重排
```

系统目前主要面向训练集中出现过的老用户。对于新用户冷启动，目前还没有正式接入画像推荐或热门高分兜底策略。

## 1. 项目结构

```text
recommend/
  data/                         # 原始 MovieLens 数据
    users.dat
    movies.dat
    ratings.dat
    README

  train_data/                   # 训练集数据
    users.dat
    movies.dat
    ratings.dat

  test_data/                    # 测试集数据
    users.dat
    movies.dat
    ratings.dat

  recall/                       # 召回模块
    two_tower.py                # 双塔召回模型、训练、推理
    evaluate.py                 # 召回模型评估
    movie_utils.py              # 电影标题等辅助工具

  model_weights/                # 双塔召回模型权重
    two_tower.pt
    model_epoch_*.pt

  rough_rank/                   # 粗排模块代码
    rough_rank_three_tower.py   # 三塔粗排模型结构
    train_rough_rank.py         # 三塔粗排训练脚本
    rough_rank_inference.py     # 三塔粗排推理

  rough_rank_model/             # 粗排模型权重
    rough_rank_three_tower.pt
    rough_rank_epoch_*.pt

  fine_rank/                    # 精排模块代码
    mmoe_ranker.py              # MMoE 精排模型结构
    train_mmoe_ranker.py        # MMoE 精排训练脚本
    mmoe_inference.py           # MMoE 精排推理

  fine_rank_model/              # 精排模型权重
    mmoe_ranker.pt
    mmoe_epoch_*.pt

  recommender_pipeline.py       # 推荐系统主链路入口
  structure.md                  # 项目设计和结构说明
```

## 2. 数据说明

项目使用 MovieLens 1M 风格数据，核心数据文件有三类。

### users.dat

```text
UserID::Gender::Age::Occupation::Zip-code
```

当前用到的用户特征：

```text
user_id
gender
age
occupation
```

### movies.dat

```text
MovieID::Title::Genres
```

当前用到的电影特征：

```text
movie_id
genres
title
```

其中 `genres` 是多标签特征，例如：

```text
Animation|Children's|Comedy
```

### ratings.dat

```text
UserID::MovieID::Rating::Timestamp
```

当前训练中常用标签规则：

```text
rating >= 4  -> 喜欢
rating == 5  -> 高评分
rating / 5   -> 归一化评分
```

## 3. 当前推荐链路

主入口在：

```text
recommender_pipeline.py
```

当前默认链路为：

```text
1. 双塔召回 300 部电影
2. 三塔粗排保留前 100 部电影
3. MMoE 精排保留前 50 部电影
4. 重排输出最终 top_k
```

代码入口：

```python
from recommender_pipeline import RecommenderPipeline

pipeline = RecommenderPipeline()
recommendations = pipeline.recommend(
    user_id=15,
    top_k=10,
    recall_size=300,
    rough_rank_size=100,
    fine_rank_size=50,
)
```

运行示例：

```bash
python recommender_pipeline.py
```

如果使用当前 conda 环境：

```bash
conda activate recommend
python recommender_pipeline.py
```

或者直接指定解释器：

```bash
D:\Anaconda\envs\recommend\python.exe recommender_pipeline.py
```

### 测试与端到端评估命令

单元测试命令：

```bash
python -m unittest discover -s tests [参数]
python -m tests.test_evaluate_pipeline [参数]
python -m tests.test_pipeline_timing [参数]
```

常用示例：

```bash
python -m unittest discover -s tests
python -m tests.test_evaluate_pipeline
python -m tests.test_pipeline_timing
```

端到端 pipeline 评估命令：

```bash
python evaluate_pipeline.py [参数]
D:\Anaconda\envs\recommend\python.exe evaluate_pipeline.py [参数]
```

常用参数：

```text
--ks 10,20              评估的 K 列表，默认是 10,20
--max-users 1000        最多评估多少个用户，不传则评估全部测试用户
--with-timing           输出推荐请求内时延和整条命令墙钟耗时
--recall-size 300       双塔召回候选数量
--rough-rank-size 100   粗排保留候选数量
--fine-rank-size 50     精排保留候选数量
```

常用示例：

```bash
D:\Anaconda\envs\recommend\python.exe evaluate_pipeline.py --ks 10,20 --max-users 1000 --with-timing
D:\Anaconda\envs\recommend\python.exe evaluate_pipeline.py --ks 5,10 --max-users 100 --recall-size 300 --rough-rank-size 100 --fine-rank-size 50 --with-timing
```

输出中的 `Latency timing` 表示模型已经加载后的单用户推荐平均耗时；`Command timing` 表示整条命令的墙钟耗时，包括模型初始化、评估循环和输出打印。

## 4. 召回模块

召回模块位于：

```text
recall/two_tower.py
```

当前使用双塔模型：

```text
用户塔:
  user_id embedding: 32 维
  gender embedding: 4 维
  age embedding: 8 维
  occupation embedding: 8 维
  MLP: 52 -> 128 -> 64 -> 64

物品塔:
  movie_id embedding: 32 维
  genres 特征: 16 维
  MLP: 48 -> 128 -> 64 -> 64
```

最后使用余弦相似度作为召回分数：

```text
score = cosine(user_vector, movie_vector)
```

当前 pipeline 已经使用 `TwoTowerRecaller` 类，模型只在初始化时加载一次，避免每次请求重复读取权重。

### 训练召回模型

```bash
python -m recall.two_tower --mode train --epochs 3
```

### 单独召回

```bash
python -m recall.two_tower --mode recommend --user-id 1 --top-k 10
```

### 评估召回模型

```bash
python -m recall.evaluate
```

默认会评估：

```text
Precision@10 / Recall@10 / HitRate@10
Precision@20 / Recall@20 / HitRate@20
Precision@100 / Recall@100 / HitRate@100
Precision@300 / Recall@300 / HitRate@300
```

并默认按照 `Recall@300` 选择最佳召回模型。

## 5. 粗排模块

粗排模块位于：

```text
rough_rank/
```

模型文件：

```text
rough_rank/rough_rank_three_tower.py
```

当前使用三塔粗排模型：

```text
用户塔:
  user_id
  gender
  age
  occupation

物品塔:
  movie_id
  genres

连续特征塔:
  user_avg_rating
  user_rating_count
  movie_avg_rating
  movie_rating_count
```

三路输出拼接后得到粗排分数：

```text
user_vector + movie_vector + dense_vector -> rough_rank_score
```

### 训练粗排模型

```bash
python -m rough_rank.train_rough_rank --epochs 3
```

也可以调大 batch：

```bash
python -m rough_rank.train_rough_rank --epochs 3 --batch-size 4096
```

训练后的模型保存在：

```text
rough_rank_model/
```

### 粗排推理

推理类：

```python
from rough_rank.rough_rank_inference import RoughRanker

ranker = RoughRanker()
rough_ranked_items = ranker.rank(
    user_id=1,
    recalled_items=recalled_items,
    top_k=100,
)
```

pipeline 当前已经接入粗排模型。

## 6. 精排模块

精排模块位于：

```text
fine_rank/
```

模型文件：

```text
fine_rank/mmoe_ranker.py
```

当前使用 MMoE 多目标精排模型。

输入特征：

```text
user_id
gender
age
occupation
movie_id
genres
recall_score
coarse_score
```

其中：

```text
recall_score 来自双塔召回
coarse_score 来自三塔粗排
```

MMoE 结构：

```text
num_experts = 4
expert_dim = 64

每个 expert:
input_dim -> 256 -> 128 -> 64

每个任务有独立 gate:
Linear(input_dim, num_experts) + softmax
```

多目标任务：

```text
like          二分类
high_rating   二分类
rating        回归，输出 0 到 1 的归一化评分
```

训练损失：

```text
total_loss =
  0.5 * like_loss +
  0.3 * high_rating_loss +
  0.2 * rating_loss
```

### 训练精排模型

```bash
python -m fine_rank.train_mmoe_ranker --epochs 3
```

训练时会先使用已有模型生成：

```text
recall_score
coarse_score
```

然后再训练 MMoE。

精排模型保存在：

```text
fine_rank_model/
```

pipeline 当前默认加载：

```text
fine_rank_model/mmoe_epoch_6.pt
```

并使用 `like` 任务分数作为精排分：

```text
fine_rank_score = sigmoid(like_logit)
```

## 7. 重排模块

重排逻辑目前写在：

```text
recommender_pipeline.py
```

当前重排做两件事：

```text
1. 过滤用户已经看过或评分过的电影
2. 按电影主类型做简单打散
```

已看电影来自：

```text
train_data/ratings.dat
```

电影类型来自：

```text
data/movies.dat
```

重排后会给结果补充：

```text
rerank_primary_genre
```

## 8. 推荐结果字段

pipeline 最终返回的是一个列表，每个元素大致包含：

```python
{
    "item_id": 2438,
    "movie_id": 2438,
    "title": "Outside Ozona (1998)",
    "recall_score": 0.9997,
    "recall_source": "two_tower",
    "rough_rank_score": 4.8976,
    "fine_rank_score": 0.9573,
    "fine_rank_source": "mmoe_epoch_6_like",
    "rerank_primary_genre": "Drama",
}
```

## 9. 运行顺序建议

如果从头训练，建议顺序是：

```text
1. 确认 train_data 和 test_data 已存在
2. 训练双塔召回
3. 评估双塔召回
4. 训练三塔粗排
5. 训练 MMoE 精排
6. 运行 recommender_pipeline.py
```

对应命令：

```bash
python -m recall.two_tower --mode train --epochs 3
python -m recall.evaluate
python -m rough_rank.train_rough_rank --epochs 3
python -m fine_rank.train_mmoe_ranker --epochs 3
python recommender_pipeline.py
```

如果使用 `recommend` 环境：

```bash
conda activate recommend
```

## 10. 当前已做的优化

当前 pipeline 已经避免了部分重复加载：

```text
TwoTowerRecaller 在 RecommenderPipeline 初始化时加载一次
RoughRanker 在 RecommenderPipeline 初始化时加载一次
MMoEFineRanker 在 RecommenderPipeline 初始化时加载一次
Reranker 在 RecommenderPipeline 初始化时加载一次
```

也就是说，正常使用时不要每个请求都重新创建 `RecommenderPipeline`。更推荐：

```python
pipeline = RecommenderPipeline()

# 后续多个请求复用这个 pipeline
pipeline.recommend(user_id=1)
pipeline.recommend(user_id=2)
pipeline.recommend(user_id=3)
```

## 11. 仍可继续优化的地方

后续可以继续优化这些点：

```text
1. 双塔召回预计算所有电影向量
2. 粗排统计特征改成 user/movie 查表
3. 精排 genres 特征预先转成 tensor
4. pipeline 增加冷启动兜底
5. 增加完整链路评估
6. 增加显式交叉特征，例如用户历史 genre 偏好 × 当前电影 genre、用户画像 × 电影类型、recall_score × coarse_score
7. 接入 MySQL 数据源
8. 增加 API 服务
```

其中优先级较高的是：

```text
双塔召回预计算电影向量
```

因为电影侧向量在模型不变时是静态的，不需要每次推荐都重新计算。

## 12. 当前限制

当前系统还有这些限制：

```text
1. 主要支持训练集中出现过的老用户
2. 新用户冷启动还没有正式接入
3. 没有 API 服务，只能本地脚本调用
4. 没有数据库接入，当前仍基于 dat 文件训练
5. 完整链路还没有统一评估脚本
6. 模型参数需要手动训练和替换
```

## 13. 后续接入 MySQL 的方向

后续可以将数据放入 MySQL，例如：

```text
users 表
movies 表
ratings 表
user_events 表
```

推荐的工程流程：

```text
1. 新用户注册写入 users
2. 用户行为写入 ratings 或 user_events
3. 定时从 MySQL 导出训练数据
4. 重新训练召回、粗排、精排模型
5. 评估新模型
6. 替换线上模型权重
```

这样系统就可以逐步从离线实验项目演进为可持续更新的推荐系统。
