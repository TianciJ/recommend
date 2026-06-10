# CHANGELOG

本文档按时间倒序记录项目各阶段的主要变化。

---

## [v0.4] 2026-06-10 — 准确率优化 + 工程整理

### 模型优化（需重新训练生效）

#### 召回层：双塔模型加入用户行为特征
- **旧版**：用户塔输入仅包含静态画像（user_id / gender / age / occupation），共 52 维
- **新版**：用户塔输入增加两个行为特征，共 54 维
  - `user_avg_rating / 5`：用户历史平均评分（归一化）
  - `log1p(rating_count) / log1p(max_count)`：用户活跃度（log 归一化）
- **动机**：静态画像无法区分"高评分用户"和"低评分用户"，行为特征能让模型学到用户的评分习惯
- 涉及文件：[recall/two_tower.py](recall/two_tower.py)

#### 粗排层：Dense 特征加入 recall_score
- **旧版**：Dense 特征 4 维（user_avg_rating、user_count、movie_avg_rating、movie_count）
- **新版**：Dense 特征扩展为 5 维，新增 `recall_score_feature = (recall_score + 1) / 2`
  - 训练时用中性值 `0.0` 占位（不引入虚假信号）
  - 推理时从召回结果取真实 recall_score 传入模型
- **动机**：粗排之前完全丢弃了召回分数，将其显式传入 Dense 塔可以让粗排感知召回置信度
- 涉及文件：[rough_rank/train.py](rough_rank/train.py)、[rough_rank/inference.py](rough_rank/inference.py)

#### 精排层：Genre 从 Mean Pooling 改为 Attention Pooling
- **旧版**：多个 genre embedding 直接做平均（mean pooling），每个 genre 权重相同
- **新版**：新增 `genre_attention = Linear(16, 1)` 层，对每个 genre 打标量权重后 softmax 加权求和；padding 位置 mask 为 `-1e9` 不参与计算
- **动机**：一部电影可能同时属于 Action / Adventure / Sci-Fi，但对用户偏好的贡献不同，attention 让模型自己学习哪个类型更重要
- 涉及文件：[fine_rank/model.py](fine_rank/model.py)

### 工程整理

#### 目录结构重组
- **旧版**：模型权重散落在三个平级目录（`model_weights/`、`rough_rank_model/`、`fine_rank_model/`）
- **新版**：统一归入 `models/` 下按环节分子目录

```
models/
  recall/          # 双塔召回权重（原 model_weights/）
  rough_rank/      # 三塔粗排权重（原 rough_rank_model/）
  fine_rank/       # MMoE 精排权重（原 fine_rank_model/）
```

#### 权重文件重命名
| 旧文件名 | 新文件名 |
|---|---|
| `model_weights/two_tower.pt` | `models/recall/two_tower.pt` |
| `model_weights/model_epoch_*.pt` | `models/recall/two_tower_epoch_*.pt` |
| `rough_rank_model/rough_rank_three_tower.pt` | `models/rough_rank/three_tower.pt` |
| `rough_rank_model/rough_rank_epoch_*.pt` | `models/rough_rank/three_tower_epoch_*.pt` |
| `fine_rank_model/mmoe_ranker.pt` | `models/fine_rank/mmoe.pt` |
| `fine_rank_model/mmoe_epoch_*.pt` | `models/fine_rank/mmoe_epoch_*.pt` |

#### 代码文件重命名（命名风格统一）
| 旧文件名 | 新文件名 |
|---|---|
| `rough_rank/rough_rank_three_tower.py` | `rough_rank/model.py` |
| `rough_rank/rough_rank_inference.py` | `rough_rank/inference.py` |
| `rough_rank/train_rough_rank.py` | `rough_rank/train.py` |
| `fine_rank/mmoe_ranker.py` | `fine_rank/model.py` |
| `fine_rank/mmoe_inference.py` | `fine_rank/inference.py` |
| `fine_rank/train_mmoe_ranker.py` | `fine_rank/train.py` |

---

## [v0.3] 2026-05-26 — MySQL 数据源 + 用户冷启动

### 新增功能

#### MySQL 数据源接入
- 新增 `database/` 模块，包含 MySQL 连接、用户画像存储、数据集查询
- 召回、粗排、精排、冷启动全链路均支持从 MySQL 读取数据，回退到 `.dat` 文件
- 新增初始化脚本：`scripts/init_mysql_schema.py`、`scripts/import_movielens_to_mysql.py`
- 新增用户注册脚本：`scripts/register_user.py`
- 配置方式：通过环境变量 `MYSQL_HOST / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE`

#### 新用户冷启动
- 新增 `cold_start/` 模块，基于 `age × occupation` 画像分群
- 推荐分数公式：`0.45 × segment_positive_score + 0.35 × movie_avg_rating + 0.20 × movie_popularity`
- 兜底层级：`age+occupation 分群 → age 分群 → occupation 分群 → 全局热门`
- 当用户不在训练集、或召回 / 粗排 / 精排返回空结果时，pipeline 自动走冷启动

#### 推荐主链路增强
- `RecommenderPipeline` 新增 `recommend_with_timing()` 方法，输出各阶段耗时
- 新增端到端评估脚本 `evaluate_pipeline.py`，支持 `Precision@K / Recall@K / HitRate@K / MRR@K / NDCG@K`
- 新增 `recommend_for_user_id_or_register()`：用户不存在时引导注册并走冷启动
- 所有模型改为 Pipeline 初始化时加载一次，不再每次请求重新加载

### Bug 修复
- 修复：MySQL 用户 Repository 未配置时 pipeline 报错，改为自动 fallback

---

## [v0.2] 2026-05-25 — 完整三阶段推荐链路

### 新增功能

#### 精排：MMoE 多目标模型
- 新增 `fine_rank/` 模块
- 模型结构：4 个 Expert（102→256→128→64），每个任务独立 Gate + Tower
- 三个任务：`like`（二分类）、`high_rating`（二分类）、`rating`（回归）
- 训练损失：`0.5 × like_loss + 0.3 × high_rating_loss + 0.2 × rating_loss`
- 训练特征：在召回分数和粗排分数基础上加入用户画像和电影 genre
- Pipeline 默认使用 `mmoe_epoch_6`（验证集准确率最优，之后出现过拟合）

#### 粗排：三塔模型
- 新增 `rough_rank/` 模块
- 模型结构：用户塔 + 物品塔 + Dense 统计特征塔，三路 64 维拼接后输出分数
- Dense 特征：`user_avg_rating`、`user_rating_count`、`movie_avg_rating`、`movie_rating_count`
- 训练目标：二分类（rating ≥ 4 为正样本），测试集准确率约 85.56%

#### 重排
- 在 `RecommenderPipeline` 中实现 `Reranker`
- 过滤用户训练集中已评分电影
- 按电影 primary genre 交替打散，避免连续同类型

#### 推荐链路
- 完整链路：双塔召回 300 → 三塔粗排保留 100 → MMoE 精排保留 50 → 重排输出 top_k
- 新增 `recommender_pipeline.py` 作为统一入口

### 性能记录
- 100 用户端到端耗时约 6.5 秒（含模型加载）
- 1000 用户推理耗时约 28.6 秒
- 当前 top 20 推荐准确率最高可达 42.7%（部分用户）

---

## [v0.1] 2026-05-25 — 双塔召回基线

### 新增功能

#### 双塔召回模型
- 新增 `recall/` 模块
- 用户塔：`user_id(32) + gender(4) + age(8) + occupation(8)` → MLP `52→128→64→64`
- 物品塔：`movie_id(32) + genre(16)` → MLP `48→128→64→64`
- 相似度：余弦相似度
- 训练数据：MovieLens 1M，rating ≥ 4 为正样本，rating ≤ 2 为负样本，rating = 3 跳过
- 评估指标：Precision@K / Recall@K / HitRate@K（K = 10, 20, 100, 300）
- 300 部召回时最佳模型（epoch 5）的 Recall@300 约 24.15%

#### 数据集划分
- 原始数据：`data/`（MovieLens 1M，100 万条评分）
- 训练集：`train_data/`
- 测试集：`test_data/`

#### 项目基础结构
- 初始化 `.gitignore`、基础目录结构
- `structure.md`：最简推荐系统设计文档
