# MovieLens 推荐系统

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

用 MovieLens 1M 数据集，从零实现工业界标准的**四阶段推荐链路**。每个阶段独立成模块，代码可直接跑通。

```
双塔召回 (300) → 三塔粗排 (100) → MMoE 精排 (50) → 重排 → Top K
```

---

## 能学到什么

| 阶段 | 技术点 |
|------|--------|
| 召回 | 双塔神经网络、用户/物品 Embedding、余弦相似度 |
| 粗排 | 三塔模型、统计特征融合、召回分数透传 |
| 精排 | MMoE 多目标学习、Attention Pooling、多任务损失加权 |
| 冷启动 | 画像分群、多级兜底策略 |
| 工程 | 模型初始化复用、端到端评估、MySQL 数据源切换 |

---

## 系统架构

```
用户请求
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                       RecommenderPipeline                        │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │ 双塔召回  │ →  │ 三塔粗排  │ →  │ MMoE精排  │ →  │   重排   │   │
│  │  300部   │    │  100部   │    │   50部   │    │  Top K  │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘   │
│       │                                                          │
│       │ 任意阶段返回空                                             │
│       ▼                                                          │
│  ┌─────────────────────────────────────────┐                    │
│  │     冷启动：age×occupation 分群 → 热门兜底  │                    │
│  └─────────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

**环境安装**

```bash
conda create -n recommend python=3.10
conda activate recommend
pip install -r requirements.txt
```

**直接运行推荐**（使用仓库内已有模型权重）

```bash
python recommender_pipeline.py
# 提示输入 user_id，例如输入 15
```

输出：

```
1.  movie_id=527   title=Schindler's List (1993)         genre=Drama    recall=0.9997  rough=4.897  fine=0.957
2.  movie_id=912   title=Casablanca (1942)                genre=Romance  recall=0.9981  rough=4.710  fine=0.944
3.  movie_id=1193  title=One Flew Over the Cuckoo's Nest  genre=Drama    recall=0.9973  rough=4.688  fine=0.941
4.  movie_id=2571  title=Matrix, The (1999)               genre=Action   recall=0.9965  rough=4.651  fine=0.938
5.  movie_id=858   title=Godfather, The (1972)            genre=Crime    recall=0.9958  rough=4.623  fine=0.935
```

---

## 从头训练

按顺序执行，每一步依赖上一步的输出：

```bash
# 1. 训练双塔召回
python -m recall.two_tower --mode train --epochs 3

# 2. 评估召回，自动选出最优 epoch
python -m recall.evaluate

# 3. 训练三塔粗排
python -m rough_rank.train --epochs 3

# 4. 训练 MMoE 精排（自动生成 recall_score / coarse_score 作为特征）
python -m fine_rank.train --epochs 3

# 5. 端到端评估
python evaluate_pipeline.py --ks 10,20 --max-users 1000 --with-timing
```

端到端评估输出示例：

```
Evaluated users=1000  Skipped users=12
K=10  Precision@10=0.0892  Recall@10=0.0341  HitRate@10=0.6120  MRR@10=0.3015  NDCG@10=0.3241
K=20  Precision@20=0.0701  Recall@20=0.0518  HitRate@20=0.7380  MRR@20=0.3015  NDCG@20=0.3489

Latency timing
AvgTotal=27.43ms  Requests=1000
recall: AvgElapsed=18.21ms  AvgItems=300.0
rough_rank: AvgElapsed=5.12ms  AvgItems=100.0
fine_rank: AvgElapsed=3.47ms  AvgItems=50.0
rerank: AvgElapsed=0.63ms  AvgItems=20.0
```

---

## Web 界面

**安装 Web 依赖**

```bash
pip install fastapi "uvicorn[standard]" pydantic
```

**启动后端**

```bash
python server.py
```

访问 [http://localhost:8000](http://localhost:8000) 即可打开推荐界面。

界面支持：输入 user_id 获取推荐卡片、冷启动路径橙色标注、MySQL 可用时展示用户注册表单。

**API 接口**

| 接口 | 说明 |
|------|------|
| `POST /api/recommend` | 输入 user_id，返回 Top-K 推荐列表 |
| `GET /api/status` | 服务状态和 MySQL 连接情况 |
| `POST /api/users` | 注册新用户（需配置 MySQL） |
| `GET /api/users/{user_id}` | 查询用户画像（需配置 MySQL） |

启动后访问 [http://localhost:8000/docs](http://localhost:8000/docs) 可查看 FastAPI 自动生成的交互式接口文档，可以在浏览器里直接测试每个接口。

MySQL 未配置时推荐接口自动回退到 `.dat` 文件，功能完全正常；注册和查询接口返回 503。

---

## 第一次读代码，从这里开始

```
第一步  recommender_pipeline.py             整条链路怎么串起来的（约 300 行，先读这个）
第二步  recall/two_tower.py                 双塔模型结构 + 训练 + 推理
第三步  rough_rank/model.py + train.py      三塔模型结构和训练
第四步  fine_rank/model.py + train.py       MMoE 多目标模型
第五步  cold_start/cold_start_recommender.py  新用户冷启动逻辑
第六步  evaluate_pipeline.py               端到端评估指标
```

`recommender_pipeline.py` 只有 300 行，把每个阶段都调用了一遍，是读整个项目最好的起点。

---

## 各模块说明

### 召回 `recall/`

双塔模型分别学习用户和电影的 64 维向量表示，用余弦相似度匹配。

| 塔 | 输入特征 | 输出 |
|---|---|---|
| 用户塔 | user_id · gender · age · occupation · 平均评分 · 活跃度 | 64 维 |
| 物品塔 | movie_id · genres | 64 维 |

```bash
python -m recall.two_tower --mode train --epochs 3
python -m recall.evaluate
python -m recall.two_tower --mode recommend --user-id 1 --top-k 10
```

### 粗排 `rough_rank/`

三塔模型在用户塔和物品塔之外，额外引入 Dense 统计特征塔（含 recall_score），测试集准确率约 **85.56%**。

| 塔 | 输入特征 |
|---|---|
| 用户塔 | user_id · gender · age · occupation |
| 物品塔 | movie_id · genres |
| Dense 塔 | 用户/电影平均评分 · 评分数量 · recall_score |

```bash
python -m rough_rank.train --epochs 3
```

### 精排 `fine_rank/`

MMoE 多目标模型，4 个共享 Expert，3 个独立 Gate，同时优化三个任务：

| 任务 | 类型 | 标签定义 | 损失权重 |
|---|---|---|---|
| like | 二分类 | rating ≥ 4 | 0.5 |
| high_rating | 二分类 | rating = 5 | 0.3 |
| rating | 回归 | rating / 5 | 0.2 |

最终精排分 = `sigmoid(like_logit)`。Genre 特征使用 Attention Pooling，学习不同类型的重要性权重。

```bash
python -m fine_rank.train --epochs 3
```

> 精排训练会自动调用召回和粗排模型生成上游特征，所以需要先训练好前两个模型。

### 重排

对精排 Top 50 做两件事：过滤用户训练集中已看过的电影；按 primary genre 贪心打散，避免推荐结果连续出现同一类型。

### 冷启动 `cold_start/`

用户不在训练集、或任意阶段返回空时自动触发。

```
冷启动分 = 0.45 × 同类人群偏好 + 0.35 × 电影平均分 + 0.20 × 电影热度
```

兜底层级：`age+occupation 分群 → age 分群 → occupation 分群 → 全局热门`

---

## 评估指标

```bash
python evaluate_pipeline.py --ks 10,20 --max-users 1000 --with-timing
```

支持：`Precision@K` / `Recall@K` / `HitRate@K` / `MRR@K` / `NDCG@K`

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--ks` | K 值列表 | `10,20` |
| `--max-users` | 最多评估多少用户 | 全部 |
| `--with-timing` | 输出各阶段延迟 | 关闭 |
| `--recall-size` | 双塔召回数量 | `300` |
| `--rough-rank-size` | 粗排保留数量 | `100` |
| `--fine-rank-size` | 精排保留数量 | `50` |

---

## 数据说明

MovieLens 1M 风格，三个文件：

| 文件 | 格式 | 说明 |
|---|---|---|
| `users.dat` | `UserID::Gender::Age::Occupation::Zip` | 用户画像 |
| `movies.dat` | `MovieID::Title::Genres` | genres 为多标签，如 `Action\|Adventure\|Sci-Fi` |
| `ratings.dat` | `UserID::MovieID::Rating::Timestamp` | 1-5 分评分 |

已按时间切分为 `train_data/` 和 `test_data/`。

**Age 编码**（7 个区间）

| 值 | 含义 |
|---|---|
| 1 | 18 岁以下 |
| 18 | 18–24 |
| 25 | 25–34 |
| 35 | 35–44 |
| 45 | 45–49 |
| 50 | 50–55 |
| 56 | 56 岁以上 |

**Occupation 编码**（0–20）

| 值 | 含义 | 值 | 含义 |
|---|---|---|---|
| 0 | 其他 | 11 | 律师 |
| 1 | 教师/学者 | 12 | 程序员 |
| 2 | 艺术家 | 13 | 退休 |
| 3 | 文员/行政 | 14 | 销售/市场 |
| 4 | 在校学生 | 15 | 科学家 |
| 5 | 客服 | 16 | 自雇 |
| 6 | 医疗 | 17 | 技术/工程师 |
| 7 | 管理层 | 18 | 技工 |
| 8 | 农业 | 19 | 失业 |
| 9 | 家庭主妇 | 20 | 作家 |
| 10 | K-12 学生 | | |

---

## MySQL 支持（可选）

不配置时所有模块自动回退到 `.dat` 文件，功能完全一致。配置 MySQL 后支持在线用户注册和动态数据读取，详见 [docs/mysql-data-source.md](docs/mysql-data-source.md)。

---

## 工具脚本

### 初始化 MySQL 表结构

```bash
python scripts/init_mysql_schema.py
```

### 导入 MovieLens 数据到 MySQL

```bash
python scripts/import_movielens_to_mysql.py --train-dir train_data --test-dir test_data
```

### 注册新用户（命令行）

```bash
python scripts/register_user.py --username alice --age 25 --occupation 4
# 输出：Registered user_id=900001 username=alice age=25 occupation=4
```

---

## 单元测试

```bash
# 运行全部测试
python -m unittest discover -s tests

# 运行单个测试文件
python -m tests.test_cold_start_recommender
python -m tests.test_evaluate_pipeline
python -m tests.test_pipeline_timing
```

---

## 项目结构

```
recommend/
├── recommender_pipeline.py       # 主链路入口 ← 从这里开始读
├── evaluate_pipeline.py          # 端到端评估
├── server.py                     # FastAPI 后端
├── requirements.txt              # 依赖清单
│
├── recall/                       # 召回：双塔模型
│   ├── two_tower.py              # 模型结构 + 训练 + 推理
│   └── evaluate.py               # 召回评估，自动选最优 epoch
│
├── rough_rank/                   # 粗排：三塔模型
│   ├── model.py
│   ├── train.py
│   └── inference.py
│
├── fine_rank/                    # 精排：MMoE 模型
│   ├── model.py
│   ├── train.py
│   └── inference.py
│
├── cold_start/                   # 冷启动
├── database/                     # MySQL 数据层（可选）
│
├── scripts/                      # 工具脚本
│   ├── init_mysql_schema.py      # 初始化数据库表结构
│   ├── import_movielens_to_mysql.py  # 导入 MovieLens 数据到 MySQL
│   └── register_user.py          # 命令行注册新用户
│
├── static/
│   └── index.html                # Web 前端（单文件）
│
├── models/                       # 训练后生成的模型权重
│   ├── recall/
│   ├── rough_rank/
│   └── fine_rank/
│
├── data/                         # 原始 MovieLens 数据
├── train_data/                   # 训练集
├── test_data/                    # 测试集
└── tests/                        # 单元测试
```
