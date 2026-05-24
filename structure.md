# 最简推荐系统框架

本文档基于当前项目的完整推荐链路进行简化，目标是做出一个能跑通、易理解、方便后续扩展的最简推荐系统。

当前项目完整链路是：

```text
数据准备 -> 召回 -> 粗排 -> 精排 -> 重排 -> API服务
```

最简版本建议保留同样的工程思想，但先把复杂模型降级为轻量实现：

```text
CSV数据 -> 数据预处理 -> 召回 -> 简单排序 -> 简单重排 -> 推荐接口
```

## 1. 项目目录框架

建议新建一个轻量目录，例如 `minimal_recommender/`，避免影响当前完整项目。

```text
minimal_recommender/
  config.py              # 全局配置
  data_loader.py         # 数据读取
  preprocess.py          # 数据预处理
  recall.py              # 召回模块
  ranking.py             # 排序模块
  rerank.py              # 重排模块
  service.py             # 推荐主服务
  api.py                 # FastAPI接口，可选
  metrics.py             # 简单评估指标，可选
  run_demo.py            # 本地演示入口
```

需要实现的内容：

| 文件 | 需要实现的内容 |
| --- | --- |
| `config.py` | 数据路径、召回数量、最终推荐数量、热门权重、相似度参数 |
| `data_loader.py` | 读取 `users_new.csv`、`items_new.csv`、`interactions_new.csv` |
| `preprocess.py` | 清洗空值、拆分类目和关键词、构建用户历史、物品索引、热门榜 |
| `recall.py` | 实现最少 2 个召回通道：热门召回、ItemCF召回 |
| `ranking.py` | 对召回候选进行简单打分和排序 |
| `rerank.py` | 去除已交互物品，控制类目重复，保留多样性 |
| `service.py` | 编排完整推荐流程，对外暴露 `recommend()` 方法 |
| `api.py` | 可选，用 FastAPI 封装 `/recommend` 接口 |
| `metrics.py` | 可选，实现 Precision@K、Recall@K |
| `run_demo.py` | 加载数据并对指定用户输出推荐结果 |

## 2. 数据层

对应当前项目中的：

- `data/users_new.csv`
- `data/items_new.csv`
- `data/interactions_new.csv`
- `entities.py`
- `feature_processor.py`

最简版本不需要先定义复杂实体类，可以先用 `pandas.DataFrame` 和字典完成。

### 需要实现的数据读取

读取三张表：

| 数据表 | 关键字段 | 用途 |
| --- | --- | --- |
| 用户表 | `user_id`, `user_categories`, `user_keywords`, `age`, `gender` | 用户画像 |
| 物品表 | `item_id`, `item_categories`, `item_keywords`, `price`, `create_time` | 物品画像 |
| 交互表 | `user_id`, `item_id`, `rating`, `click`, `cart`, `buy`, `forward` | 用户行为 |

### 需要实现的预处理

1. 将用户和物品的类目、关键词从字符串拆成列表。
2. 构建 `user_history`：

```python
{
    "U00001": {"I0001", "I0002"},
    "U00002": {"I0003"}
}
```

3. 构建 `user_item_rating`：

```python
{
    "U00001": {"I0001": 1, "I0002": 3}
}
```

4. 构建 `item_info`：

```python
{
    "I0001": {
        "category": "文学",
        "keywords": ["随笔", "外国文学"],
        "price": 87.45
    }
}
```

5. 构建热门榜 `popular_items`，按交互次数或加权行为分排序。

推荐使用的行为权重：

```text
score = click * 1 + cart * 3 + buy * 5 + forward * 2
```

## 3. 召回层

对应当前项目中的：

- `recall.py`
- `swing.py`
- `twin_towers_model.py`
- `LightGCN.py`

完整项目中召回通道很多，最简版本建议先保留两个：

```text
热门召回 + ItemCF召回
```

这样既能解决新用户冷启动，也能体现个性化推荐。

### 3.1 热门召回

用途：

- 新用户没有历史行为时兜底。
- ItemCF 召回数量不足时补齐。

需要实现：

1. 根据交互表统计每个物品的热度分。
2. 按热度降序得到全局热门列表。
3. 推荐时过滤用户已交互物品。
4. 返回 TopK 物品 ID。

输入：

```text
user_id, top_k
```

输出：

```text
[item_id_1, item_id_2, ...]
```

### 3.2 ItemCF召回

用途：

- 根据用户历史交互物品，找相似物品。
- 是最简个性化推荐的核心。

需要实现：

1. 根据交互表构建用户到物品的倒排表。
2. 计算物品共现次数。
3. 计算物品相似度。
4. 为每个物品保留最相似的 TopN 物品。
5. 推荐时遍历用户历史物品，将相似物品分数累加。
6. 过滤已交互物品。

推荐相似度公式：

```text
sim(i, j) = co_count(i, j) / sqrt(count(i) * count(j))
```

推荐分数：

```text
score(j) = sum rating(user, i) * sim(i, j)
```

### 3.3 召回合并

需要实现：

1. 分别调用 ItemCF 和热门召回。
2. 使用字典合并候选物品分数。
3. 如果 ItemCF 分数存在，优先保留个性化分数。
4. 热门召回只作为补充，不覆盖 ItemCF 高分。

推荐输出格式：

```python
{
    "I0001": 0.91,
    "I0002": 0.73,
    "I0003": 0.12
}
```

## 4. 排序层

对应当前项目中的：

- `rough_ranking.py`
- `fine_ranking.py`
- `DCN.py`
- `models/`

最简版本不需要三塔、DCN、MMoE，可以先用规则排序。

### 需要实现的排序特征

对每个候选物品计算以下特征：

| 特征 | 来源 | 说明 |
| --- | --- | --- |
| `recall_score` | 召回层 | ItemCF 或热门召回得分 |
| `hot_score` | 交互表 | 物品近期热度 |
| `category_match` | 用户表 + 物品表 | 用户偏好类目是否命中 |
| `keyword_match` | 用户表 + 物品表 | 用户偏好关键词命中数量 |
| `fresh_score` | 物品表 | 新物品适当加分，可选 |

### 推荐排序公式

先用可解释规则：

```text
final_score =
  0.50 * recall_score +
  0.25 * hot_score +
  0.15 * category_match +
  0.10 * keyword_match
```

需要实现：

1. 对不同分数做归一化，避免热度分过大。
2. 对每个候选物品计算最终分数。
3. 按 `final_score` 降序排序。
4. 输出排序后的候选列表。

输出格式：

```python
[
    {"item_id": "I0001", "score": 0.92},
    {"item_id": "I0002", "score": 0.87}
]
```

## 5. 重排层

对应当前项目中的：

- `rearrangement.py`

完整项目使用 CLIP 内容特征 + MMR。最简版本可以先做规则重排，控制同类目物品连续出现。

### 需要实现

1. 过滤用户已交互物品。
2. 控制推荐结果中同一类目数量。
3. 避免连续多个相同类目的物品。
4. 如果重排后数量不足，从排序列表继续补齐。

简单策略：

```text
从高分到低分遍历候选：
  如果该物品未被用户看过，并且该类目未超过上限，则加入结果
  直到达到 top_k
```

建议配置：

```python
MAX_PER_CATEGORY = 3
FINAL_TOP_K = 20
```

## 6. 推荐服务层

对应当前项目中的：

- `main.py`
- `interface/recommender_system.py`

最简版本建议用一个 `MinimalRecommenderSystem` 类把数据、召回、排序、重排串起来。

### 需要实现的类

```python
class MinimalRecommenderSystem:
    def __init__(self):
        self.load_data()
        self.preprocess()
        self.fit()

    def fit(self):
        # 构建热门榜、ItemCF相似度等离线数据
        pass

    def recommend(self, user_id: str, top_k: int = 20):
        # 召回 -> 排序 -> 重排 -> 返回物品ID列表
        pass
```

### 推荐流程

```text
1. 接收 user_id
2. 查询用户画像和历史行为
3. 召回候选物品
4. 对候选物品打分排序
5. 重排增强多样性
6. 返回最终推荐列表
```

### 返回格式

本地函数可以先返回：

```python
["I0001", "I0002", "I0003"]
```

如果后续要做 API，建议返回：

```json
{
  "status": "success",
  "user_id": "U00001",
  "recommendations": ["I0001", "I0002", "I0003"]
}
```

## 7. API接口层

对应当前项目中的：

- `interface/main.py`

最简版本 API 可以只保留一个接口。

### POST /recommend

请求体：

```json
{
  "user_id": "U00001",
  "top_k": 20
}
```

响应体：

```json
{
  "status": "success",
  "user_id": "U00001",
  "recommendations": ["I0001", "I0002", "I0003"]
}
```

需要实现：

1. 参数校验。
2. 用户不存在时返回热门推荐或错误信息。
3. 调用 `MinimalRecommenderSystem.recommend()`。
4. 返回 JSON。

## 8. 评估层

当前完整项目偏工程实现，最简版本建议补一个简单离线评估，方便确认推荐效果。

### 需要实现

1. 按用户切分训练集和测试集。
2. 用训练集构建推荐器。
3. 对测试集用户生成 TopK 推荐。
4. 计算 Precision@K、Recall@K。

指标公式：

```text
Precision@K = 推荐命中的物品数 / K
Recall@K = 推荐命中的物品数 / 用户真实喜欢的物品数
```

## 9. 最小可运行版本优先级

建议按下面顺序实现，保证每一步都能看到结果。

### 第一阶段：跑通推荐主链路

需要完成：

1. `data_loader.py`
2. `preprocess.py`
3. `recall.py` 中的热门召回
4. `service.py`
5. `run_demo.py`

阶段目标：

```text
输入 user_id，输出热门推荐列表
```

### 第二阶段：加入个性化

需要完成：

1. `recall.py` 中的 ItemCF
2. 召回结果合并
3. 已交互物品过滤

阶段目标：

```text
老用户看到与历史行为相关的推荐，新用户看到热门推荐
```

### 第三阶段：加入排序

需要完成：

1. `ranking.py`
2. 类目命中、关键词命中、热度分融合
3. 排序分归一化

阶段目标：

```text
推荐结果不只依赖相似度，还能结合用户画像和物品热度
```

### 第四阶段：加入重排

需要完成：

1. `rerank.py`
2. 类目多样性控制
3. 最终 TopK 截断

阶段目标：

```text
最终推荐结果更丰富，避免同一类目刷屏
```

### 第五阶段：封装 API

需要完成：

1. `api.py`
2. `/recommend` 接口
3. 请求参数校验
4. 异常处理

阶段目标：

```text
可以通过 HTTP 请求获取推荐结果
```

## 10. 与当前完整项目的对应关系

| 完整项目模块 | 最简版本模块 | 简化方式 |
| --- | --- | --- |
| `main.py` / `interface/recommender_system.py` | `service.py` | 保留编排逻辑，减少模型加载 |
| `feature_processor.py` | `preprocess.py` | 只做基础清洗和索引构建 |
| `recall.py` | `recall.py` | 保留热门召回和 ItemCF |
| `rough_ranking.py` | `ranking.py` | 用规则分替代三塔模型 |
| `fine_ranking.py` | 暂不实现 | 等最简版跑通后再扩展 |
| `rearrangement.py` | `rerank.py` | 用类目规则替代 CLIP + MMR |
| `interface/main.py` | `api.py` | 只保留 `/recommend` |
| `model_weights/` | 暂不依赖 | 最简版不加载深度模型权重 |

## 11. 推荐开发验收标准

最简系统完成后，应满足：

1. 可以读取当前 `data/` 目录下的三份 CSV。
2. 可以输入一个用户 ID，返回 TopK 推荐物品 ID。
3. 老用户推荐结果会过滤已交互物品。
4. 新用户或无历史用户有热门推荐兜底。
5. 推荐结果按分数排序。
6. 推荐结果不会被单一类目完全占满。
7. 代码不依赖 GPU、不依赖模型权重、不依赖数据库。
8. 后续可以平滑扩展到当前项目的完整链路。

## 12. 后续扩展方向

最简版本跑通后，可以按当前项目的成熟模块逐步增强：

1. 增加 UserCF 或 Swing 召回。
2. 增加基于类目和关键词的冷启动召回。
3. 用 LR、GBDT 或简单 MLP 替换规则排序。
4. 引入双塔模型做向量召回。
5. 引入 DCN 或 MMoE 做精排。
6. 引入 MMR 做真正的多样性重排。
7. 用 FastAPI 部署成推荐服务。
8. 接入数据库，实现离线更新和在线推荐分离。
