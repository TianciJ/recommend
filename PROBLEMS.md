# 生产部署问题清单

本文档记录当前推荐系统部署到真实服务器时会遇到的工程化问题，按严重程度分级，每个问题标注了具体文件和行号。

---

## 🔴 严重问题（上线前必须修复，否则会直接崩溃）

---

### P1 — 数据库没有连接池

**文件：** `database/mysql_client.py`

**现状：** 每次数据库操作都新建一个 TCP 连接，用完就关。

```python
# mysql_client.py：每次调用都重新握手认证
return pymysql.connect(host=config.host, port=config.port, ...)
```

**后果：**
- 每次连接需要 TCP 握手 + MySQL 认证，额外增加 50-100ms 延迟
- 30 个并发用户就会撑爆 MySQL 的 `max_connections` 限制，报 `Too many connections`
- 连接数随并发线性增长，无上限

**修复方案：** 引入连接池（`DBUtils.PooledDB` 或 `SQLAlchemy`），初始化时建好 5-10 个复用连接。

---

### P2 — 数据库查询无 LIMIT，全量加载进内存

**文件：** `database/dataset_repository.py`、`database/user_repository.py`、`recommender_pipeline.py`

**现状：** 多处查询没有分页限制，启动时会把整张表拉进内存。

```python
# dataset_repository.py：拉全部 ratings，无 LIMIT
SELECT user_id, movie_id, rating, rating_timestamp
FROM ratings WHERE split_name = %s
ORDER BY user_id, movie_id, rating_timestamp

# user_repository.py：拉全部用户，无 LIMIT
SELECT user_id, age, occupation FROM users

# recommender_pipeline.py 第 261-262 行：Reranker 把所有用户历史加载进内存
self.user_seen_movies = load_user_seen_movies(ratings=ratings)
self.movie_genres = load_movie_genres(movies=movies)
```

**后果：**
- MovieLens 1M 有 100 万条评分，真实业务数据 1000 万条时，启动就要等 30 秒以上
- Pipeline 初始化时 `list_ratings()` 和 Reranker 各拉一次，共拉两次全量数据
- 10 个 worker 进程 × 每份 ~100MB = 1GB 仅用于"已看电影过滤"
- 用户量到百万级时直接 OOM

**修复方案：**
- `list_ratings()` 改为按 `user_id` 分页拉取
- Reranker 的"过滤已看"改为实时查库（`SELECT movie_id FROM ratings WHERE user_id = ?`，加索引后很快），或用 Redis 缓存热用户历史
- `list_user_profiles()` 加 `LIMIT`，冷启动分段构建统计

---

### P3 — 精排特征 Key 命名不一致，运行时必然 KeyError

**文件：** `fine_rank/inference.py` 第 73-75 行 vs `rough_rank/inference.py` 第 117-119 行

**现状：** 两个模块训练时存的 feature_info key 命名不统一，精排推理时用了错误的 key。

```python
# fine_rank/inference.py 第 73-75 行（当前代码）
gender_indexes.append(user_feature["gender"])
age_indexes.append(user_feature["age"])
occupation_indexes.append(user_feature["occupation"])

# rough_rank/inference.py 第 117-119 行（正确写法）
gender_indexes.append(user_feature["gender_index"])
age_indexes.append(user_feature["age_index"])
occupation_indexes.append(user_feature["occupation_index"])
```

**后果：** 若两个模块的 checkpoint feature_info 格式不一致，精排阶段直接 `KeyError` 崩溃，整条推荐链路中断。

**修复方案：** 统一所有模块的 feature_info 命名，全部使用 `gender_index`、`age_index`、`occupation_index`，训练脚本和推理脚本同步修改。

---

## 🟠 高优先级问题（影响稳定性，应在上线后短期修复）

---

### P4 — 所有错误用 `print()` 输出，线上无法监控

**涉及文件：** `recommender_pipeline.py` 第 146、259 行；`recall/two_tower.py` 第 473 行；及其他共 10+ 处

**现状：**

```python
print(f"MySQL user profile lookup failed; using cold-start fallback: {error}")
print("这个用户没有出现在训练集中，暂时无法用双塔召回")
```

**后果：**
- 没有日志级别（ERROR / WARN / INFO），所有输出混在一起
- 无法接入 ELK / Datadog / 云日志平台
- 无法统计错误率、触发告警
- 没有时间戳和 request_id，出问题无法定位是哪个用户的请求
- 多进程时 print 输出交错，难以阅读

**修复方案：** 全部替换为 `logging` 模块，配置 JSON 格式输出，每条日志带上 `user_id`、`request_id`、时间戳字段，接入日志收集平台。

---

### P5 — 召回推理每次请求都重算所有电影向量

**文件：** `recall/two_tower.py` 第 481-515 行

**现状：** 每次为某个用户召回时，都重新为所有电影建 Tensor 并过一遍物品塔。

```python
# 每次推荐都执行：3000 部电影 × Python 循环逐一构建
genre_vectors = []
for movie_index in range(movie_count):       # 逐一 Python 循环
    movie_id = index_to_movie_id[movie_index]
    genre_vectors.append(movie_features[movie_id]["genre_vector"])
genre_tensor = torch.tensor(genre_vectors, ...)  # 每次重新建 Tensor
```

**后果：**
- 电影侧向量在模型不变时是静态的，重复计算是纯浪费
- 100 并发请求时，内存中同时存在 100 份完全相同的 genre_tensor
- GPU 设备上内存压力更大，容易 OOM

**修复方案：** 在 `TwoTowerRecaller.__init__` 里预计算并缓存所有电影的向量，推理时直接用 `torch.matmul(user_vector, all_movie_vectors.T)` 做矩阵乘法，一次算完所有相似度。耗时从 O(电影数) 降到接近 O(1)。

---

### P6 — 模型无法热更新，上线必须重启

**文件：** `recommender_pipeline.py` 第 13-15 行

**现状：** 模型在 Pipeline 初始化时加载一次，之后固定不变。

```python
self.recaller = build_recaller()
self.rough_ranker = build_rough_ranker()
self.fine_ranker = build_fine_ranker()
```

**后果：**
- 重新训练出更好的模型，必须重启服务才能生效，会有几秒到几十秒的服务中断
- 无法做 A/B 测试（用 10% 流量跑新模型验证效果后再全量）
- 无法按用户分桶走不同版本模型

**修复方案：**
- 模型路径改为从环境变量读取，方便切换
- 支持通过管理接口触发热加载（用锁保护切换过程）
- 或采用蓝绿部署：启动新进程加载新模型，流量切换完成后停旧进程

---

### P7 — 冷启动推荐器数据一次性构建，永不刷新

**文件：** `cold_start/cold_start_recommender.py` 第 38 行

**现状：** 电影热度统计、分群统计在初始化时算一次，此后固定不变。

```python
self._build_rating_statistics(ratings_path=ratings_path, ratings=ratings)
```

**后果：**
- 新上线的电影热度为 0，永远不会出现在冷启动推荐里
- 用户行为数据更新后，推荐结果不跟随变化
- 只能靠重启服务刷新，运营侧无法自助触发

**修复方案：** 增加定时刷新机制（每天凌晨重建统计），或接入实时热度服务（Redis sorted set 按热度排序，实时更新）。

---

## 🟡 中优先级问题（影响可扩展性，中期规划解决）

---

### P8 — 没有 API 服务层和健康检查接口

**现状：** 系统只能通过脚本调用，无法作为网络服务对外暴露。

**后果：**
- 无法接入负载均衡器（Nginx / AWS ALB）
- 没有 `/health` 接口，负载均衡器无法判断实例是否正常，模型加载失败也无法自动摘除
- 无法横向扩容（多实例）
- 无法设置请求超时，单个慢请求会拖垮整个进程

**修复方案：** 用 FastAPI 包一层，提供以下接口：

```
POST /recommend          推荐接口，接收 user_id 返回推荐列表
GET  /health             返回模型是否加载完成、DB 是否可达
GET  /metrics            暴露 Prometheus 格式指标（延迟 P50/P99、错误率）
```

---

### P9 — 关键配置全部硬编码在代码里

**涉及位置：**
- `recall/two_tower.py`：`MODEL_PATH = MODEL_DIR / "two_tower.pt"`
- `rough_rank/inference.py`：`MODEL_PATH = MODEL_DIR / "three_tower.pt"`
- `fine_rank/inference.py`：`DEFAULT_MODEL_PATH = BASE_DIR / "models" / "fine_rank" / "mmoe_epoch_6.pt"`
- 各处召回数量、排序截断数量均为魔法数字

**后果：**
- 换一个 epoch 的精排模型需要改代码重新部署
- 开发 / 测试 / 生产环境的模型路径不同，只能靠代码区分
- 无法通过 Docker / K8s 的环境变量注入配置
- 召回数量等超参无法在不重新部署的情况下调整

**修复方案：** 关键配置改为从环境变量或配置文件读取，提供 `.env.example` 说明所有可配置项。示例：

```
RECALL_MODEL_PATH=models/recall/two_tower.pt
ROUGH_RANK_MODEL_PATH=models/rough_rank/three_tower.pt
FINE_RANK_MODEL_PATH=models/fine_rank/mmoe_epoch_6.pt
RECALL_SIZE=300
ROUGH_RANK_SIZE=100
FINE_RANK_SIZE=50
```

---

### P10 — 没有请求级链路追踪

**现状：** 日志（即 print）没有 request_id，无法把一次推荐请求在召回、粗排、精排各阶段的日志串联起来。

**后果：**
- 用户反馈"推荐结果不对"，无法定位是哪个阶段出了问题
- 多用户并发时日志乱序，无法区分哪条日志属于哪个请求
- 性能瓶颈在哪个阶段无法定位

**修复方案：** 每个请求生成一个 `request_id`（UUID），作为参数或 context 变量贯穿整条链路，写入每条日志。

---

## 总览

| 编号 | 问题 | 文件 | 严重程度 | 主要后果 |
|------|------|------|----------|----------|
| P1 | 无数据库连接池 | `database/mysql_client.py` | 🔴 严重 | 并发 >30 直接崩溃 |
| P2 | 全量查询无 LIMIT | `dataset_repository.py` / `pipeline.py` | 🔴 严重 | 启动 OOM，耗时 30s+ |
| P3 | 精排 feature key 不一致 | `fine_rank/inference.py` | 🔴 严重 | 运行时 KeyError 崩溃 |
| P4 | print 替代日志 | 多处 | 🟠 高 | 线上错误无法监控 |
| P5 | 召回每次重算电影向量 | `recall/two_tower.py` | 🟠 高 | 高并发 OOM，性能差 |
| P6 | 模型不能热更新 | `recommender_pipeline.py` | 🟠 高 | 上线必须停服重启 |
| P7 | 冷启动数据不刷新 | `cold_start_recommender.py` | 🟠 高 | 新电影永远不出现 |
| P8 | 没有 API 层和健康检查 | 无 | 🟡 中 | 无法接入负载均衡 |
| P9 | 配置硬编码 | 多处 | 🟡 中 | 无法多环境部署 |
| P10 | 无请求链路追踪 | 多处 | 🟡 中 | 出问题无法定位 |
