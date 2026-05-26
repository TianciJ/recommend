# MySQL 用户画像接入设计

## 背景

当前推荐系统主要依赖离线文件：

```text
train_data/users.dat
train_data/ratings.dat
data/movies.dat
```

冷启动 V1 已经支持通过 `age` 和 `occupation` 做画像分群推荐，但调用方仍然需要手动传入：

```python
pipeline.recommend(user_id=900001, age=25, occupation=4)
```

如果后续要实现用户注册，注册信息应该进入数据库。推荐系统在新用户请求推荐时，应能根据 `user_id` 自动读取注册画像，再触发冷启动推荐。

## 目标

1. 使用 MySQL 存储新注册用户画像。
2. 不引入复杂后端服务，先通过脚本模拟注册入库。
3. 推荐链路可以通过 `user_id` 自动查询 `age` 和 `occupation`。
4. 保持当前 `pipeline.recommend()` 的调用方式兼容。
5. 数据库不可用或用户不存在时，推荐系统不崩溃，继续走已有兜底逻辑。
6. 代码边界清晰，后续可以平滑接入 FastAPI 或其他后端。

## 非目标

1. 本阶段不做登录、鉴权、密码管理。
2. 本阶段不做前端注册页面。
3. 本阶段不做 HTTP API 服务。
4. 本阶段不把 MovieLens 全量训练数据迁移到 MySQL。
5. 本阶段不改变召回、粗排、精排模型结构。

## 推荐方案

采用“轻量 MySQL 用户画像仓库”方案：

```text
MySQL users 表
  -> UserProfileRepository
  -> RecommenderPipeline
  -> ColdStartRecommender
```

注册用户先通过命令行脚本写入 MySQL。推荐系统只负责读取用户画像，不负责复杂后端流程。

## MySQL 表设计

第一版只保留冷启动需要的核心字段：

```sql
CREATE TABLE IF NOT EXISTS users (
    user_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(64) NOT NULL UNIQUE,
    age INT NOT NULL,
    occupation INT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) AUTO_INCREMENT = 900001;
```

字段说明：

```text
user_id       新注册用户 ID，使用较大的自增起点，避免和 MovieLens 老用户 ID 混淆
username      注册用户名，第一版只做唯一标识，不做登录认证
age           用户年龄，冷启动分群特征
occupation    用户职业，冷启动分群特征
created_at    注册时间
updated_at    最近更新时间
```

`AUTO_INCREMENT = 900001` 的目的是让新注册用户明显区别于训练集中已有用户，便于调试和展示冷启动效果。

## 配置方式

数据库连接信息通过环境变量读取，不写死在代码里：

```text
MYSQL_HOST
MYSQL_PORT
MYSQL_USER
MYSQL_PASSWORD
MYSQL_DATABASE
MYSQL_CONNECT_TIMEOUT
```

默认值建议：

```text
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=recommend
MYSQL_CONNECT_TIMEOUT=3
```

`MYSQL_USER` 和 `MYSQL_PASSWORD` 必须由本地环境提供，避免把密码提交到仓库。

## 模块设计

新增数据库相关模块：

```text
database/
  __init__.py
  mysql_client.py
  user_repository.py

scripts/
  init_mysql_schema.py
  register_user.py
```

### mysql_client.py

职责：

1. 从环境变量读取 MySQL 配置。
2. 创建 MySQL 连接。
3. 隐藏具体连接库，避免业务代码直接依赖连接细节。

第一版建议使用 `pymysql`，因为它是纯 Python 依赖，安装和跨平台运行比较轻。

### user_repository.py

核心接口：

```python
class UserProfileRepository:
    def get_user_profile(self, user_id):
        ...

    def create_user(self, username, age, occupation):
        ...
```

返回画像格式：

```python
{
    "user_id": 900001,
    "age": 25,
    "occupation": 4,
}
```

如果用户不存在，返回 `None`。

### init_mysql_schema.py

用于初始化数据库表：

```powershell
python scripts/init_mysql_schema.py
```

职责：

1. 连接 MySQL。
2. 创建 `users` 表。
3. 打印初始化结果。

### register_user.py

用于模拟注册：

```powershell
python scripts/register_user.py --username alice --age 25 --occupation 4
```

职责：

1. 校验 `age` 和 `occupation`。
2. 写入 `users` 表。
3. 打印新用户 `user_id`。

## Pipeline 接入方式

`RecommenderPipeline` 增加一个可注入的用户画像仓库：

```python
pipeline = RecommenderPipeline(user_profile_repository=repo)
```

默认情况下，如果 MySQL 配置存在，则创建 MySQL repository；如果配置不存在，则不启用数据库读取。

推荐调用保持兼容：

```python
pipeline.recommend(user_id=900001)
```

内部流程：

```text
1. 调用 recommend(user_id)
2. 先走双塔召回
3. 如果召回为空，准备进入冷启动
4. 如果调用方没有传 age / occupation，则从 MySQL 查询用户画像
5. 如果查到画像，使用 age / occupation 做分群冷启动
6. 如果查不到画像，继续使用当前全局热门兜底
```

显式传参优先级最高：

```python
pipeline.recommend(user_id=900001, age=35, occupation=7)
```

如果调用方显式传了 `age` 或 `occupation`，则优先使用调用方传入值；缺失的字段再尝试从数据库补齐。

## 错误处理

推荐请求不应该因为数据库问题整体失败。

第一版错误处理策略：

1. MySQL 未配置：跳过数据库读取，继续已有推荐/冷启动逻辑。
2. MySQL 连接失败：记录错误提示，冷启动退化为传入画像或全局热门。
3. 用户不存在：返回 `None`，冷启动退化为传入画像或全局热门。
4. 注册用户名重复：注册脚本返回清晰错误。
5. `age` 或 `occupation` 非法：注册脚本拒绝写入。

这能保证老用户推荐链路不受数据库影响，新用户画像缺失时也不会崩溃。

## 测试计划

单元测试优先使用 fake repository，不依赖真实 MySQL：

1. `recommend(user_id)` 在冷启动时可以从 repository 补齐 `age` 和 `occupation`。
2. 显式传入的 `age` 和 `occupation` 优先于数据库值。
3. repository 返回 `None` 时，冷启动继续走全局热门兜底。
4. repository 抛异常时，推荐请求不崩溃。
5. `UserProfileRepository.create_user()` 能返回新用户 ID。
6. `UserProfileRepository.get_user_profile()` 能把数据库行转换成标准画像字典。

真实 MySQL 的集成测试不放进默认单元测试，避免本地没有 MySQL 时测试失败。后续可以增加手动验证命令：

```powershell
python scripts/init_mysql_schema.py
python scripts/register_user.py --username alice --age 25 --occupation 4
python -m recommender_pipeline
```

## README 更新

README 需要增加：

1. MySQL 环境变量说明。
2. 初始化表命令。
3. 模拟注册命令。
4. 注册用户推荐命令。
5. 数据库未配置时的降级行为。

示例：

```powershell
set MYSQL_HOST=localhost
set MYSQL_PORT=3306
set MYSQL_USER=root
set MYSQL_PASSWORD=your_password
set MYSQL_DATABASE=recommend

python scripts/init_mysql_schema.py
python scripts/register_user.py --username alice --age 25 --occupation 4
```

## 后续升级方向

### V2：接入 FastAPI

增加接口：

```text
POST /register
GET /recommendations/{user_id}
```

推荐系统复用当前 repository，不需要重写冷启动逻辑。

### V3：用户行为入库

增加行为表：

```text
user_events
ratings
```

用于记录点击、收藏、评分等短期行为，为冷启动后的兴趣更新做准备。

### V4：用户画像更新

根据实时行为更新用户偏好，例如最近喜欢的电影类型、最近高评分电影、活跃度等。

## 验收标准

1. 可以通过脚本创建 MySQL `users` 表。
2. 可以通过脚本注册新用户，并得到 `user_id`。
3. `pipeline.recommend(user_id)` 对新注册用户可以自动读取 `age` 和 `occupation`。
4. 数据库不可用时，推荐请求不崩溃。
5. 老用户原有推荐链路不受影响。
6. README 有完整的 MySQL 配置和使用命令。
