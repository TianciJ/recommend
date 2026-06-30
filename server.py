# 推荐系统 Web 服务入口
# 启动：python server.py
# 访问：http://localhost:8000
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database.mysql_client import get_mysql_config_from_env
from database.dataset_repository import MysqlDatasetRepository
from recommender_pipeline import RecommenderPipeline, build_user_profile_repository, build_dataset_repository


# ---------- 请求/响应模型 ----------

class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = Field(default=20, ge=1, le=100)
    recall_size: int = Field(default=300, ge=10, le=1000)
    rough_rank_size: int = Field(default=100, ge=10, le=500)
    fine_rank_size: int = Field(default=50, ge=10, le=200)
    age: Optional[int] = Field(default=None, ge=1, le=120)
    occupation: Optional[int] = Field(default=None, ge=0, le=20)


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    age: int = Field(ge=1, le=120)
    occupation: int = Field(ge=0, le=20)


class RatingRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: int = Field(ge=1, le=5)


# ---------- 应用启动/关闭 ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 模型加载耗时较长，启动时加载一次，后续所有请求复用
    logger.info("正在加载推荐模型，请稍候...")
    app.state.pipeline = RecommenderPipeline()
    app.state.user_repo = build_user_profile_repository()
    app.state.dataset_repo = build_dataset_repository()
    app.state.mysql_available = app.state.user_repo is not None
    app.state.retrain_status = {"running": False, "last_result": None}
    logger.info("模型加载完成。MySQL 可用: %s", app.state.mysql_available)
    yield


app = FastAPI(title="MovieLens 推荐系统", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------- 工具函数 ----------

def format_items(items: list) -> list:
    """把 pipeline 返回的 item 列表统一格式化为 API 响应格式。"""
    result = []
    for rank, item in enumerate(items, start=1):
        mid = item.get("movie_id", item.get("item_id"))
        recall = item.get("recall_score")
        rough = item.get("rough_rank_score")
        fine = item.get("fine_rank_score")
        cold = item.get("cold_start_score")
        result.append({
            "rank": rank,
            "movie_id": mid,
            "title": item.get("title", ""),
            "genre": item.get("rerank_primary_genre", ""),
            "recall_score": round(recall, 4) if recall is not None else None,
            "rough_rank_score": round(rough, 4) if rough is not None else None,
            "fine_rank_score": round(fine, 4) if fine is not None else None,
            "cold_start_score": round(cold, 4) if cold is not None else None,
            "cold_start_source": item.get("cold_start_source"),
        })
    return result


def require_mysql(request: Request):
    """MySQL 未配置时抛出 503。"""
    if not request.app.state.mysql_available:
        raise HTTPException(status_code=503, detail="MySQL 未配置，该功能不可用。")


# ---------- 路由 ----------

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/static/index.html")


@app.get("/api/status")
def get_status(request: Request):
    """返回服务状态，前端据此决定是否显示注册表单。"""
    return {
        "pipeline_ready": True,
        "mysql_available": request.app.state.mysql_available,
    }


@app.post("/api/recommend")
def recommend(body: RecommendRequest, request: Request):
    """核心推荐接口。user_id 不在训练集时自动走冷启动。"""
    pipeline: RecommenderPipeline = request.app.state.pipeline
    try:
        items = pipeline.recommend(
            user_id=body.user_id,
            top_k=body.top_k,
            recall_size=body.recall_size,
            rough_rank_size=body.rough_rank_size,
            fine_rank_size=body.fine_rank_size,
            age=body.age,
            occupation=body.occupation,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    formatted = format_items(items)
    is_cold_start = bool(formatted) and formatted[0]["cold_start_score"] is not None

    return {
        "user_id": body.user_id,
        "count": len(formatted),
        "is_cold_start": is_cold_start,
        "items": formatted,
    }


@app.post("/api/users", status_code=201)
def create_user(body: CreateUserRequest, request: Request):
    """注册新用户。需要配置 MySQL。"""
    require_mysql(request)
    repo = request.app.state.user_repo
    try:
        user_id = repo.create_user(
            username=body.username,
            age=body.age,
            occupation=body.occupation,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # 用户名重复等数据库约束错误
        msg = str(e)
        if "Duplicate" in msg or "duplicate" in msg:
            raise HTTPException(status_code=409, detail=f"用户名 '{body.username}' 已存在")
        raise HTTPException(status_code=500, detail=msg)

    return {"user_id": user_id, "username": body.username}


@app.get("/api/users/{user_id}")
def get_user(user_id: int, request: Request):
    """查询用户画像。需要配置 MySQL。"""
    require_mysql(request)
    profile = request.app.state.user_repo.get_user_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 不存在")
    return profile


@app.post("/api/ratings", status_code=201)
def add_rating(body: RatingRequest, request: Request):
    """提交用户对电影的评分（1-5 星）。需要配置 MySQL。"""
    require_mysql(request)
    repo: MysqlDatasetRepository = request.app.state.dataset_repo
    try:
        repo.add_rating(
            user_id=body.user_id,
            movie_id=body.movie_id,
            rating=body.rating,
            timestamp=int(time.time()),
            split="train",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"user_id": body.user_id, "movie_id": body.movie_id, "rating": body.rating}


@app.post("/api/retrain")
def retrain(request: Request):
    """在后台重新训练所有模型，训练完后热替换 pipeline，不需要重启服务。"""
    require_mysql(request)
    status = request.app.state.retrain_status
    if status["running"]:
        return {"status": "running", "message": "训练正在进行中，请稍候"}

    def run_training():
        status["running"] = True
        status["last_result"] = None
        try:
            logger.info("开始重新训练模型...")
            _retrain_all_models()
            # 训练完成，热替换 pipeline（原子赋值，正在处理的请求不受影响）
            request.app.state.pipeline = RecommenderPipeline()
            status["last_result"] = {"success": True, "message": "训练完成，模型已热更新"}
            logger.info("模型热更新完成")
        except Exception as e:
            status["last_result"] = {"success": False, "message": str(e)}
            logger.error("训练失败: %s", e)
        finally:
            status["running"] = False

    threading.Thread(target=run_training, daemon=True).start()
    return {"status": "started", "message": "训练已在后台启动，可轮询 /api/retrain/status 查看进度"}


@app.get("/api/retrain/status")
def retrain_status(request: Request):
    """查询训练进度。"""
    status = request.app.state.retrain_status
    return {
        "running": status["running"],
        "last_result": status["last_result"],
    }


def _retrain_all_models():
    """依次重训召回、粗排、精排三个模型（epochs 都取 3）。"""
    from recall.two_tower import train_model as train_recall
    from rough_rank.train import train_model as train_rough
    from fine_rank.train import train_model as train_fine
    logger.info("训练召回模型...")
    train_recall(epochs=3)
    logger.info("训练粗排模型...")
    train_rough(epochs=3)
    logger.info("训练精排模型...")
    train_fine(epochs=3)


# ---------- 启动 ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
