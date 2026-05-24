"""Optional FastAPI wrapper for minimal recommender.

Implement:
- expose /recommend endpoint
- accept user_id and top_k parameters
- return JSON response with recommended item IDs
"""

from fastapi import FastAPI, Query

from .service import MinimalRecommender

app = FastAPI()
recommender = MinimalRecommender()


@app.get("/recommend")
def recommend(user_id: str = Query(...), top_k: int = Query(10, gt=0, le=50)):
    items = recommender.recommend(user_id, top_k=top_k)
    return {"user_id": user_id, "recommendations": items}
