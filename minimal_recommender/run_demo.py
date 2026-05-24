"""Local demo entry point for minimal recommender.

Implement:
- load the minimal recommender service
- accept a sample user_id and top_k
- print recommended item IDs to console
"""

if __name__ == "__main__":
    from .service import MinimalRecommender

    recommender = MinimalRecommender()
    sample_user = "U00001"
    top_k = 10
    recommendations = recommender.recommend(sample_user, top_k=top_k)
    print(f"Recommendations for {sample_user}: {recommendations}")
