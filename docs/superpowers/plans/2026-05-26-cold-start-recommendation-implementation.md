# Cold Start Recommendation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a V1 new-user cold-start fallback based on `age + occupation` profile statistics.

**Architecture:** Add a focused `cold_start` package with `ColdStartRecommender`, then wire it into `RecommenderPipeline` only when the normal model chain cannot produce candidates. Existing old-user recommendations keep the same path and return shape.

**Tech Stack:** Python standard library, existing MovieLens `.dat` files, `unittest`.

---

### Task 1: ColdStartRecommender

**Files:**
- Create: `cold_start/__init__.py`
- Create: `cold_start/cold_start_recommender.py`
- Test: `tests/test_cold_start_recommender.py`

- [ ] Write failing tests for `age + occupation`, `occupation`, and global fallback paths.
- [ ] Run `python -m unittest tests.test_cold_start_recommender` and confirm import/behavior failures.
- [ ] Implement `ColdStartRecommender` with train-user, rating, and movie loaders.
- [ ] Score candidates with segment positive count, movie average rating, and popularity.
- [ ] Add simple primary-genre diversification.
- [ ] Run `python -m unittest tests.test_cold_start_recommender` and confirm pass.

### Task 2: Pipeline Fallback

**Files:**
- Modify: `recommender_pipeline.py`
- Modify: `tests/test_pipeline_timing.py`

- [ ] Write failing tests proving `recommend()` calls cold start when recall returns empty.
- [ ] Write failing tests proving `recommend_with_timing()` records a `cold_start` stage.
- [ ] Run `python -m unittest tests.test_pipeline_timing` and confirm failures.
- [ ] Add optional `age` and `occupation` parameters to `recommend()` and `recommend_with_timing()`.
- [ ] Build `ColdStartRecommender` during pipeline initialization.
- [ ] Route empty recall/rough/fine results to cold start fallback.
- [ ] Run `python -m unittest tests.test_pipeline_timing` and confirm pass.

### Task 3: Verification

**Files:**
- Modify only files touched above if verification exposes defects.

- [ ] Run `python -m unittest discover -s tests`.
- [ ] Run `D:\Anaconda\envs\recommend\python.exe recommender_pipeline.py` to ensure old-user demo still works.
- [ ] Run a small manual new-user check with `RecommenderPipeline().recommend(user_id=900001, age=25, occupation=4, top_k=10)`.
