# Online Recommendation Latency Optimization Design

## Background

The current recommendation pipeline is:

```text
two-tower recall -> three-tower rough rank -> MMoE fine rank -> rerank
```

The main optimization goal is to reduce the response time of online recommendation calls such as:

```python
pipeline.recommend(user_id=15)
```

The first optimization phase must measure where time is spent before changing model or feature logic. The second phase will use those measurements to decide which precomputation or lookup-table optimizations should be implemented first.

## Goals

1. Add lightweight timing visibility for one online recommendation request.
2. Keep the existing `recommend()` return format unchanged.
3. Report time spent in recall, rough ranking, fine ranking, reranking, and the full request.
4. Report item counts after each stage.
5. Use the measurements to guide the next phase: precomputation and lookup-table optimization.

## Non-Goals

1. Do not change ranking logic in the timing phase.
2. Do not change model weights or training code in the timing phase.
3. Do not add MySQL, Redis, or API service work in the timing phase.
4. Do not optimize startup time in this phase.

## Proposed Approaches

### Recommended: Add a Separate `recommend_with_timing()`

Add a new method on `RecommenderPipeline`:

```python
recommendations, timing = pipeline.recommend_with_timing(
    user_id=15,
    top_k=10,
    recall_size=300,
    rough_rank_size=100,
    fine_rank_size=50,
)
```

This keeps `recommend()` stable for existing callers while making timing explicit for debugging and benchmarking.

Trade-offs:

1. Slight duplication with `recommend()` unless shared helper logic is extracted.
2. Clearer and safer than changing the return value of `recommend()`.
3. Easy to remove or keep as a diagnostic method.

### Alternative: Add `debug_timing=True` to `recommend()`

This would allow:

```python
pipeline.recommend(user_id=15, debug_timing=True)
```

Trade-offs:

1. Convenient for quick use.
2. Risks making `recommend()` return different shapes depending on a flag.
3. Less clean for future API use.

### Alternative: Use External Profiling Only

Run profiling tools without changing source code.

Trade-offs:

1. No code changes.
2. Harder to compare stage-level latency repeatedly.
3. Less readable for this project because the pipeline already has clear stage boundaries.

## Design

### Architecture

The first phase adds a small timing layer inside `RecommenderPipeline`. The existing pipeline stages remain unchanged:

```text
recommend_with_timing()
  -> recall()
  -> rough_rank()
  -> fine_rank()
  -> rerank()
```

Each stage is timed with a monotonic clock. The method returns both the final recommendations and a structured timing dictionary.

### Data Flow

The timed flow is:

1. Start total timer.
2. Run recall and record elapsed time plus output count.
3. Run rough rank and record elapsed time plus output count.
4. Run fine rank and record elapsed time plus output count.
5. Run rerank and record elapsed time plus output count.
6. Stop total timer.
7. Return final recommendations and timing metadata.

Example timing output:

```python
{
    "total_ms": 142.31,
    "stages": {
        "recall": {"elapsed_ms": 60.12, "item_count": 300},
        "rough_rank": {"elapsed_ms": 35.44, "item_count": 100},
        "fine_rank": {"elapsed_ms": 42.18, "item_count": 50},
        "rerank": {"elapsed_ms": 2.57, "item_count": 10},
    },
}
```

### Error Handling

Timing must not hide existing errors. If recall, ranking, or rerank raises an exception, the exception should still propagate normally. The timing method is diagnostic, not a fallback mechanism.

If a stage returns an empty list, the timing output should still record that stage's elapsed time and `item_count = 0`. Later stages should follow the existing behavior.

### Testing

The timing phase should verify:

1. `recommend()` still returns the same shape as before.
2. `recommend_with_timing()` returns a tuple of recommendations and timing metadata.
3. Timing metadata includes `total_ms`, all four stage names, `elapsed_ms`, and `item_count`.
4. Item counts match the actual outputs of each stage.
5. A local demo run prints timing information clearly enough to guide the next optimization phase.

## Phase 2 Direction: Precompute and Lookup Tables

After timing identifies the slowest stages, optimize in measured order. The likely candidates are:

1. Recall: prebuild movie tensors and genre tensors.
2. Recall: precompute movie-side embeddings so online recall only computes the user vector and similarity scores.
3. Rough rank: replace repeated dense feature construction with user/movie statistic lookups.
4. Fine rank: prebuild candidate-independent movie index and genre tensors.
5. Rerank: keep seen movies and genres as in-memory lookup tables first; database migration can come later.

Phase 2 should preserve recommendation behavior as much as possible, then compare latency before and after each optimization.

## Phase 3 Direction: Explicit Cross Features

After latency measurement and lookup-table optimization are stable, add explicit cross-feature work as a separate feature-enhancement phase. This is not part of the current implementation phase.

Candidate cross features:

1. User historical genre preference x candidate movie genre match.
2. User profile fields such as gender, age, and occupation x candidate movie genres.
3. User average rating or activity level x movie average rating or popularity.
4. Upstream model score interaction such as `recall_score x coarse_score`.

The goal of this phase is to make ranking features more interpretable and give the rough-rank or fine-rank model stronger user-item interaction signals beyond the current implicit interactions learned by MLP/MMoE layers.

## Acceptance Criteria

1. Existing `pipeline.recommend()` callers continue to work without return-format changes.
2. A new timing entry point can measure total and per-stage latency.
3. The timing output includes per-stage item counts.
4. Running the pipeline for a sample user produces timing data that clearly shows the largest bottleneck.
5. No ranking, model, or database behavior changes are introduced in phase 1.
