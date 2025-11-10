# Test Matrix (Accuracy & Performance)

This document standardizes how we validate every native rewrite. Each module must
ship with:

1. **Accuracy tests** (`tests/` via `pytest`): deterministic parity checks between
   legacy, tensor‑python, and tensor‑fast implementations.
2. **Performance benchmarks** (`tools/` scripts): throughput comparisons across
   legacy / tensor‑python / tensor‑fast.

All new tests should be added to this matrix and follow the conventions below.

---

## Global Conventions

| Item                     | Requirement                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------------- |
| Device                   | Default to `cpu`. If a script supports CUDA, call it out explicitly at the top of the file.        |
| Random seeds (accuracy)  | `rng = random.Random(0xF00DCAFE)` (plus matching `torch.manual_seed` if tensors are used).         |
| State sampling (accuracy)| `num_states = 10_000`, `max_random_moves = 80`.                                                     |
| Benchmark CLI            | Use the shared parser signature (see **Benchmark CLI Template**).                                   |
| Output artifacts         | Accuracy logs → `tests/result/<name>.txt`; Benchmark logs → `tools/result/<name>.txt`.             |
| Usage strings            | Every test/benchmark file must include a short “Usage” section or module docstring showing the CLI. |

### Benchmark CLI Template

```python
parser = argparse.ArgumentParser(
    description="Benchmark <module> throughput across implementations."
)
parser.add_argument("--states", type=int, default=1000, help="Number of random states to sample.")
parser.add_argument("--batch-size", type=int, default=64, help="Chunk size for tensor fast path.")
parser.add_argument("--runs", type=int, default=5, help="Benchmark repetitions.")
parser.add_argument("--device", type=str, default="cpu", help="Device (cpu by default; mention if cuda is supported).")
parser.add_argument("--max-random-moves", type=int, default=80, help="Random rollout depth for sampled states.")
parser.add_argument("--seed", type=int, default=0, help="Seed for state sampling.")
```

Scripts may clamp `--device` back to CPU if the underlying kernel lacks CUDA support, but the flag must still exist for parity with other benchmarks.

---

## Accuracy Tests (`pytest` under `tests/`)

| File                                        | Purpose / Notes                                                                                     | Usage Example                                                                 | Result File                      |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | -------------------------------- |
| `tests/v1/test_policy_projection_fast_accuracy.py` | Ensures tensor-fast policy projection matches legacy distributions (uses shared seed & state budget). | `pytest tests/v1/test_policy_projection_fast_accuracy.py -q > tests/result/policy_proj_accuracy.txt` | `tests/result/policy_proj_accuracy.txt` |
| `tests/v1/test_fast_apply_moves.py`         | Validates C++ `batch_apply_moves` against Python `apply_move` on 10k sampled states.                | `pytest tests/v1/test_fast_apply_moves.py -q > tests/result/apply_moves_accuracy.txt`               | `tests/result/apply_moves_accuracy.txt` |

**Guidelines**
- Always seed both Python’s `random` and `torch`.
- Mention the seed and sampling parameters in the docstring or test header.
- If a native extension is unavailable, the test should `pytest.skip`, but still emit a log to the result file explaining why.

---

## Performance Benchmarks (`tools/` or `tests/…_performance.py`)

| File / Module                                           | Notes                                                                                          | Usage Example                                                                                             | Result File                           |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `tests/v1/test_policy_projection_fast_performance.py`   | Pytest-marked “slow” micro-benchmark for policy projection (legacy vs tensor-python vs C++ fast). | `pytest tests/v1/test_policy_projection_fast_performance.py -s > tools/result/policy_proj_perf.txt`       | `tools/result/policy_proj_perf.txt`   |
| `tools/benchmark_policy_projection.py`                  | CLI benchmark with batching, summarizing mean ± stdev across runs.                              | `python -m tools.benchmark_policy_projection --states 1000 --batch-size 128 --runs 5 > tools/result/policy_proj_bench.txt` | `tools/result/policy_proj_bench.txt` |
| `tests/v1/test_fast_apply_moves_performance.py`         | Pytest “slow” benchmark for apply pipeline (uses metadata + fast path).                         | `pytest tests/v1/test_fast_apply_moves_performance.py -s > tools/result/apply_moves_perf.txt`             | `tools/result/apply_moves_perf.txt`   |
| `tools/benchmark_apply_moves.py`                        | Standalone CLI benchmark for apply_move throughput (CPU only today).                            | `python -m tools.benchmark_apply_moves --states 1000 --runs 5 > tools/result/apply_moves_bench.txt`       | `tools/result/apply_moves_bench.txt`  |

**Guidelines**
- Benchmarks should reuse the accuracy seed/state sampler when generating scenarios for timing.
- Always print per-run timings and a final summary in the format `legacy=X ms, tensor-python=Y ms, tensor-fast=Z ms`.
- When the fast path is unavailable, print “N/A” but keep legacy/tensor-python numbers for reference.
- Store raw stdout in the corresponding `tools/result/*.txt` file so regressions can be tracked.

---

## Adding a New Native Implementation

1. **Accuracy**: create `tests/v1/test_<module>_accuracy.py` following the seed/state conventions. Update the table above with Usage + result path.
2. **Performance**: add both a pytest-style micro benchmark and/or a CLI under `tools/`. Ensure the CLI matches the shared parser and logs to `tools/result`.
3. **Documentation**: append the new entries to this README as soon as the tests land, so future contributors can discover them quickly.
4. **CI / Local workflow**: before merging a native rewrite, run both accuracy and performance suites, capture the outputs into the result folders, and mention them in the PR summary if relevant.

This single document should stay in sync with the actual test files—treat it as the source of truth for how we validate native kernels.

