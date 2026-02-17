# v1 Full-GPU Self-Play Design
Date: 2026-02-16  
Scope: Design for migrating v0 CPU-bottleneck self-play to a GPU-first v1 pipeline.

## User Requirement (Original Message, English Copy)
The user requested full delegated execution on a Windows machine, including:
- Run any smoke tests and add scripts in-repo for efficiency/functional checks.
- Prove whether current compute is mainly CPU-bottlenecked.
- Design whether full self-play can be moved to GPU (not only storage + forward pass), with torch-format output directly feeding training.
- Produce a detailed v1 pipeline design based on v0.
- Completion criteria:
  - Training efficiency should improve significantly in GPU version.
  - GPU power behavior should improve (not just high util with only small watt increase over idle).
  - Increasing CPU workers under GPU version should no longer provide large throughput gains.
- Persist this requirement into `v1/Design.md` for continuity.

## Current v0 Diagnosis
### Code-path evidence (why CPU is still central)
- `v0/src/mcts/mcts_core.cpp` still converts leaf states to CPU tensors (`FromGameStates(..., CPU)`), then frequently copies tensors back to CPU (`probs/values/legal_mask/metadata`) for action decoding, sorting, and node expansion.
- Child-state generation in `ExpandBatch` materializes CPU-side vectors (`std::vector<GameState>`) and uses `ToGameStates(...)` (GPU -> CPU conversion) before inserting nodes.
- Tree storage is CPU object-oriented (`std::vector<Node>`, each `Node` contains `GameState`), so selection/backprop are CPU loops.
- Python wrapper `v0/python/mcts.py` decodes actions through CPU-side `GameState` replication for move dicts.
- Output pipeline stores per-position Python `GameState` + Python legal move lists, then JSONL serialization (`v0/python/state_io.py`), which is CPU-heavy.

### Smoke measurements on this Windows machine (RTX 3060 Laptop, 90W cap)
Environment: `conda run -n torchenv`, v0 self-play smoke with `inference_backend=py`, `mcts_simulations=12`, `total_games=8`.

| workers | time(s) | games/s | pos/s | GPU util avg | GPU power avg |
|---|---:|---:|---:|---:|---:|
| 1 | 29.81 | 0.268 | 33.7 | 23.0% | 22.4W |
| 2 | 24.81 | 0.322 | 43.1 | 33.6% | 25.0W |
| 4 | 24.89 | 0.321 | 41.1 | 31.9% | 25.4W |

Pure inference baseline (same model, large batch loop):
- GPU util avg: ~100%
- GPU power avg: ~44.9W

Interpretation:
- Self-play remains low-power and not compute-dense on GPU.
- Throughput changes materially with CPU worker count (1 -> 2 improved), indicating non-trivial CPU-side bottlenecks.
- GPU has headroom (pure inference can draw much higher power/utilization).

## v1 Objective
Build an end-to-end GPU self-play path where:
1. MCTS selection/expansion/backprop state math runs on GPU tensors.
2. Legal-mask, policy projection, action sampling, and state transition stay on GPU.
3. Self-play output is emitted directly as torch tensors (no mandatory Python `GameState` reconstruction).
4. CPU workers are orchestration-only; scaling by CPU workers should no longer be a major speed lever.

## Flow-First Direction (2026-02-17)
Primary engineering rule for v1:
- First optimize data/control flow (remove host sync and Python control hot paths).
- Then apply micro-optimizations (kernel tricks, backend toggles, launch tuning).

What this means concretely:
1. Keep tensor->tensor on device across one self-play step.
2. Avoid tensor->Python scalar in hot loops (`.item()`, `.tolist()`).
3. Move per-game branching (`terminal/winner/draw`) to batched tensor logic first.
4. Only after (1)-(3), do fused kernels and backend-specific tuning.

### Secondary optimizations (de-prioritized for now)
- `v1` graph backend toggle: useful as diagnostic, not a guaranteed throughput win on RTX 3060.
- `child value-only` branch: kept because semantically verified and positive, but treated as incremental gain (not the core bottleneck fix).
- Additional runtime knobs: retained for validation, but not considered the main delivery axis.

### Current status against flow-first plan
Implemented:
- Root-PUCT visit allocation in C++ op (`v0_core.root_puct_allocate_visits`).
- Child `value_only` evaluation path with strict A/B semantic gate.
- Root action/policy writeback in `mcts_gpu.py` converted to gather/scatter tensor path (removed per-root Python scalar extraction).
- `self_play_gpu_runner.py` main loop changed to batch tensor terminal/winner/draw detection and reduced per-step scalar sync points.
- Trajectory path upgraded to preallocated tensor arena with batched append/finalize APIs (`append_steps`, `finalize_games`).
- Self-play wave loop now uses pure tensor index flow for step tracking/finalization (removed `cpu().tolist()` from hot path).
- Added C++ `v0_core.self_play_step_inplace` op to execute state apply, terminal/winner/draw checks, and in-place state/plies/done updates in one call.
- Dual-track validation (`P0` main gate + fixed-clock diagnostics).

Not implemented yet:
- Self-play still has small host sync points for aggregate counters (`sum().item()`), but no longer does per-element host extraction in the hot path.
- Full GPU tree arena for multi-depth selection/expansion/backprop.
- Fused CUDA kernel for root sparse pack/projection/writeback chain (visit allocation itself is now fused in R1).

## Detailed Technical Route (Post-stepop, 2026-02-17)
This route is mandatory for chasing the `fixed-worker >=10x` goal under semantic parity constraints.
The current implementation already removed major Python orchestration hotspots, so next gains must come from kernel-level consolidation, not from adding runtime knobs.

### Baseline facts (why route changes now)
- After `self_play_step_inplace`, v1 throughput is stable around `~0.34-0.36 games/s` on this RTX 3060 setup.
- `concurrent_games` sweep (`8/16/32/64`) shows only marginal gain after `16`, indicating launch/orchestration overhead dominates before raw GPU compute saturates.
- `root_puct_allocate_visits` and action pack/writeback are still composition-heavy tensor workflows, not fused kernels.

### Stage R1: Fuse Root-PUCT Visit Allocation (highest priority)
Goal:
- Replace ATen composition loop in `root_puct_allocate_visits` with a dedicated CUDA kernel while keeping exact root-PUCT semantics.

Scope:
- `v0/src/bindings/module.cpp` (`RootPuctAllocateVisits` entry remains API-compatible).
- New CUDA/C++ implementation file for fused visit allocation.

Execution-level implementation plan:
- Add fused implementation pair:
  - `v0/include/v0/root_puct_fused.hpp`
  - `v0/src/mcts/root_puct_fused.cu`
- Keep Python API unchanged:
  - `v0_core.root_puct_allocate_visits(priors, leaf_values, valid_mask, num_simulations, exploration_weight)`
- Kernel contract:
  - Input tensors are contiguous on one CUDA device.
  - Output tensors keep existing dtype/shape contract (`float32`, `[R,A]` and `[R]`).
  - Invalid actions are masked exactly as today; no search-budget or math shortcut.
- Tie behavior:
  - Preserve deterministic first-index-max selection semantics to avoid A/B drift.
- Instrumentation:
  - Add per-call wall-time metric (`root_puct_ms`) in validator/profiler path.
  - Emit before/after median and p95 in the stage report.

Invariants:
- Same inputs/outputs (`visits`, `value_sum`, `root_values`).
- Same selection rule (`q + u`, masked invalid actions, identical tie behavior as current implementation).

Risks:
- Numeric drift from reduction/tie behavior.
- Warp divergence when legal action counts vary.

Validation gate:
- A/B semantic parity against current implementation:
  - action match ratio `=1.0`,
  - root value mean/max abs diff `=0`,
  - W/L/D match `True`.
- Local perf gate:
  - Root-PUCT segment wall-time reduced by >=40% at fixed config (`threads=1`, `cg=16`, `sims=128/256`).

Exit criteria:
- Pass semantic gate and local perf gate in one report.
- If semantic gate fails, keep fused path disabled by default and block R2 entry.

### Stage R2: Move Sparse Action Pack/Writeback Chain to C++/CUDA
Goal:
- Remove Python-side dense/sparse reshaping and scatter/gather orchestration in root search.

Scope:
- New op that ingests per-root legal mask + projected probs + metadata and emits:
  - compact action buffer for child expansion,
  - legal index map for policy writeback,
  - chosen action indices/codes.
- Replace the Python tensor choreography in `v1/python/mcts_gpu.py` for this path.

Execution-level implementation plan:
- Introduce one fused sparse pipeline op (single call from Python):
  - input: `legal_mask_bool`, `probs`, `metadata`, `valid_root_indices`, `temperature`, `sample_moves`, optional noise config.
  - output:
    - `row_ptr` (CSR row split),
    - `action_codes_flat`,
    - `parent_local_flat`,
    - `policy_dense_valid` (or writeback index/value pair),
    - `chosen_action_indices`,
    - `chosen_action_codes`.
- Keep `TOTAL_ACTION_DIM` mapping and `metadata` encoding unchanged.
- Remove Python-side ragged assembly at:
  - packing path (`mcts_gpu.py` root legal-action compaction),
  - writeback path (`policy_dense` scatter/index_copy sequence).
- Instrumentation:
  - Add `root_pack_writeback_ms` timer and include in stage regression report.

Invariants:
- Action encoding unchanged (same `TOTAL_ACTION_DIM` index semantics).
- Root policy tensor identical to current path under fixed seed.

Risks:
- Incorrect index mapping on ragged legal-action rows.
- Hidden host sync if fallback path is accidentally triggered.

Validation gate:
- Fixed-seed parity (`top-k`, chosen action, root value).
- `validate_v1_claims.py` regression with same parameters as previous baseline.

Exit criteria:
- >=25% throughput gain at `threads=1` versus post-R1 baseline.
- If <10% gain, require hotspot breakdown before allowing R3 implementation.

### Stage R3: Convert step-op to true slot-inplace CUDA path
Goal:
- Remove internal gather/apply/scatter trip in `self_play_step_inplace` and execute slot updates directly on global state tensors.

Scope:
- Rework `self_play_step_inplace` internals to avoid repeated `index_select/index_copy` state shuttling.
- Keep current Python signature unchanged for compatibility.

Execution-level implementation plan:
- Introduce slot-direct kernel path:
  - iterate active slots directly,
  - decode/apply chosen action in-place to global tensors,
  - compute terminal/draw/winner and `done` mark in the same kernel chain.
- Preserve current return payload:
  - `finalize_slots`, `result_from_black`, `soft_value_from_black`.
- Do not alter external orchestration API in `v1/python/self_play_gpu_runner.py`.
- Instrumentation:
  - add `self_play_step_ms` and split into:
    - `apply_ms`,
    - `terminal_eval_ms`,
    - `finalize_emit_ms`.

Invariants:
- Rule transitions, draw/winner logic, and plies accounting identical.
- Finalization outputs (`finalize_slots`, `result`, `soft_value`) identical.

Risks:
- In-place write hazards across slots.
- More difficult debugging for mixed terminal/non-terminal slots.

Validation gate:
- Fixed-seed end-to-end self-play parity (`action/value/WLD`, tensor targets).
- No regression in long-stability (`rounds>=3`, mean/std/worst tracked).

Exit criteria:
- >=25% self-play step segment speedup versus post-R2 baseline.
- If race/in-place hazard appears, rollback to previous stable step path and keep R3 behind explicit opt-in.

### Stage R4: Finalize Buffer/Target Fill Downshift (last-mile)
Goal:
- Reduce remaining Python indexing orchestration in target finalization.

Scope:
- Move `finalize_games` flat index expansion/value fill path to C++/CUDA.

Execution-level implementation plan:
- Add target-fill op:
  - input: `step_index_matrix`, `step_counts`, `slots`, `result_from_black`, `soft_value_from_black`, `player_signs`.
  - output/inplace: `value_targets`, `soft_value_targets`.
- Keep trajectory schema unchanged for train bridge compatibility.
- Instrumentation:
  - add `target_finalize_ms` and report jitter (std/p95) reduction.

Invariants:
- Value sign convention unchanged.
- Soft-value target exactly matched for fixed seed.

Risks:
- Index broadcast mismatch when game lengths differ.

Validation gate:
- Tensor parity (`value_targets`, `soft_value_targets`) exact match.

Exit criteria:
- Additional 5-10% throughput gain or measurable host-overhead reduction.

### Route-level stop/go criteria
1. Proceed to next stage only if current stage passes semantic gate fully.
2. Keep same benchmark harness and same seed policy across all stage reports.
3. If after R3 `speedup_fixed_worker_min < 3.0` on the same RTX 3060 setup, trigger architecture review:
   - move from mixed `v0_core + Python orchestration` to dedicated `v1` C++ module boundary for root-search + step pipeline.

### R1 status (implemented, 2026-02-17)
Delivered:
- Added fused CUDA implementation for root visit allocation:
  - `v0/src/mcts/root_puct_fused.cu`
  - `v0/include/v0/root_puct_fused.hpp`
- Kept Python API unchanged (`v0_core.root_puct_allocate_visits`), with CUDA/CPU dispatch in:
  - `v0/src/bindings/module.cpp`

R1 microbenchmark (fixed `roots=16`, `actions=220`):
- `results/root_puct_benchmark_after_r1.json`
- `sims=128`: fused median `0.624ms` vs reference `137.319ms` (drop `99.5%`, `220.17x`)
- `sims=256`: fused median `3.726ms` vs reference `195.683ms` (drop `98.1%`, `52.52x`)
- Numeric diff in direct microbenchmark is near machine precision (`root_value_max_abs_diff=1.19e-07`).

End-to-end semantic gate after R1:
- `results/v1_child_value_ab_after_r1_256.json`
- `results/v1_child_value_ab_after_r1_512.json`
- Result: action/value/WLD gates all PASS in fixed-seed A/B.

End-to-end throughput snapshot after R1:
- Fixed-worker (`results/v1_validation_workers_py_128_after_r1.json`):
  - `speedup_best_v1_vs_v0_worker1=11.78`
  - `speedup_fixed_worker_min=5.93`
- Concurrency sweep (`results/v1_gpu_matrix_8_16_32_64_py_graph_after_r1.json`):
  - `v1(py)` reaches `~1.11 games/s` at `cg=64`
  - `v1(graph)` reaches `~1.10 games/s` with higher power (`~39W`)
- Long-stability (`rounds=3`):
  - `results/v1_validation_workers_py_128_rounds3_after_r1.json`
  - `results/v1_validation_workers_graph_128_rounds3_after_r1.json`

### Delivery discipline (applies to all stages)
1. Do not change search budget semantics.
2. Do not add pruning/surrogate shortcuts for benchmark-only gains.
3. Keep one baseline validator (`tools/validate_v1_claims.py`) for longitudinal comparability.
4. Every stage must ship with:
   - semantic A/B report,
   - fixed-worker report (`1/2/4`),
   - concurrency sweep report (`8/16/32/64`),
   - short long-run report (`rounds>=3`).

## Non-goals (v1)
- No rule change.
- No action encoding spec change.
- No mandatory rewrite of legacy (`src/`) training path.

## v1 Architecture
### 1) GPU State Arena (SoA)
Replace per-node CPU `GameState` storage with GPU tensor arenas:
- `board[N,6,6] int8`
- `marks_black/marks_white[N,6,6] bool`
- `phase/current_player/...` scalar tensors
- `parent`, `action_index`, `visit_count`, `value_sum`, `prior`, `first_child`, `child_count`

Key property: node metadata and game-state payload remain on device.

### 2) GPU Tree Ops
Implement CUDA/Torch kernels for:
- `select_paths`: batched PUCT traversal using tensorized node stats.
- `expand_nodes`: legal mask + policy projection + top-k/compaction on device.
- `backprop_paths`: batched sign-flip accumulation to ancestors.

No GPU->CPU round-trip in search loop.

### 3) GPU Action Decode/Apply
Keep encoded action IDs as primary format in search and trajectory.
- Only decode to move dict when explicitly needed for debug/UI.
- Extend `batch_apply_moves_cuda` pipeline to consume compact action lists directly and output child states in device tensors.

### 4) GPU Trajectory Buffer -> Training
Self-play emits tensors:
- `state_tensor[B,C,H,W]`
- `policy_dense[B,A]` (or sparse index/value pair)
- `value[B]`, `soft_value[B]`
- Optional `legal_mask[B,A]`

Expose via torch-native dataset/iterator:
- zero-copy if training on same device;
- pinned-memory staged copy only when needed.

### 5) Runtime Topology
- One GPU process per device owns full self-play compute for that device.
- CPU workers become optional feeders/control threads, not throughput-critical compute workers.

## Proposed File/Module Changes
### New (v1)
- `v1/python/self_play_gpu_runner.py`
- `v1/python/trajectory_buffer.py`
- `v1/python/train_bridge.py`
- `v1/src/mcts_gpu/` (GPU-first tree ops)
- `v1/src/game/gpu_state_arena.*`
- `v1/src/bindings/v1_module.cpp`

### Touch points in v0/v1 shared area
- Reuse/extend CUDA kernels in `v0/src/game/fast_legal_mask_cuda.cu` and `v0/src/game/fast_apply_moves_cuda.cu`.
- Add v1-only pybind APIs for tensor-native rollout output.

## Migration Plan
### Phase P0: Instrumentation first
- Add consistent timing and GPU power/util logging script (`tools/smoke_v0_bottleneck.py`).
- Freeze baseline numbers for same seeds/configs.

### Phase P1: Remove hot-path GPU<->CPU bounces
- In `ExpandBatch`, keep policy/metadata/action assembly entirely on GPU.
- Eliminate `ToGameStates` in expansion path.

### Phase P2: GPU node arena
- Introduce GPU node tensor storage.
- Port selection/backprop to batched tensor kernels.

### Phase P3: Tensor-native rollout export
- Self-play returns torch tensors directly.
- Training loop consumes tensors without JSONL/Python object detour.

### Phase P4: Throughput tuning
- Kernel fusion, pre-allocation, stream overlap, graph capture for full search step.
- Validate CPU-worker sensitivity collapse.

## Invariants (must hold)
- Rule semantics identical to `README.md` and `rule_description.md`.
- Action index mapping identical to v0 spec.
- Terminal/winner logic unchanged.
- Value target sign convention unchanged.
- No benchmark-distorting pruning/shortcut behavior is allowed when comparing v1 vs v0 (same semantics, same search budget scale).

## Risks
- GPU tree memory fragmentation/overflow.
- Atomic update contention in backprop.
- Numerical drift from parallel reduction order.
- Debug complexity increase (GPU-only state).

## Validation Plan (acceptance aligned with user criteria)
### Test Standards (User-defined, 2026-02-16)
Primary standards:
1. GPU power state should stay in `P0` during v1 self-play runs as a stable baseline.
2. With fixed worker/thread count, v1 throughput should be at least `10x` of v0 (same scale, e.g., `1 vs 1`, `2 vs 2`, `4 vs 4`).
3. CPU parallel scaling should become secondary in v1 (adding CPU threads no longer gives large gains).

Dual-track policy (to avoid benchmark drift):
- Keep the above primary standards as milestone acceptance gates.
- Add diagnostic gates for fixed-clock runs:
  - graphics/sm clock range stability (`min/max` bands),
  - power range stability (`min/max` bands).
- Diagnostic gates do not replace primary gates; they are used to interpret P-state behavior on mobile GPUs.
- Comparison discipline:
  - Keep v0/v1 search budget and rollout semantics aligned.
  - Do not introduce pruning or heuristic shortcuts that change the information flow for benchmark-only gains.

Mandatory semantic gate before enabling `child value-only` in production:
- Fixed-seed root-action consistency (`full` vs `value_only`).
- Fixed-seed root-value deviation (`mean/max abs diff`).
- Fixed-seed self-play W/L/D consistency and tensor target parity.

Operational thresholds used in validator:
- `v1_p0_ratio >= 0.90` (minimum across tested v1 thread settings).
- `v1_speedup_fixed_worker_min >= 10.0` (minimum across same-scale pairs).
- `v1_thread_gain <= 0.15` (gain from best thread setting vs thread=1).
- Optional diagnostics:
  - `diag_v1_graphics_clock_[min,max]_mhz`
  - `diag_v1_sm_clock_[min,max]_mhz`
  - `diag_v1_power_[min,max]_w`

### Functional correctness
- Cross-check v1 vs v0 on fixed seeds:
  - root policy top-k overlap
  - winner/result parity
  - action legality parity
- Reuse and extend `tests/v0/cuda`-style parity tests for v1 kernels.

### Performance and power
Use same hardware and config family as baseline:
1. Self-play throughput (`games/s`, `positions/s`) must significantly exceed v0.
2. GPU average power in self-play should move closer to inference baseline (no longer near idle+few watts).
3. Worker sensitivity test (`workers=1,2,4`):
   - v1 target: adding workers gives marginal gain (for example <10-15% from 1->4) because compute is GPU-dominant.

### Training handoff
- Verify tensor-native export plugs into training without JSONL intermediate.
- Measure end-to-end iteration time (`self_play + train`) vs v0.

## Vibe Coding Plan Checklist
### 1. Goal and Scope
- Build v1 full-GPU self-play and tensor-native training handoff.
- Out of scope: rule changes and legacy refactor.

### 2. Impact Surface
- MCTS core, state representation, action pipeline, self-play output format, train bridge.

### 3. Invariants
- Rule flow, action encoding, value/policy semantics unchanged.

### 4. Risk Points
- GPU memory, kernel race/contention, numeric drift, debugging difficulty.

### 5. Validation
- Parity tests + throughput/power benchmarks + worker sensitivity benchmark.

### 6. Deliverables
- v1 GPU core modules, tensor output bridge, smoke/benchmark scripts, updated docs.

## Implementation Progress (2026-02-16)
### Added prototype path (v1, GPU-first)
- Implemented a new v1 self-play/MCTS path that keeps legal-mask, policy projection, action apply, and trajectory construction on GPU tensors.
- Added tensor-native train bridge so self-play output can feed training directly without JSONL.
- Kept `v0` unchanged.

### Current scope
- Implemented root-PUCT search on GPU (`v1/python/mcts_gpu.py`) as a migration baseline.
- Implemented batched root-search + batched multi-game self-play (`concurrent_games`) to reduce Python per-game overhead and increase GPU-side throughput.
- For benchmark comparability, v1 validation path now disables action pruning/heuristic surrogate; root visit allocation uses equivalent PUCT semantics.
- Not yet implemented: full multi-depth GPU node arena kernels (`v1/src/mcts_gpu/`, `v1/src/game/gpu_state_arena.*`, v1 pybind C++ module).

### New files
- `v1/__init__.py`
- `v1/python/__init__.py`
- `v1/python/mcts_gpu.py`
- `v1/python/self_play_gpu_runner.py`
- `v1/python/trajectory_buffer.py`
- `v1/python/train_bridge.py`
- `tools/smoke_v1_gpu_pipeline.py`

### Build and smoke status (Windows)
- Rebuilt `v0_core` successfully (`build/v0/src/v0_core.cp313-win_amd64.pyd`).
- Smoke pipeline passed:
  - `v1 self-play -> tensor batch -> tensor-native train` end-to-end runnable.
  - Example command:
    - `conda run -n torchenv cmd /c "set PYTHONPATH=d:\CODES\liuzhou\build\v0\src;d:\CODES\liuzhou&& python tools/smoke_v1_gpu_pipeline.py --device cuda:0 --num_games 1 --mcts_simulations 8 --epochs 1 --batch_size 64"`

### Pending milestones
- Replace root-only search with full GPU tree selection/expansion/backprop arena.
- Add v1 parity tests versus v0 on fixed seeds.
- Add v1 worker-sensitivity and power benchmark matrix (`1/2/4` workers) for acceptance criteria.

## Validation Scripts (2026-02-16)
### Unified acceptance validator
- Script: `tools/validate_v1_claims.py`
- Purpose:
  - Run v0 worker scaling (`--v0-workers`, default `1,2,4`)
  - Run v1 CPU-thread sensitivity (`--v1-threads`, default `1,2,4`)
  - Control v1 game-level batching (`--v1-concurrent-games`, default Windows=`8`)
  - Sample GPU util/power during each run
  - Compute acceptance indicators:
    - v1 speedup vs v0(worker=1)
    - v1 fixed-worker speedup (same scale pairs)
    - v1 power uplift vs v0(worker=1)
    - v1 P0 ratio stability
    - v1 thread-sensitivity gain ratio
  - Compute diagnostic indicators (fixed-clock rail):
    - v1 graphics/sm clock min-max ranges
    - v1 power min-max range
  - Emit JSON report and PASS/FAIL criteria summary.
- Recommended command (Windows):
  - `conda run -n torchenv cmd /c "set PYTHONPATH=d:\CODES\liuzhou\build\v0\src;d:\CODES\liuzhou&& python tools/validate_v1_claims.py --device cuda:0 --seed 12345 --rounds 1 --v0-workers 1,2,4 --v1-threads 1,2,4 --v1-concurrent-games 8 --v1-child-eval-mode value_only --total-games 8 --v0-mcts-simulations 512 --v1-mcts-simulations 512 --v0-batch-leaves 512 --v0-inference-backend graph --v0-inference-batch-size 512 --v0-inference-warmup-iters 5 --v0-opening-random-moves 2 --v0-resign-threshold -0.8 --v0-resign-min-moves 36 --v0-resign-consecutive 3 --min-v1-speedup-fixed-worker 10.0 --min-v1-p0-ratio 0.90 --max-v1-thread-gain 0.15 --diag-v1-graphics-clock-min-mhz 0 --diag-v1-sm-clock-min-mhz 0 --diag-v1-power-min-w 0 --with-inference-baseline --output-json results/v1_validation_latest_after_vec.json"`

### Child value-only semantic A/B gate
- Script: `tools/ab_v1_child_value_only.py`
- Purpose:
  - Compare `child_eval_mode=full` vs `child_eval_mode=value_only` with fixed seed.
  - Report:
    - root action match ratio,
    - root value mean/max abs diff,
    - self-play W/L/D match,
    - value/policy target tensor diff.
- Recommended command (Windows):
  - `conda run -n torchenv cmd /c "set PYTHONPATH=d:\CODES\liuzhou\build\v0\src;d:\CODES\liuzhou&& python tools/ab_v1_child_value_only.py --device cuda:0 --seed 12345 --num-states 32 --state-plies 8 --mcts-simulations 128 --self-play-games 8 --self-play-concurrent-games 8 --strict --output-json results/v1_child_value_ab_latest.json"`

### Concurrency/Backend matrix
- Script: `tools/sweep_v1_gpu_matrix.py`
- Purpose:
  - Sweep `concurrent_games` (e.g., `8/16/32`) and compare `v1(py)` vs `v1(graph)` under the same MCTS and thread settings.
  - Emit throughput (`games/s`, `positions/s`) + GPU util/power/memory/P0/clock telemetry in one JSON.
- Recommended command (Windows):
  - `conda run -n torchenv cmd /c "set PYTHONPATH=d:\CODES\liuzhou\build\v0\src;d:\CODES\liuzhou&& python tools/sweep_v1_gpu_matrix.py --device cuda:0 --seed 12345 --rounds 1 --threads 1 --concurrent-games 8,16,32 --backends py,graph --total-games 8 --mcts-simulations 128 --child-eval-mode value_only --output-json results/v1_gpu_matrix_8_16_32_py_graph.json"`

### Latest result snapshot (Windows, RTX 3060, 2026-02-16)
- Config: `sims=512`, `total_games=8`, `v1-concurrent-games=8`, `v1-inference-backend=py`
- `v0` (workers 1/2/4): `0.046 / 0.072 / 0.116 games/s`
- `v1` (threads 1/2/4): `0.150 / 0.151 / 0.154 games/s`
- Inference baseline: `~100% util`, `~44.9W`
- Criteria outcome:
  - `v1_speedup_ge_threshold`: PASS (`3.345` vs v0 worker=1)
  - `v1_thread_gain_le_threshold`: PASS (`0.024`)
  - `v0_worker_gain_ge_threshold`: PASS (`1.517`)
  - `v1_speedup_fixed_worker_ge_threshold`: FAIL (`min=1.329`, target `>=10.0`)
  - `v1_power_delta_ge_threshold`: FAIL (`+2.97W`, threshold `+5W`)
  - `v1_p0_ratio_ge_threshold`: FAIL (`0.0%`, target `>=90%`)

### Recent implementation note
- Added a new C++ binding op `v0_core.root_puct_allocate_visits` to move batched root-PUCT visit updates out of Python loops while preserving root search semantics.
- Added `v0_core.self_play_step_inplace` to move one full self-play step's state transition + terminal bookkeeping out of Python orchestration.

### Latest result snapshot (Windows, RTX 3060, after `self_play_step_inplace`, 2026-02-17)
- Fixed-worker regression (`results/v1_validation_workers_py_128_after_stepop.json`):
  - `v0` workers 1/2/4: `0.106 / 0.155 / 0.211 games/s`
  - `v1(py)` threads 1/2/4: `0.346 / 0.356 / 0.362 games/s`
  - `speedup_fixed_worker_min`: `1.716` (still below `10x` target)
  - `v1_thread_gain`: `0.044` (passes CPU-scaling gate)
- Fixed-worker regression (`results/v1_validation_workers_graph_128_after_stepop.json`):
  - `v1(graph)` threads 1/2/4: `0.328 / 0.343 / 0.338 games/s`
  - Higher power than `py`, but no throughput win.
- Concurrency sweep (`results/v1_gpu_matrix_8_16_32_64_py_graph_after_stepop.json`):
  - `v1(py)` games/s: `0.321 / 0.343 / 0.341 / 0.355` at `cg=8/16/32/64`
  - `v1(graph)` games/s: `0.336 / 0.333 / 0.333 / 0.335` at `cg=8/16/32/64`
  - Current saturation remains around `~0.33-0.36 games/s`.
- Long-stability (`rounds=3`, `results/v1_validation_workers_py_128_rounds3_after_stepop.json`):
  - threads=1: games/s mean/std/worst = `0.341 / 0.010 / 0.328`
  - threads=2: games/s mean/std/worst = `0.345 / 0.003 / 0.341`
  - threads=4: games/s mean/std/worst = `0.344 / 0.002 / 0.343`
- Long-stability (`rounds=3`, `results/v1_validation_workers_graph_128_rounds3_after_stepop.json`):
  - threads=1: games/s mean/std/worst = `0.348 / 0.003 / 0.346`
  - threads=2: games/s mean/std/worst = `0.352 / 0.002 / 0.350`
  - threads=4: games/s mean/std/worst = `0.353 / 0.003 / 0.349`

### One-click wrapper
- Script: `scripts/validate_v1_gpu.cmd`
- Purpose: launch the above validator with default matrix and output path.
