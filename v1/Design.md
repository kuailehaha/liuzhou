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

### Latest result snapshot (Windows, RTX 3060, after R2 sparse pack/writeback downshift, 2026-02-17)
- Semantic A/B gate:
  - `results/v1_child_value_ab_after_r2_256.json`: PASS
  - `results/v1_child_value_ab_after_r2_512.json`: PASS
- Fixed-worker regression (`results/v1_validation_workers_py_128_after_r2.json`):
  - `v0` workers 1/2/4: `0.125 / 0.186 / 0.258 games/s`
  - `v1(py)` threads 1/2/4: `1.203 / 1.477 / 1.580 games/s`
  - `speedup_fixed_worker_min`: `6.128`
  - `speedup_best_v1_vs_v0_worker1`: `12.589`
  - `v1_thread_gain`: `0.313` (fails `<=0.15` gate)
  - `v1_p0_ratio_min`: `0.0` (fails `>=0.9` gate)
- Fixed-worker regression (`results/v1_validation_workers_graph_128_after_r2.json`):
  - `v1(graph)` threads 1/2/4: `0.385 / 0.398 / 0.397 games/s`
  - `speedup_fixed_worker_min`: `3.432`
  - `v1_thread_gain`: `0.035` (passes CPU-scaling gate)
  - `v1_p0_ratio_min`: `0.0` (fails `>=0.9` gate)
- Concurrency sweep (`results/v1_gpu_matrix_8_16_32_64_after_r2.json`):
  - `v1(py)` games/s at `cg=8/16/32/64`: `1.039 / 1.407 / 1.351 / 1.393`
  - `v1(py)` power at `cg=8/16/32/64`: `28.7 / 27.9 / 26.7 / 26.7 W`
  - `v1(graph)` games/s at `cg=8/16/32/64`: `1.219 / 1.244 / 1.224 / 1.228`
  - `v1(graph)` power at `cg=8/16/32/64`: `40.8 / 41.5 / 42.1 / 41.9 W`
- Long-stability (`rounds=3`, `results/v1_validation_workers_py_128_rounds3_after_r2.json`):
  - threads=1: games/s mean/std/worst = `0.932 / 0.116 / 0.777`
  - threads=2: games/s mean/std/worst = `0.994 / 0.093 / 0.909`
  - threads=4: games/s mean/std/worst = `1.143 / 0.024 / 1.110`
  - summary: `speedup_fixed_worker_min=8.358`, `v1_thread_gain=0.246`, `v1_p0_ratio_min=0.0`
- Long-stability (`rounds=3`, `results/v1_validation_workers_graph_128_rounds3_after_r2.json`):
  - threads=1: games/s mean/std/worst = `0.308 / 0.034 / 0.281`
  - threads=2: games/s mean/std/worst = `0.267 / 0.034 / 0.219`
  - threads=4: games/s mean/std/worst = `0.321 / 0.036 / 0.280`
  - summary: `speedup_fixed_worker_min=4.829`, `v1_thread_gain=0.043`, `v1_p0_ratio_min=0.0`
- Current acceptance status under main gates:
  - Progress is substantial versus pre-R2 baseline.
  - `fixed-worker >=10x` is still not met on the minimum scale pair.
  - `P0 ratio >= 0.9` is still not met on this RTX 3060 setup.

### Latest result snapshot (Windows, RTX 3060, R3 step1 slot-path reduction, aligned baseline, 2026-02-17)
- Baseline comparability correction:
  - Use the same batch scale for both sides in validator runs: `v0-batch-leaves=512` and `v1-inference-batch-size=512`.
  - The previous `v0-batch-leaves=128` run is kept as diagnostic only, not as final acceptance evidence.
- Semantic A/B gate:
  - `results/v1_child_value_ab_after_r3_step1_256.json`: PASS
  - `results/v1_child_value_ab_after_r3_step1_512.json`: PASS
- Fixed-worker regression (`results/v1_validation_workers_py_128_after_r3_step1_aligned512.json`):
  - `v0` workers 1/2/4: `0.067 / 0.032 / 0.029 games/s`
  - `v1(py)` threads 1/2/4: `0.715 / 0.913 / 0.953 games/s`
  - `speedup_fixed_worker_min`: `10.670` (meets `>=10x` gate in this run)
  - `v1_thread_gain`: `0.333` (fails `<=0.15` gate)
  - `v1_p0_ratio_min`: `0.0` (fails `>=0.9` gate)
- Fixed-worker regression (`results/v1_validation_workers_graph_128_after_r3_step1_aligned512.json`):
  - `v1(graph)` threads 1/2/4: `0.294 / 0.265 / 0.325 games/s`
  - `speedup_fixed_worker_min`: `4.815` (still below `10x`)
  - `v1_thread_gain`: `0.104` (passes CPU-scaling gate)
  - `v1_p0_ratio_min`: `0.0` (fails `>=0.9` gate)
- Concurrency sweep (`results/v1_gpu_matrix_8_16_32_64_after_r3_step1.json`):
  - `v1(py)` games/s at `cg=8/16/32/64`: `0.816 / 0.935 / 1.099 / 1.080`
  - `v1(graph)` games/s at `cg=8/16/32/64`: `0.396 / 0.369 / 0.329 / 0.276`
  - Observation: `graph` keeps higher util/power but lower games/s, indicating orchestration/shape mismatch rather than compute saturation.

### Latest result snapshot (Windows, RTX 3060, R3 step2 graph small-batch fallback, aligned baseline, 2026-02-17)
- Implementation note:
  - In `mcts_gpu.py`, graph backend now falls back to eager model forward for small batches and small remainder chunks (no new external flags/params), to avoid pathological pad-to-512 overhead.
- Semantic A/B gate:
  - `results/v1_child_value_ab_after_r3_step2_graphhybrid_256.json`: PASS
- Fixed-worker regression (`results/v1_validation_workers_graph_128_after_r3_step2_graphhybrid_aligned512.json`):
  - `v0` workers 1/2/4: `0.061 / 0.042 / 0.033 games/s`
  - `v1(graph-hybrid)` threads 1/2/4: `0.740 / 0.870 / 0.895 games/s`
  - `speedup_fixed_worker_min`: `12.209` (passes `>=10x`)
  - `v1_thread_gain`: `0.209` (still above `<=0.15` gate)
  - `v1_p0_ratio_min`: `0.0` (still below `>=0.9` gate)
- Concurrency sweep (`results/v1_gpu_matrix_8_16_32_64_after_r3_step2_graphhybrid.json`):
  - `v1(py)` games/s at `cg=8/16/32/64`: `0.878 / 1.010 / 0.887 / 0.928`
  - `v1(graph-hybrid)` games/s at `cg=8/16/32/64`: `0.941 / 0.864 / 0.812 / 0.989`
  - Compared to step1 graph (`0.396 / 0.369 / 0.329 / 0.276`), graph throughput is now in the same band as py under the same sweep settings.

### Latest result snapshot (Windows, RTX 3060, R4 finalize downshift, aligned baseline, 2026-02-17)
- Implementation note:
  - Added `v0_core.finalize_trajectory_inplace` to move trajectory target finalize and W/L/D counting into C++.
  - `self_play_gpu_runner.py` now uses this op and accumulates outcome counters on device, avoiding per-step `sum().item()` host sync.
- Semantic A/B gate:
  - `results/v1_child_value_ab_after_r4_256.json`: PASS
- Fixed-worker regression (`results/v1_validation_workers_py_128_after_r4_aligned512.json`):
  - `v0` workers 1/2/4: `0.100 / 0.045 / 0.025 games/s`
  - `v1(py)` threads 1/2/4: `0.891 / 1.009 / 0.764 games/s`
  - `speedup_fixed_worker_min`: `8.932`
  - `v1_thread_gain`: `0.132` (passes `<=0.15` in this run)
  - `v1_p0_ratio_min`: `0.0`
- Fixed-worker regression (`results/v1_validation_workers_graph_after_r4_graphhybrid_aligned512.json`):
  - `v0` workers 1/2/4: `0.054 / 0.038 / 0.030 games/s`
  - `v1(graph-hybrid)` threads 1/2/4: `0.851 / 0.757 / 0.921 games/s`
  - `speedup_fixed_worker_min`: `15.669` (passes `>=10x`)
  - `v1_thread_gain`: `0.082` (passes `<=0.15`)
  - `v1_p0_ratio_min`: `0.0`
- Concurrency sweep (`results/v1_gpu_matrix_8_16_32_64_after_r4.json`):
  - `v1(py)` games/s at `cg=8/16/32/64`: `0.557 / 0.897 / 1.110 / 1.005`
  - `v1(graph-hybrid)` games/s at `cg=8/16/32/64`: `1.017 / 1.013 / 1.038 / 0.943`
  - `graph-hybrid` remains at least comparable to `py` in this sweep and is stronger at low concurrency.

### Acceptance execution (Phase A automation)
- New unified acceptance script:
  - `tools/run_v1_acceptance_suite.py`
- Scope:
  - A/B semantic checks (`sims=256/512`)
  - fixed-worker regression (`py` + `graph`) with aligned baseline (`v0-batch-leaves=512`, `v1-inference-batch-size=512`)
  - concurrency matrix (`cg=8/16/32/64`)
  - smoke (`tests/v1/test_v1_tensor_pipeline_smoke.py`)
  - one consolidated summary JSON for gate review
- Full command (project sign-off run):
  - `conda run -n torchenv cmd /c "set PYTHONPATH=d:\CODES\liuzhou\build\v0\src;d:\CODES\liuzhou&& python tools/run_v1_acceptance_suite.py --device cuda:0 --seed 12345 --rounds 3 --repeats 3 --total-games 8 --mcts-simulations 128 --output-json results/v1_acceptance_suite_latest.json"`
- Smoke command (pipeline sanity):
  - `conda run -n torchenv cmd /c "set PYTHONPATH=d:\CODES\liuzhou\build\v0\src;d:\CODES\liuzhou&& python tools/run_v1_acceptance_suite.py --device cuda:0 --seed 12345 --rounds 1 --repeats 1 --total-games 4 --mcts-simulations 64 --output-json results/v1_acceptance_suite_smoke.json"`

### Code cleanup (Phase B first pass)
- Removed redundant Python-only finalize branches now replaced by C++ finalize path:
  - `v1/python/trajectory_buffer.py`: removed unused `finalize_games(...)` and `finalize_game(...)` methods.
- Kept a single active finalize path:
  - `v1/python/trajectory_buffer.py::finalize_games_inplace(...)` -> `v0_core.finalize_trajectory_inplace(...)`.

### Code cleanup (Phase B second pass, 2026-02-17)
- Removed obsolete ABI fallback branches in `v1/python/mcts_gpu.py`:
  - dropped legacy `batch_apply_moves` arity compatibility path;
  - dropped Python fallback loop for root PUCT allocation.
- Simplified evaluation helpers in `v1/python/mcts_gpu.py`:
  - removed unused private helpers (`_apply_dirichlet`, `_visits_to_policy`);
  - reduced `_evaluate_batch` and `_evaluate_values_only` return payloads to only active fields.
- Validation after cleanup:
  - `pytest tests/v1/test_v1_tensor_pipeline_smoke.py -q`: PASS.
  - `tools/ab_v1_child_value_only.py` (smoke config, `mcts=64`):
    - output: `results/v1_child_ab_redundancy_cleanup_smoke.json`;
    - action/value/WLD criteria: PASS.

### One-click wrapper
- Script: `scripts/validate_v1_gpu.cmd`
- Purpose: launch the above validator with default matrix and output path.

### Observability Metrics (2026-02-18)
To support the three requested diagnostics (Nsight timeline, per-step segment ratio, fixed-config stable run), new instrumentation and scripts were added.

Code instrumentation:
- `v1/python/mcts_gpu.py`
  - Added NVTX ranges:
    - `v1.root_pack_sparse_actions`
    - `v1.root_puct_allocate_visits`
    - `v1.root_sparse_writeback`
  - Added optional timing aggregation (`collect_timing`) for:
    - `root_puct_ms`
    - `pack_writeback_ms`
- `v1/python/self_play_gpu_runner.py`
  - Added NVTX ranges:
    - `v1.search_batch`
    - `v1.self_play_step_inplace`
    - `v1.finalize_trajectory_inplace`
  - Added optional segment timing output in `SelfPlayV1Stats`:
    - `step_timing_ms`
    - `step_timing_ratio`
    - `step_timing_calls`
  - New tracked keys:
    - `root_puct_ms`
    - `pack_writeback_ms`
    - `self_play_step_ms`
    - `finalize_ms`

New scripts:
- `tools/run_selfplay_workload.py`
  - Unified fixed-config runner (`mode=v0|v1`) with:
    - throughput stats,
    - GPU telemetry sampling (`util/power/memory/pstate/graphics_clock/sm_clock`),
    - optional v1 step timing export,
    - optional step breakdown plots (bar + pie),
    - optional stable-run plots (throughput line + GPU telemetry line).
- `tools/nsys_v0_v1_compare.py`
  - Runs Nsight Systems for `v0` and `v1` workloads.
  - Exports and parses `gputrace` + `cudaapisum`.
  - Produces:
    - timeline image (`v0 vs v1`) highlighting kernel/memcpy/memset distribution,
    - summary image for kernel count / memcpy count / sync calls / kernel gap / kernel fragmentation,
    - JSON with parsed metrics and deltas.

Recommended commands:
- Per-step segment breakdown (with plots):
  - `conda run -n torchenv cmd /c "set PYTHONPATH=d:\CODES\liuzhou\build\v0\src;d:\CODES\liuzhou&& python tools/run_selfplay_workload.py --mode v1 --device cuda:0 --seed 12345 --num-games-per-iter 8 --iterations 1 --mcts-simulations 128 --v1-threads 1 --v1-concurrent-games 8 --v1-child-eval-mode value_only --v1-inference-backend py --collect-step-timing --plot-step-breakdown --output-json results/v1_step_breakdown_latest.json"`
- Fixed-config stable run (`>=180s`, with plots):
  - `conda run -n torchenv cmd /c "set PYTHONPATH=d:\CODES\liuzhou\build\v0\src;d:\CODES\liuzhou&& python tools/run_selfplay_workload.py --mode v1 --device cuda:0 --seed 12345 --num-games-per-iter 8 --duration-sec 180 --mcts-simulations 128 --v1-threads 1 --v1-concurrent-games 8 --v1-child-eval-mode value_only --v1-inference-backend py --plot-stability --output-json results/v1_stable_run_180s.json"`
- Nsight timeline compare (`v0 vs v1`):
  - `conda run -n torchenv cmd /c "set PYTHONPATH=d:\CODES\liuzhou\build\v0\src;d:\CODES\liuzhou&& python tools/nsys_v0_v1_compare.py --device cuda:0 --seed 12345 --duration-sec 30 --num-games-per-iter 8 --mcts-simulations 128 --v0-workers 1 --v0-batch-leaves 512 --v1-threads 1 --v1-concurrent-games 8 --v1-inference-backend py --output-dir results/nsys_v0_v1_latest"`

### Observability Run Snapshot (Windows, RTX 3060, 2026-02-18)
#### Nsight timeline (`v0` vs `v1`)
- Nsight binary used:
  - `C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.2\target-windows-x64\nsys.exe`
- Run config:
  - `duration=30s`, `mcts_simulations=64`, `v0(workers=1,batch_leaves=512,backend=graph)`, `v1(threads=1,concurrent_games=8,backend=py)`.
- Artifacts:
  - Summary JSON: `results/nsys_v0_v1_20260218_run3/nsys_compare_summary.json`
  - Timeline image: `results/nsys_v0_v1_20260218_run3/nsys_timeline_v0_vs_v1.png`
  - Summary image: `results/nsys_v0_v1_20260218_run3/nsys_summary_v0_vs_v1.png`
- Key parsed metrics (from `nsys_compare_summary.json`):
  - `kernel_count`: `69,850 -> 532,806` (`7.63x`)
  - `memcpy_count`: `21,859 -> 15,447` (`0.71x`)
  - `sync_api_calls`: `20,569 -> 16,792` (`0.82x`)
  - `sync_api_total_ms`: `28,151.69 -> 2,424.14` (`0.086x`)
  - `kernel_gap_mean_us`: `624.995 -> 32.695` (`0.052x`)
  - `kernel_idle_ratio`: `0.979 -> 0.535`
- Workload throughput in the same Nsight run:
  - `v0`: `0.087 games/s` (`results/nsys_v0_v1_20260218_run3/v0_trace_workload.json`)
  - `v1`: `0.578 games/s` (`results/nsys_v0_v1_20260218_run3/v1_trace_workload.json`)

#### Fixed-config stable run (`v1`, 120s)
- Output JSON:
  - `results/v1_stable_120s_20260218.json`
- Stability plots:
  - `results/v1_stable_120s_20260218_stable_throughput.png`
  - `results/v1_stable_120s_20260218_stable_gpu.png`
- Step breakdown plots:
  - `results/v1_stable_120s_20260218_step_breakdown_bar.png`
  - `results/v1_stable_120s_20260218_step_breakdown_pie.png`
- Summary:
  - `run_elapsed=121.08s`, `iterations=19`, `games=76`, `positions=9,936`
  - `games/s=0.628`, `positions/s=82.06`
  - `gpu_util_avg=30.8%`, `gpu_power_avg=25.1W`, `gpu_p0_ratio=0.0`
- Step segment ratio:
  - `pack_writeback_ms`: `49.55%`
  - `self_play_step_ms`: `41.51%`
  - `root_puct_ms`: `8.31%`
  - `finalize_ms`: `0.64%`

## Launch-Reduction Execution Plan (2026-02-18, nsys-driven)
### Problem Statement
- Current bottleneck has shifted from host-device sync to launch overhead and kernel fragmentation:
  - `cudaLaunchKernel` API share: `~75%` (`~510k` launches / `30s`).
  - `cudaDeviceSynchronize` API share: `~16%` (not dominant, but still a tail-latency source).
- Step timing confirms two dominant stages:
  - `pack_writeback_ms`: `49.55%`
  - `self_play_step_ms`: `41.51%`
- Strategy switch:
  - Keep semantics unchanged (no pruning/surrogate shortcuts).
  - Prioritize reducing launch count and increasing per-launch work.

### Scope And Invariants
- In scope:
  - `v1/python/mcts_gpu.py` pack/writeback path.
  - `v0/src/bindings/module.cpp` + CUDA kernels used by self-play step.
  - Optional CUDA Graph capture for shape-stable segments.
- Out of scope:
  - Rule changes, action encoding changes, search-policy semantic changes.
- Invariants:
  - Root action/value parity against baseline A/B remains strict-pass.
  - Same benchmark comparability (`v0-batch-leaves=512`, `v1-inference-batch-size=512`).

### Milestones
#### R5-A (Highest priority): Pack/Writeback fusion
- Goal:
  - Replace fragmented tensor-op chain (`nonzero/masked_select/prefix-sum/gather/scatter/index_copy`) with 1-2 fused C++/CUDA ops.
- Implementation:
  - Add fused op(s) in `v0_core` for root sparse pack + policy writeback.
  - Keep current Python call contract in `mcts_gpu.py` stable.
  - Add micro-benchmark for op-level launch/time reduction.
- Expected impact:
  - Direct reduction of launch count in the largest stage (`~50%` share).

#### R5-B: Self-play step kernel consolidation
- Goal:
  - Reduce short-kernel chain inside `self_play_step` path by fusing active-slot transition and bookkeeping.
- Implementation:
  - Extend current `self_play_step_inplace` internal CUDA path to avoid fragmented gather/apply/scatter subchains.
  - Keep external API unchanged for runner integration.
- Expected impact:
  - Reduce second-largest stage (`~42%` share) and improve sustained GPU occupancy.

#### R5-C: Synchronize source cleanup
- Goal:
  - Remove avoidable syncs and isolate unavoidable ones.
- Implementation:
  - Run nsys with `cuda+nvtx`; inspect `cudaDeviceSynchronize` call stacks.
  - Remove accidental sync triggers in hot path (`.item()/.cpu()/explicit synchronize`) if present.
- Expected impact:
  - Lower long-tail stalls and throughput jitter.

#### R6: CUDA Graph capture on shape-stable segment
- Goal:
  - Batch repeated launch sequences into graph replay to cut launch API overhead.
- Implementation:
  - Capture only stable-shape core segment; keep eager fallback for small/remainder batches.
  - Reuse current graph-hybrid strategy and avoid external config expansion.
- Expected impact:
  - Additional launch-count reduction after R5-A/R5-B cleanups.

### Acceptance Gates For This Plan
- Semantic gates:
  - `tools/ab_v1_child_value_only.py` strict pass at `sims=256/512`.
- Throughput/efficiency gates (same aligned baseline):
  - `speedup_fixed_worker_min` should not regress vs latest accepted baseline.
  - `games/s` in stable run should improve after each milestone.
- Launch-fragmentation gates (nsys):
  - `cudaLaunchKernel` API share: target down from `~75%` to `<45%` (phase target).
  - Kernel launch rate: target down from `~17k/sec` to `<10k/sec` (phase target).
  - Keep `kernel_gap_mean_us` near current low level (no regression back to host-starved profile).

### Validation And Artifacts
- Required runs per milestone:
  - `tools/ab_v1_child_value_only.py` (`256/512`).
  - `tools/validate_v1_claims.py` (aligned 512, `py` + `graph-hybrid`).
  - `tools/sweep_v1_gpu_matrix.py` (`cg=8/16/32/64`).
  - `tools/run_selfplay_workload.py` (`duration>=120s`, step timing enabled).
  - `tools/nsys_v0_v1_compare.py` (`duration=30s`, same benchmark shape).
- Required artifact bundle:
  - JSON summaries + timeline plots + step-breakdown plots under `results/`.
