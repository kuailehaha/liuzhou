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

## Risks
- GPU tree memory fragmentation/overflow.
- Atomic update contention in backprop.
- Numerical drift from parallel reduction order.
- Debug complexity increase (GPU-only state).

## Validation Plan (acceptance aligned with user criteria)
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
    - v1 power uplift vs v0(worker=1)
    - v1 thread-sensitivity gain ratio
  - Emit JSON report and PASS/FAIL criteria summary.
- Recommended command (Windows):
  - `conda run -n torchenv cmd /c "set PYTHONPATH=d:\CODES\liuzhou\build\v0\src;d:\CODES\liuzhou&& python tools/validate_v1_claims.py --device cuda:0 --seed 12345 --rounds 1 --v0-workers 1,2,4 --v1-threads 1,2,4 --v1-concurrent-games 8 --total-games 8 --v0-mcts-simulations 512 --v1-mcts-simulations 512 --v0-batch-leaves 512 --v0-inference-backend graph --v0-inference-batch-size 512 --v0-inference-warmup-iters 5 --v0-opening-random-moves 2 --v0-resign-threshold -0.8 --v0-resign-min-moves 36 --v0-resign-consecutive 3 --with-inference-baseline --output-json results/v1_validation_latest_after_vec.json"`

### Latest result snapshot (Windows, RTX 3060, 2026-02-16)
- Config: `sims=512`, `total_games=8`, `v1-concurrent-games=8`
- `v0` (workers 1/2/4): `0.029 / 0.053 / 0.091 games/s`
- `v1` (threads 1/2/4): `0.085 / 0.085 / 0.086 games/s`
- Inference baseline: `~100% util`, `~44.9W`
- Criteria outcome:
  - `v1_speedup_ge_threshold`: PASS (`speedup=2.960` vs v0 worker=1)
  - `v1_thread_gain_le_threshold`: PASS (`0.007`)
  - `v0_worker_gain_ge_threshold`: PASS (`2.128`)
  - `v1_power_delta_ge_threshold`: FAIL (`+2.18W`, threshold `+5W`)

### One-click wrapper
- Script: `scripts/validate_v1_gpu.cmd`
- Purpose: launch the above validator with default matrix and output path.
