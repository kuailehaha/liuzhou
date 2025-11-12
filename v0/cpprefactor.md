# v0 C++ Refactor Plan

This document tracks the status of the v0 C++ rewrite and the remaining work needed to make it the default pipeline for self-play.

## Objectives

1. Re-implement the runtime-critical pieces of the Python stack (state handling, rules, move generation, tensor encoders, MCTS) in C++ without changing the public APIs.
2. Allow callers to switch between `src.*` and `v0.*` with a simple flag so we can cross-check behaviour.
3. Keep fast kernels (`fast_apply_moves`, `fast_legal_mask`, `project_policy_logits_fast`) close to the C++ MCTS core to avoid Python ↔ C++ traffic.
4. Maintain exhaustive parity tests (`tools/verify_v0_*`) so every module can be validated against the legacy Python implementation.

## Directory & Build Layout

```
v0/
├── include/v0/…        # headers for states, rules, moves, net, mcts
├── src/
│   ├── game/            # GameState + tensor batch utilities + fast_* kernels
│   ├── rules/           # rule engine port
│   ├── moves/           # move generator + action encoding
│   ├── net/             # input encoding + policy projection
│   └── mcts/            # batched MCTSCore implementation
└── python/
    ├── tensor_utils.py
    ├── state_batch.py
    ├── rules_tensor.py
    ├── fast_legal_mask.py
    ├── move_encoder.py
    └── mcts.py          # thin wrapper around v0_core.MCTSCore
```

The top-level `CMakeLists.txt` builds a single `v0_core` extension that links all of the above. Call `cmake -S v0 -B build/v0 …` followed by `cmake --build build/v0 --config Release` to rebuild the module.

## Migration Status

| Stage | Description | Status |
|-------|-------------|--------|
| 1. State Core | `GameState`, bitset-backed mark sets, serialization, tensor batch helpers | **Done** – `v0/include/v0/game_state.hpp` + `v0/src/game/game_state.cpp` + `tensor_state_batch.cpp`|
| 2. Rule Engine | Forming shapes, capture/removal/movement helpers | **Done** – `v0/src/rules/rule_engine.cpp` mirrors `src/rule_engine.py` |
| 3. Move Generator | `MoveRecord`, `ActionCode`, legal move enumeration, C++ `apply_move` | **Done** – exported via `v0_core` and wrapped by `v0/python/move_encoder.py` |
| 4. Tensor / Net | `states_to_model_input`, fast projection, local fast kernels | **Done** – fast kernels now live under `v0/src/net` / `v0/src/game` and Python copies no longer import `v1.*` |
| 5. MCTS Core | Batched selection/expansion/backprop + PyBind wrapper | **Done** – `v0/src/mcts/mcts_core.cpp` + `v0/python/mcts.py` (still CPU host tree, GPU forward supported) |
| 6. Glue / CLI / Docs | Python package, toggles, benchmarking scripts, documentation | **In progress** – `tools/verify_v0_mcts.py` supports `--fwd-device`; docs now reflect the self-contained state; remaining work tracked below |

## Next Steps / TODO

1. **GPU legality & move kernels**  
   - `fast_legal_mask` now has a CUDA path (`src/game/fast_legal_mask_cuda.cu`) that builds behind `-DBUILD_CUDA_KERNELS=ON` and is covered by `tests/test_fast_legal_mask_cuda.py`. Next step is to port `fast_apply_moves` so the expansion chain can stay on device.
2. **MCTSCore pointer/index hygiene**  
   - Finish replacing all long-lived `Node&` references with index lookups (selection / virtual loss / backprop) to avoid reallocation hazards when `std::vector<Node>` grows.
3. **Tensor batch direct-to-GPU path**  
   - `FromGameStates` still builds tensors on CPU then copies to CUDA. Add a GPU writer or serialization buffer to cut one memcpy when the forward device is GPU.
4. **CI / tooling**  
   - Re-enable the `tools/verify_*` scripts inside automated runs, add `pytest -k v0` target, and document the VS build requirements to avoid the Windows SDK `stdlib.h` detection failure.
5. **Documentation & README updates**  
   - Fold these notes into the main project README / training docs so users know how to switch to `v0` (`USE_V0_CPP=1`, `--fwd-device`, etc.).
6. **Self-play / training integration**  
   - Wire `v0.python.mcts.MCTS` into the self-play runner, add config flags, and benchmark end-to-end training loops on both CPU and GPU forward paths.

### GPU/CUDA Rewrite Plan

Goal: keep tensors on GPU from `FromGameStates → NN → legal mask → expanded child batch` so the only unavoidable host interaction is when we convert child batches back to `GameState` for storage in `nodes_`.

We will tackle this in four layers. Each layer should land with parity tests mirroring the CPU path (e.g., compare CUDA kernel outputs against the existing CPU implementation on random states).

#### Stage 1 – GPU-friendly Tensor Batch Construction (`tensor_state_batch.cpp`)
1. **Pinned host buffer**: add an optional path where we build into pinned CPU memory (using `torch::empty_pin_memory`) and immediately issue an async `cudaMemcpyAsync` into a CUDA tensor, synchronized on the default stream. This is the low-risk stepping stone.
2. **Direct CUDA kernel**: write a simple CUDA kernel that copies the `GameState` struct into the target tensors (board/marks/phase etc.), invoked via `at::cuda::CUDAStream`. This avoids the intermediate host tensor entirely.
3. **API surface**: expose `tensor_batch_from_game_states(states, device)` that chooses CPU/pinned/CUDA paths automatically, and update PyBind so `v0_core.tensor_batch_from_game_states(states, "cuda")` is supported.

#### Stage 2 – CUDA Legal Mask (`fast_legal_mask`)
1. **Kernel design**: break the CPU implementation into kernels per phase (placement, movement, selections). Each kernel should read the flattened board/marks arrays and write into the same metadata layout (`(B, total_dim, 4)`).
2. **Launch configuration**: use `(batch_size, cells)` style grids so each warp handles a subset of board cells; leverage shared memory for the 6x6 board to reduce global-memory reads.
3. **Fallback**: keep the current CPU code behind `#ifndef FAST_LEGAL_MASK_NO_MODULE` and compile CUDA version when `TORCH_CUDA_FOUND`. At runtime choose CUDA kernel only if `board.device().is_cuda()`.
4. **Testing**: extend `tools/verify_v0_state_batch.py` or add `tools/verify_v0_legal_mask.py` to compare CPU vs CUDA masks/metadata on random states.

#### Stage 3 – CUDA Fast Apply Moves (`fast_apply_moves`)
1. **Data layout**: operate on the same `(B, H, W)` int8 board plus bool marks arrays. Each kernel should:
   - copy parent board/marks into child slots,
   - apply action-specific mutations (placement/move/selection) using thread-blocks per parent,
   - update scalar tensors (phase, current_player, pending counters).
2. **Action metadata**: the CUDA kernel will read the `(N,4)` action tensor and parent indices exactly as the CPU version does; ensure we keep metadata in GPU memory throughout.
3. **Housekeeping**: maintain the `BatchInputs`/`BatchOutputs` structs for the CPU fallback, but add CUDA equivalents guarded with `#ifdef __CUDACC__`.
4. **Parity tests**: add a CUDA-aware variant of `tests/v1/test_fast_apply_moves.py` that runs both CPU and CUDA implementations and checks resulting states.

#### Stage 4 – MCTSCore Integration
1. **Device plumbing**: extend `MCTSConfig` with `tensor_device` (default `"cpu"`). During `ExpandBatch`, build the batch on that device, run NN/legal mask/move kernels without toggling back to CPU unless CUDA is unavailable.
2. **Stream management**: ensure `forward_cb_` (PyTorch model) and CUDA kernels share the same stream/guard (`at::cuda::CUDAGuard guard(tensor_device)`).
3. **Host conversion**: only when `child_states` are ready do we move the tensors back to CPU via `TensorStateBatch child_batch_gpu -> child_batch_cpu`; reuse the existing `ToGameStates`.
4. **Fallback logic**: if CUDA kernels are not compiled or the device is `"cpu"`, continue to use the current CPU path.

Each stage lands separately with documentation plus scripts demonstrating speedups (e.g., update `tools/verify_v0_mcts` or add `tools/benchmark_cuda_kernels.py` to show the CPU vs CUDA delta).

## Reference Commands

```bash
# CPU benchmark
python -m tools.verify_v0_mcts --samples 128 --sims 128 --batch-size 128 --fwd-device cpu --timing
[verify_v0_mcts] timing summary
legacy: avg=0.467s median=0.448s std=0.150s min=0.208s max=0.910s
v0: avg=0.074s median=0.066s std=0.027s min=0.037s max=0.176s
speedup: avg=6.91x median=6.40x min=2.96x max=18.73x

# GPU benchmark (forward runs on CUDA, tree still on CPU)
python -m tools.verify_v0_mcts --samples 128 --sims 128 --batch-size 128 --fwd-device cuda --timing
[verify_v0_mcts] timing summary
legacy: avg=0.464s median=0.447s std=0.158s min=0.219s max=1.198s
v0: avg=0.046s median=0.035s std=0.041s min=0.013s max=0.450s
speedup: avg=12.99x median=11.40x min=2.66x max=49.48x
```

Use `V0_MCTS_DEBUG=1` to print detailed traces from the C++ core when diagnosing expansion issues.
