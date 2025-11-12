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
   - Current fast kernels only run on CPU. Port `fast_legal_mask` and `fast_apply_moves` to CUDA (or add CUDA siblings) so we can keep tensors on GPU during expansion.
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
