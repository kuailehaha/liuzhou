# v0 C++ Refactor Plan

## Objectives
- 完整复刻 `src/` 提供的推理/对局/MCTS 行为，重写为 C++ 实现（不含训练与测试框架本身）。
- 维持与现有 Python API 完全一致，允许调用层通过简单开关在 `src` ↔ `v0` 之间切换。
- 利用当前已完成的高频算子 (`fast_apply_moves`, `fast_legal_mask`, `project_policy_logits_fast`) 作为核心积木，最大化批量执行效率。
- 以测试比照 (`src` 作为黄金参考) 驱动迭代，每完成一个模块即运行差分测试确保零偏差。

## 目录 & 构建布局
- `v0/include/`：公共头文件（`game_state.hpp`, `rule_engine.hpp`, `move_generator.hpp`, `mcts.hpp`, `encoding.hpp` 等）。
- `v0/src/`：
  - `game/`：状态对象、序列化、基本工具。
  - `rules/`：形状检测、捕获/标记/移除逻辑。
  - `moves/`：合法走子生成与编码。
  - `net/`：输入编码、策略投影（复用 `project_policy_logits_fast.cpp`）。
  - `mcts/`：搜索树、批量节点扩展、策略温度处理等。
  - `python/`：PyBind11 桥接层（按功能拆分多个模块，便于增量替换）。
- 构建：沿用 `torch.utils.cpp_extension.load`（`setup_v0_ext.py`）+ CMakeLists，以可插拔方式生成 `v0.game`, `v0.rules`, `v0.moves`, `v0.net`, `v0.mcts` 五个扩展模块。

### CMake / PyBind11 现状
- 顶层 `v0/CMakeLists.txt` + `v0/src/CMakeLists.txt` 已就绪，默认开启 C++17 + `-fPIC`（或 `/bigobj`）。
- `pybind11_add_module` 生成 `v0_core` 基础占位模块，我们会在后续阶段将其拆成多个目标。
- 依赖：
  1. `pybind11`（提供 `pybind11Config.cmake`，可通过 `python -m pybind11 --cmakedir` 获取路径）。
  2. PyTorch（提供 `TorchConfig.cmake`，路径来源于 `python - <<<'import torch; print(torch.utils.cmake_prefix_path)'`）。
- 构建示例：
  1. 通过 Python 获取依赖路径  
     `python -c "import pybind11; print(pybind11.get_cmake_dir())"`  
     `python -c "import torch; print(torch.utils.cmake_prefix_path)"`
  2. 传入 CMake：  
     `cmake -S v0 -B build/v0 -DCMAKE_PREFIX_PATH="</pybind11/cmake>; </torch/cmake>"`  
     `cmake --build build/v0 --config Release`
- 生成的 `v0_core.*` 可由 `python -c "import v0_core; print(v0_core.version())"` 验证加载。

## 迁移顺序（必须严格按依赖推进）
1. **State Core**：Phase/Player enum、`GameState` 数据结构、复制/序列化工具。
2. **Rule Engine**：所有形状/连线检测与阶段推进方法（完全移植 `src/rule_engine.py`）。
3. **Move Generator**：合法招生成、动作编码/解码、与 rule engine 双向耦合。
4. **Tensor Encoders**：`states_to_model_input` 等网络输入准备、legal mask 快速路径封装。
5. **MCTS Core**：节点结构、批量 selection/expansion/backprop、与 fast kernels 集成。
6. **Glue & Switches**：Python 端轻量 wrapper、配置开关、工具脚本更新。

## 各阶段任务详情

### 1. State Core + Tensor batches (`v0/src/game`)
- `GameState`（持久化 6x6 board、标记集合、pending 计数器、move_count）。
- 标记集合使用紧凑 bitset (两枚 64-bit) 以便复制；提供 Python dict <-> C++ 转换。
- 工具函数：`is_board_full`, `get_player_pieces`, `has_reached_move_limit`, `get_winner`。
- 添加轻量断言：阶段与 pending 数一致（避免隐式 bug）。
- **现状**：
  - `v0/include/v0/game_state.hpp` 与 `v0/src/game/game_state.cpp` 已实现 Phase/Player 枚举、bitset 标记集与基础方法，并通过 `v0_core.GameState` 暴露到 Python（board/marks 属性支持列表/元组读写）。
  - `v0/include/v0/tensor_state_batch.hpp` + `v0/src/game/tensor_state_batch.cpp` 提供 `TensorStateBatch`（含 `to/clone`）、`tensor_batch_from_game_states`、`tensor_batch_to_game_states`，并在 `v0/python/state_batch.py` 暴露 Python 友好封装。
  - 验收：`python tools/verify_v0_state_batch.py --samples 128 --max-moves 60 --device cpu`（如需 CUDA 替换 `--device cuda:0`）；该脚本会随机生成局面后做 C++ ↔ Python round-trip 校验。

### 2. Rule Engine (`v0/src/rules`)
- 逐函数复刻：`generate_placement_positions`, `generate_mark_targets`, `generate_capture_targets`, `process_phase2_removals`, `apply_*` 系列。
- 形状检测：方块、直线、层叠；需与 `fast_apply_moves` 使用同一邻接逻辑（共用 `Directions`）。
- 提供纯函数接口：`GameState -> GameState/Vector<Action>`，禁止内部 new/delete。
- 单元比对：读取 Python 随机状态，分别调用 py/c++ 版本，序列化断言一致。
- **现状**：`v0/include/v0/rule_engine.hpp` 与 `v0/src/rules/rule_engine.cpp` 已完整移植 `src/rule_engine.py` 行为，`v0_core` 暴露与 Python 相同签名（含 phase1/phase3 兼容包装、captue/mark/forced removal 等分支），可直接用于差分测试。

### 3. Move Generator (`v0/src/moves`)
- 实现 `generate_all_legal_moves` 及三个特殊 helper（forced removal/no-move/counter removal）。
- 引入动作编码规格（与 `ActionEncodingSpec` 对齐），实现
  - 结构化 move <-> 扁平索引互转；
  - `encode_actions`、`decode_action_indices` C++ 版。
- 产生的 move/编码直接喂给 `fast_apply_moves`，减少 Python 粘合。
- **现状**：`v0/include/v0/move_generator.hpp`+`src/moves/move_generator.cpp` 提供 `MoveRecord`/`ActionCode`、全阶段生成、C++ `apply_move` 及编码函数。`v0_core` 暴露了 `generate_all_legal_moves_struct`/`generate_moves_with_codes` 等接口，`v0/python/move_generator.py` 现已封装成与 `src.move_generator` 兼容（含 `return_codes` 选项），并附带 `apply_move` 与 `_generate_moves_*` helper。

### 4. Tensor/Net (`v0/src/net`)
- C++ 实现 `states_to_model_input`：批量构造 BCHW tensor，与原 Python 完全等价。
- 包装已有 `project_policy_logits_fast`，暴露统一 API（含 legal mask 生成）。
- 处理 value head post-process（sigmoid/tanh/temperature 配置）及 dirichlet 噪声采样入口（可用 ATen 随机）。
- **现状**：`v0/src/net/encoding.cpp` + `v0_core.states_to_model_input` 现已提供 BCHW 组装；`v0/src/net/project_policy_logits_fast.cpp`、`v0/src/game/fast_legal_mask.cpp`、`v0/src/game/fast_apply_moves.cpp` 已本地化，不再依赖 `v1/*` 源。Python 侧也同步迁移了 `move_encoder.py`、`state_batch.py`、`rules_tensor.py`、`fast_legal_mask.py`，`v0/python/mcts.py` 全量引用这些副本即可跑完自博弈。新增 `postprocess_value_head` 与 `apply_temperature_scaling` 便于后续替换遗留 Python 温度/取值处理。

### 5. MCTS Core (`v0/src/mcts`)
- `_MCTSNode` 结构（value, visits, prior, children map / flat vector）。
- Batched selection：按 `batch_leaves` 并行推进，用 C++ 实现 UCB 计算、mask 过滤。
- Expansion：调用 C++ move generator + `fast_apply_moves` + NN 编码；支持缓存/staleness。
- 回传：值反转、虚拟损失、温度策略生成；输出与现有 Python `src.mcts.MCTS` / `v0.python.mcts.MCTS` 兼容。
- PyBind 对应类：`VectorizedMCTS`，方法 `search`, `advance_roots`, `set_root_states`, `clear_cache` 等。
- **现状（S2/S3/S4完成）**：
  - `v0/include/v0/mcts_core.hpp` + `v0/src/mcts/mcts_core.cpp` 提供 C++ `MCTSCore`，实现节点池化、PUCT selection、虚损、批量 expansion/backprop。Expansion 已串联 `TensorStateBatch` → `states_to_model_input` → forward 回调 → `project_policy_logits_fast` → `fast_legal_mask` → `batch_apply_moves_fast`，并保持编码动作排序、Dirichlet 注入与 Python 逻辑一致。
  - `v0_core.MCTSCore` / `MCTSConfig` 通过 PyBind 暴露，可在 Python 端调用 `set_forward_callback` 注册模型前向函数，回调只需返回 `(log_p1, log_p2, log_pmc, value)` 四个 Tensor，其他搜索逻辑完全在 C++ 执行。
  - `v0/python/mcts.py` 新增 `MCTS` 封装类，表面接口（`search`/`advance_root`/`reset`/`set_root_state(s)`/`set_temperature`）与 `src.mcts.MCTS` 对齐，便于上层替换。使用示例：

    ```python
    from v0.python.mcts import MCTS as V0MCTS
    from src.neural_network import ChessNet

    model = ChessNet(board_size=6)
    mcts = V0MCTS(model, num_simulations=256, device="cuda:0", add_dirichlet_noise=True)
    moves, policy = mcts.search(GameState())
    ```

  - 临时切换：在自博弈/评测脚本中用环境变量控制，示例：
    ```bash
    USE_V0_MCTS=1 python tools/self_play_runner.py ...
    ```
    （当 `USE_V0_MCTS` 为 `1` 时导入 `v0.python.mcts.MCTS`，否则沿用 `src.mcts.MCTS`。）
- **现状**：`v0/python/mcts.py` 搬运 `VectorizedMCTS` 并在 `_expand_nodes`/多根模拟中批量调用 `batch_apply_moves_fast`，节点扩展支持重用 C++ 子局面；新增 `MCTS` 包装类以兼容 `src.mcts.MCTS` API（`search`/`advance_root`/`reset`）。后续可将核心逻辑继续下沉到纯 C++。

### 6. Glue / CLI / Docs
- `v0/python/` 下提供 shim（例如 `from v0.python import move_generator as move_generator_v0`），可通过环境变量 `USE_V0_CPP=1` 选择。
- 更新 `tools/run_test_matrix.py`、`tools/benchmark_apply_moves.py` 支持 `--impl cpp`.
- 文档：`README`, `TEST_README` 添加构建、切换说明；`TODO` 列出剩余模块。

## 测试与验证
- 对每个模块新增 `tools/verify_v0_<module>.py`：
  - 随机生成 N 个游戏状态；
  - 比较 Python 与 C++ 输出（moves, next states, legal mask, policy logits, MCTS distributions）。
- 集成测试：复用 `python -m tools.cross_check_mcts`，扩展为 `--impl cpp`，跑固定随机种子。
- 性能基准：`tools/benchmark_apply_moves.py` + 新的 `benchmark_mcts_cpp.py`，输出对比日志。
- CI/手动：`pytest tests/v0 -k cpp`（可逐步补充）；全部测试通过后再设为默认实现。

## 风险 & 缓解
- **复杂阶段状态机**：先写 exhaustive assertions + golden tests，严格对照 `src/rule_engine.py`。
- **批量搜索同步**：调试时添加 `V0_MCTS_TRACE=1`，记录关键批次的 UCB/visit 变化。
- **Python <-> C++ 数据转换成本**：最大化批处理（B,Tensors）并且缓存编码结果。
- **维护成本**：保持 `v0/` 中的头/源拆分清晰，配合 `clang-format`/`cpplint` 统一风格。

---
准备工作完成后，建议按上述顺序逐模块推进，每完成一步即可运行对应 diff 测试，确保重构在“测试为纲”的节奏下进行。
- `tools/verify_v0_mcts.py`：MCTS 行为一致性。命令示例  
  `python tools/verify_v0_mcts.py --samples 25 --sims 64 --max-moves 60 --tolerance 5e-3`

脚本阈值：当前在 CPU 上 `max_linf ≈ 1e-3`，`max_l1 ≈ 3e-3`；若超出 5e-3 需调查差异来源。

切换说明：在自博弈/评测入口（self_play runner、evaluate runner 等）加入 `USE_V0_MCTS` 判断，默认为 0 时使用 `src.mcts.MCTS`，置为 1 时使用 `v0.python.mcts.MCTS`。这样可以在回归/性能对比中一键切换，也便于遇到问题时快速回退。


python -m tools.verify_v0_mcts --samples 128 --sims 128 --batch-size 128 --fwd-device cuda --timing
[verify_v0_mcts] timing summary
legacy: avg=0.464s median=0.447s std=0.158s min=0.219s max=1.198s
v0: avg=0.046s median=0.035s std=0.041s min=0.013s max=0.450s
speedup: avg=12.99x median=11.40x min=2.66x max=49.48x


python -m tools.verify_v0_mcts --samples 128 --sims 128 --batch-size 128 --fwd-device cpu --timing
[verify_v0_mcts] timing summary
legacy: avg=0.467s median=0.448s std=0.150s min=0.208s max=0.910s
v0: avg=0.074s median=0.066s std=0.027s min=0.037s max=0.176s
speedup: avg=6.91x median=6.40x min=2.96x max=18.73x
