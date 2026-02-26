# P1 推理引擎验收结论

本文记录 P1（C++ InferenceEngine + CUDA Graph 固定 batch=512）的验收结论与范围声明，方便后续阶段对齐预期。

## 功能验收

- ✅ `inference_backend={graph|ts|py}` 端到端生效，默认 `graph`
- ✅ TorchScript 导出在 self-play/train 路径内“每次 run 只发生一次”，避免 per-game trace/capture
- ✅ parity：Graph vs TS 的 forward parity test 通过；MCTS 的稳定对齐测试固定走 `py` backend

## 性能验收（已回填，TS 待补）

- 结果锚点：
  - `results/v1_gpu_matrix_8_16_32_64_py_graph_after_r1.json`
  - `results/v1_validation_workers_graph_128_after_r1.json`

### Graph vs Py（R1 后同配置对照，threads=1）

| concurrent_games | py positions/s | graph positions/s | graph相对py | py功耗均值 | graph功耗均值 |
|---|---:|---:|---:|---:|---:|
| 8  | 100.45 | 128.74 | +28.16% | 27.48W | 37.68W |
| 16 | 149.37 | 149.36 | -0.01%  | 24.77W | 38.95W |
| 32 | 154.65 | 147.05 | -4.91%  | 24.31W | 39.34W |
| 64 | 137.43 | 133.29 | -3.01%  | 24.35W | 39.44W |

结论：
- Graph 在低并发（cg=8）下吞吐提升明显，并显著提高 GPU 功耗与时钟占用。
- 中高并发（cg=16/32/64）下，吞吐优势不稳定，需结合 workload 选择后端。

### 固定工况补充（Graph 验证）

- 在 `results/v1_validation_workers_graph_128_after_r1.json` 中：
  - `speedup_best_v1_vs_v0_worker1 = 12.90`
  - `power_delta_best_v1_minus_v0_worker1_w = +20.35W`
- 说明 Graph 路径在 R1 阶段已达到“明显提速 + 更高计算负载”的验收目标。

### TS 路径状态

- TS 路径功能可用（见功能验收），但当前文档尚未补齐与 Graph 的同工况批量对照表。
- 后续补充项：在同 `concurrent_games`、同 `mcts_simulations` 下给出 TS/Graph/py 三路中位数对比。

## 范围声明

- P1 覆盖：推理后端切换 + Graph 固定 batch=512
- P1 不覆盖（部分已在后续迭代中完成）：
  - ~~EvalBatcher~~ → 已实现（`v0/src/mcts/eval_batcher.cpp`，异步批量推理 + 超时合并）
  - ~~多局合批~~ → MCTSCore 批量搜索已支持
  - C++ self-play 主循环（游戏循环仍在 Python，MCTS 核心在 C++）
  - encode/H2D 深度优化（TensorStateBatch 仍为 CPU→CUDA 拷贝，未使用 pinned memory/直接 GPU 写入）
