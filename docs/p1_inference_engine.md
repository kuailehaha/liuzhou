# P1 推理引擎验收结论

本文记录 P1（C++ InferenceEngine + CUDA Graph 固定 batch=512）的验收结论与范围声明，方便后续阶段对齐预期。

## 功能验收

- ✅ `inference_backend={graph|ts|py}` 端到端生效，默认 `graph`
- ✅ TorchScript 导出在 self-play/train 路径内“每次 run 只发生一次”，避免 per-game trace/capture
- ✅ parity：Graph vs TS 的 forward parity test 通过；MCTS 的稳定对齐测试固定走 `py` backend

## 性能验收（待填数据）

- 非独占 GPU 环境下，建议多次测量取中位数
- Graph vs TS 的 `avg_ms` / `positions/s` / `speedup`：TBD（待回填）

## 范围声明

- P1 覆盖：推理后端切换 + Graph 固定 batch=512
- P1 不覆盖（部分已在后续迭代中完成）：
  - ~~EvalBatcher~~ → 已实现（`v0/src/mcts/eval_batcher.cpp`，异步批量推理 + 超时合并）
  - ~~多局合批~~ → MCTSCore 批量搜索已支持
  - C++ self-play 主循环（游戏循环仍在 Python，MCTS 核心在 C++）
  - encode/H2D 深度优化（TensorStateBatch 仍为 CPU→CUDA 拷贝，未使用 pinned memory/直接 GPU 写入）
