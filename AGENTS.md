# Agent Guide

本文件用于指导在本项目中进行快速迭代式开发。
目标是保持规则一致性，高效定位、修改、验证并交付功能。在开发过程中，应当充分发挥主观能动性进行深入理解。你不需要执行训练。

## 项目速览

- 六洲棋 AI 系统：规则引擎 + MCTS + 强化学习训练 + 人机对战前后端
- 规则权威来源：`README_zh.md`、`rule_description.md`
- 两套实现：
  - Legacy：纯 Python（`src/`）
  - V0：C++/CUDA 核心（`v0/`）

## 目录导航

- `src/`：核心规则与搜索
  - `game_state.py`：GameState/阶段枚举/玩家枚举
  - `rule_engine.py`：阶段规则（落子/标记/移除/走子/提吃等）
  - `move_generator.py`：合法动作生成与落子/走子应用
  - `mcts.py`、`neural_network.py`、`train.py`、`evaluate.py`
- `v0/`：C++/CUDA 版本核心与训练/数据脚本
- `backend/`：服务端（人机对战/推理接口）
- `web_ui/`：前端界面
- `tests/`：单元/集成/对照测试
- `tools/`：基准与诊断工具
- `scripts/`：训练/运行辅助脚本

## 规则一致性

- 规则解释以 `README_zh.md`、`rule_description.md` 为准。
- 规则或动作编码变更时，以当前时间同步更新：
  - `src/` 中的规则与动作生成
  - `v0/` 中的参考实现/核心逻辑
  - 相关测试或对照脚本
  - `TODO.md`

## 工作流

- 规则改动：先更新规则文档与 Python 逻辑，再检查 v0 参考逻辑与对拍脚本。
- 训练/评估：`src/train.py` 或 `v0/train.py`；自博弈数据在 `v0/data/`。
- 性能与回归：`tools/benchmark_*` 与 `tests/` 中的对照脚本。

## Vibe Coding 计划内容

1. 目标与范围：本次要解决的问题/不做的部分。
2. 影响面：计划改动的模块/文件。
3. 不变量：需保持一致的规则/编码/行为（如阶段流转、动作索引）。
4. 风险点：可能破坏的边界条件或历史兼容。
5. 验证方式：具体要跑的测试/脚本或手工验收步骤。
6. 产出清单：新增或修改的文件列表。

## 测试建议

- 基础：`pytest`（可用 `-m "not slow"` 跳过慢测）
- 规则回归：运行 `tests/check_rule_engine_cases.py`
- 性能/自博弈：`tools/benchmark_self_play.py` 等

## GPU 性能优化指南

当前需关注的两个性能异常：

### 自博弈：低功耗高显存

- **症状**：功耗仅 37% (185W/500W)，但显存占用高 (72GB+)，GPU-Util 99%
- **排查方向**：
  1. 使用 `torch.cuda.memory_summary()` 分析显存分布
  2. 检查 MCTS 节点缓存 (`v0/mcts.py`) 是否持有过多历史状态
  3. 确认批量推理是否充分利用 GPU 算力（batch size 是否过小）
- **优化思路**：节点缓存压缩/淘汰、增大 batch 提升计算密度

### 训练阶段：GPU 利用率低

- **症状**：功耗仅 26% (132W/500W)，GPU-Util 仅 11%
- **排查方向**：
  1. 使用 `torch.profiler` 或 `nsys` 定位瓶颈
  2. 检查 DataLoader 配置 (`v0/train.py`)：`num_workers`、`prefetch_factor`
  3. 确认 batch size 是否过小、是否存在频繁的 CPU-GPU 同步
- **优化思路**：增大 batch size、async data loading、减少同步点

### 相关脚本

- 自博弈性能测试：`tools/benchmark_self_play.py`
- 训练脚本：`v0/train.py`、`v0/generate_data.py`

## 提交前自检

- 变更是否与规则文档一致？
- legacy 与 v0 是否仍保持行为一致？
- 是否补充了对应的测试或对照脚本？


