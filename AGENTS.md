# Agent Guide

本文件用于指导在本项目中进行快速迭代式开发。
目标是保持规则一致性，高效定位、修改、验证并交付功能。在开发过程中，应当充分发挥主观能动性进行深入理解。你不需要执行训练。在每次coding或bugfix后，你始终以简洁英文给出git commit的建议：[feature/bugfix/..]&[selfplay/train/eval/..]: `message(In A Sentence)`。你应当保持简洁的设计准则，如无必要，不增加开关和参数传递。返回时，请明确列出待测的功能清单。

## 资源和环境

如果是Linux系统，项目可用4张H20（CUDA_VISIBLE_DEVICES=0,1,2,3），256核CPU。
项目环境在/2023533024/users/zhangmq/condaenvs/naivetorch
项目训练一般使用./scripts/toy_train.sh
我们谈及优化，一般都是针对于v0 pipeline进行的。legacy只作功能性验证。

如果是Windows系统，项目可用1张RTX 3060，16核CPU。
项目环境启动方式为conda activate torchenv

如需测试运行速度，为保证准确性，应当先查看当前系统中有无正在运行的任务。你可以选择低占用资源进行测试。
可以通过查看git status来判断是否已经提交对话中的修改。

## 项目速览

- 六洲棋 AI 系统：规则引擎 + MCTS + 强化学习训练 + 人机对战前后端
- 规则权威来源：`README.md`、`rule_description.md`
- 两套实现：
  - Legacy：纯 Python（`src/`）
  - V0：C++/CUDA 核心（`v0/`）

## 目录导航

- `src/`：Legacy 纯 Python 核心规则与搜索
  - `game_state.py`：GameState/阶段枚举(Phase)/玩家枚举(Player)
  - `rule_engine.py`：阶段规则（落子/标记/移除/走子/提吃等）
  - `move_generator.py`：合法动作生成与落子/走子应用
  - `mcts.py`：Monte Carlo 树搜索（MCTSNode）
  - `neural_network.py`：神经网络模型（ChessNet）与状态张量转换
  - `train.py`：训练主脚本（自博弈 + 训练 + 评估循环）
  - `evaluate.py`：评估模块（MCTS vs Random 等并行评估）
  - `policy_batch.py`：训练阶段策略损失批量化（legal_mask/target_dense、combined logits、masked log_softmax），与 v0 动作编码一致；修改动作空间或 spec 时需同步 v0。
  - `random_agent.py`：随机 agent，用于测试和基准对比
- `v0/`：C++/CUDA 高性能实现
  - `train.py`：V0 训练主脚本（AlphaZero 风格循环：自博弈→训练→评估）
  - `generate_data.py`：独立自博弈数据生成脚本，输出 JSONL 至 `v0/data/self_play/`
  - `move_generator_reference.py`、`rule_engine_reference.py`：Python 参考实现，用于与 C++ 交叉验证
  - `python/`：Python 封装层
    - `mcts.py`：C++ MCTSCore 的 Python 封装，支持 graph/ts/py 后端
    - `self_play_runner.py`：V0 自博弈核心入口（`self_play_v0()`），多 GPU 并行
    - `move_encoder.py`：动作编码（`ActionEncodingSpec`、`action_to_index`/`decode_action_indices`）
    - `move_generator.py`：C++ 动作生成器的 Python 适配层
    - `state_batch.py`：`TensorStateBatch` 批量状态表示
    - `fast_legal_mask.py`：`encode_actions_fast` 封装（CUDA → CPU 回退）
    - `rules_tensor.py`：张量化规则工具函数
    - `state_io.py`：状态序列化/JSONL 读写
    - `tensor_utils.py`：张量工具（`TensorGameConfig`、`ensure_device`）
  - `src/`：C++/CUDA 源码
    - `game/`：GameState、TensorStateBatch、fast_legal_mask、fast_apply_moves（含 CUDA 内核）
    - `rules/`：C++ 规则引擎（镜像 `src/rule_engine.py`）
    - `moves/`：C++ 动作生成器（含动作编码/解码）
    - `mcts/`：MCTSCore（批量树搜索）、EvalBatcher（异步批量推理）
    - `net/`：网络编码（encoding）、InferenceEngine（CUDA Graph）、TorchScriptRunner、策略投影
    - `bindings/`：PyBind11 绑定模块（`v0_core`）
  - `include/v0/`：C++ 头文件
  - `CMakeLists.txt`：构建配置（pybind11 + LibTorch + 可选 CUDA）
- `backend/`：FastAPI 服务端（人机对战/推理接口）
  - `main.py`：应用入口与路由；`game_manager.py`：会话管理；`model_loader.py`：模型加载
  - `schemas.py`：Pydantic 请求/响应模型；`utils.py`：坐标转换与序列化工具
- `web_ui/`：前端界面（HTML/CSS/JS），棋盘渲染与交互
- `tests/`：单元/集成/对照测试
  - `check_rule_engine_cases.py`：规则引擎回归测试（1000+ 用例）
  - `test_batched_policy_loss.py`：策略损失批量化单测
  - `legacy/`：Legacy MCTS 与自博弈测试
  - `v0/`：V0 动作编码、MCTS、StateBatch 对照测试
  - `v0/cuda/`：CUDA 内核正确性与性能测试（fast_legal_mask、fast_apply_moves）
  - `integration/`：集成测试（自博弈流水线）
  - `random_agent/`：随机 agent 测试套件
- `tools/`：基准与诊断工具
  - `benchmark_self_play.py`：Legacy vs V0 自博弈性能对比
  - `benchmark_mcts.py`：Legacy vs V0 MCTS 吞吐量对比
  - `benchmark_cuda.py`：多设备（CPU/CUDA）MCTS 基准
  - `benchmark_inference_engine.py`：TorchScriptRunner vs InferenceEngine(CUDA Graph) 对比
  - `benchmark_eval_batcher.py`：EvalBatcher 合并推理基准
  - `benchmark_torchscript_forward.py`：Eager vs TorchScript 前向吞吐
  - `analyze_node_cache.py`：MCTS 节点缓存增长分析
  - `profile_self_play_gpu.py`：GPU 自博弈性能剖析
  - `run_test_matrix.py`：测试矩阵批量运行器
- `scripts/`：训练/运行辅助脚本
  - `toy_train.sh`：主训练脚本（V0 管线，含稳定性评估）
  - `optimized_train.sh`：大规模优化训练脚本
  - `train_loop.sh`/`train_loop.py`：训练循环调度器
  - `parallel_generate.sh`：多进程并行数据生成
  - `export_torchscript.py`：TorchScript 模型导出
  - `filter_decisive_jsonl.py`：过滤决定性样本（value ≠ 0）
  - `monitor_resources.sh`：资源监控；`kill_top_io.sh`：高 IO 进程清理
- `docs/`：项目文档
  - `complexity_estimation.md`：六洲棋复杂度估算
  - `legacy_tensor_pipeline.md`：Legacy 张量流水线参考
  - `p1_inference_engine.md`：P1 InferenceEngine + CUDA Graph 验收结论

## 规则一致性

- 规则解释以 `README.md`、`rule_description.md` 为准。
- 规则或动作编码变更时，以当前时间同步更新：
  - `src/` 中的规则与动作生成
  - `v0/` 中的参考实现/核心逻辑
  - 相关测试或对照脚本
  - `TODO.md`

## 工作流

- 规则改动：先更新规则文档与 Python 逻辑，再检查 v0 参考逻辑与对拍脚本。
- 训练/评估：`src/train.py` 或 `v0/train.py`；自博弈数据在 `v0/data/`。
- 训练稳定性与“是否训进去”的判断：见 `TRAINING_STABILITY.md`；推荐使用 `--eval_games_vs_previous` 看对上一迭代胜率。
- 性能与回归：`tools/benchmark_*` 与 `tests/` 中的对照脚本。

## Vibe Coding 计划内容

1. 目标与范围：本次要解决的问题/不做的部分。
2. 影响面：计划改动的模块/文件。
3. 不变量：需保持一致的规则/编码/行为（如阶段流转、动作索引）。
4. 风险点：可能破坏的边界条件或历史兼容。
5. 验证方式：具体要跑的测试/脚本或手工验收步骤。
6. 产出清单：新增或修改的文件列表。




