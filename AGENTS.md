# Agent Guide

本文件用于指导在本项目中进行快速迭代式开发。
目标是保持规则一致性，高效定位、修改、验证并交付功能。在开发过程中，应当充分发挥主观能动性进行深入理解。你不需要执行训练。在每次coding或bugfix后，你始终以简洁英文给出git commit的建议：[feature/bugfix/..]&[selfplay/train/eval/..]: `message(In A Sentence)`。你应当保持简洁的设计准则，如无必要，不增加开关和参数传递。返回时，请明确列出待测的功能清单。

## 资源和环境

如果是Linux系统，项目可用4张H20（CUDA_VISIBLE_DEVICES=0,1,2,3），256核CPU。
项目环境在/2023533024/users/zhangmq/condaenvs/naivetorch
项目训练一般使用scripts\big_train_v1.sh
我们谈及优化，一般都是针对于v1 pipeline进行的。legacy、v0只作功能性验证。

如果是Windows系统，项目可用1张RTX 3060，16核CPU。
项目环境启动方式为conda activate torchenv

如需测试运行速度，为保证准确性，应当先查看当前系统中有无正在运行的任务。你可以选择低占用资源进行测试。
可以通过查看git log\git status来判断是否已经提交对话中的修改。
你的一切判断和计划应当从当前代码实现出发。得到问题后，你应当首先查看代码有无相关改动，在理解代码的基础上进行回复和下一步操作。
进行代码优化时，你应当从框架入手，调整收效最大的模块或流程，而不是改变实验条件或改变特定实现方式。
训练日志文件名时间戳默认使用 UTC；如需和本地时间对齐，按 UTC+8 换算后再进行“修改前后”归因分析。

## 项目速览

- 六洲棋 AI 系统：规则引擎 + MCTS + 强化学习训练 + 人机对战前后端
- 规则权威来源：`README.md`、`rule_description.md`
- 三层实现（当前主线为 V1）：
  - Legacy：纯 Python（`src/`），用于规则功能验证
  - V0：C++/CUDA 核心（`v0/`），作为底层能力与对照实现
  - V1：训练流水线（`v1/`），建立在 `v0_core` 与现有 CUDA 内核能力上的 staged 管线优化

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
- `v1/`：V1 训练流水线（当前主线）
  - `train.py`：V1 staged 训练入口（`all/selfplay/train/infer`）
  - `python/self_play_gpu_runner.py`：GPU 自博弈主循环（张量化输出）
  - `python/self_play_worker.py`：多进程分片自博弈 worker（process-per-GPU）
  - `python/train_bridge.py`：张量数据训练桥接（single/DP/DDP）
  - `python/mcts_gpu.py`：V1 Root-MCTS（含温度采样与 value_only 路径）
  - `python/trajectory_buffer.py`：轨迹缓存与终局 target 回填
  - `Design.md`：V1 设计、里程碑与验收记录
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
  - `big_train_v1.sh`：V1 大规模 staged 训练脚本（selfplay -> train -> eval -> infer）
  - `train_entry.py`：统一训练入口（`--pipeline {v0,v1}`）
  - `tournament_v1_eval.py`：V1 模型锦标赛评估脚本（多阶段淘汰/循环赛）
  - `local_train_v1_3iter.ps1`：本地小规模 3 轮快速回归脚本（Windows）
  - `toy_train.sh`：统一 toy 训练脚本（通过 `PIPELINE=v0|v1` 选择）
  - `toy_train_v1.sh`：V1 toy 训练包装脚本
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
  - `v1/` 中相关自博弈与训练数据路径（尤其是 target 语义）
  - 相关测试或对照脚本
  - `TODO.md`

## 工作流

- 规则改动：先更新规则文档与 Python 逻辑，再检查 `v0/` 参考逻辑与对拍脚本。
- 主训练入口：`scripts/big_train_v1.sh`（大规模）或 `scripts/train_entry.py --pipeline v1`（可控分阶段）。
- 主评估入口：`scripts/eval_checkpoint.py --backend v1`，并结合 `--v1_concurrent_games`、`--v1_opening_random_moves`。
- V1 staged 运行：`--stage selfplay/train/infer`；`train_strategy=ddp` 需采用 staged 方式。
- 性能与回归：`tools/validate_v1_claims.py`、`tools/sweep_v1_gpu_matrix.py`、`tests/v1/test_v1_tensor_pipeline_smoke.py`。
- 强度评估口径：`vs_random` 以 `win-loss` 视角使用（与锦标赛判胜口径一致），模型选择以锦标赛/Elo 为主。
- 训练基线约定：每轮训练基于 `latest` 连续推进，`best` 仅用于 gating 与棋力对照保存。

### 评估与选模约定（关键）

- `gating` 判定标准：仅要求候选模型在 `vs_previous(vs_best)` 中 `wins > losses`，和棋不计入通过条件。
- `vs_random`：默认使用确定性评估（`temperature=0`，`opening_random_moves=0`），用于回归探针和健康度监控。
- `vs_previous`：默认保持可分辨性（非零温度 + 采样），避免出现 `0-0-1000`/`0-500-500` 这类退化离散分布主导判断。
- 评估输出约定：`vs_random` 与 `vs_previous` 输出分离；用于 gating 的 `output_json` 只承载 `vs_previous` 结果，避免覆盖/串扰。
- 若调整以上默认评估行为，必须同步更新 `scripts/big_train_v1.sh` 的启动日志打印项，确保日志可直接追溯参数。

## 当前优先级（2026-02-26）

- 结论：V1 训练加速链路已完成，并具备单节点大规模训练能力。
- 当前主问题：棋力随数据/算力增长的转化效率不稳定，不再是纯吞吐瓶颈。
- 下一阶段目标：
  - 在 v1 训练链路加入 LR scheduler。
  - 继续优化 draw 倾向控制（不改规则）。
  - 采用锦标赛/Elo + `vs_random(win-loss)` + 自博弈有效样本的联合 KPI 评估。

## Vibe Coding 计划内容

1. 目标与范围：本次要解决的问题/不做的部分。
2. 影响面：计划改动的模块/文件。
3. 不变量：需保持一致的规则/编码/行为（如阶段流转、动作索引）。
4. 风险点：可能破坏的边界条件或历史兼容。
5. 验证方式：具体要跑的测试/脚本或手工验收步骤。
6. 产出清单：新增或修改的文件列表。
7. 实现闭环：实现内容和总结结果回归至哪些工作记录/计划文件。



