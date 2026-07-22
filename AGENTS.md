# Agent Guide

本文件用于指导 Agent 在本项目中进行可验证、可回退的快速迭代。目标不是“尽快改出代码”，而是在保持规则、动作编码、价值语义和训练接口一致的前提下，高效定位、修改、验证并交付。

默认约束：

- 以当前代码和可复现证据为起点，不凭记忆推断现状。
- 当前开发主线是 `v1/`；`v0/` 提供底层能力和对照实现，Legacy `src/` 主要用于功能与规则验证。
- 除非用户明确要求，否则不启动正式训练、大规模自博弈、长时间 GPU 基准或锦标赛。
- 保持简单设计；没有当前需求和证据时，不增加开关、参数透传、抽象层或兼容分支。
- 默认不提交、不推送、不合并、不删除分支；交付时给出一条简洁英文 commit 建议：`[feature|bugfix|refactor|docs]&[selfplay|train|eval|rules|ui|infra]: <one-sentence message>`。
- 返回时必须区分“已实现”“已验证”“未验证/受阻”，并明确列出仍需测试的功能。

## 资源和环境

Linux 训练机可用 4 张 H20（`CUDA_VISIBLE_DEVICES=0,1,2,3`）和 256 核 CPU，项目环境为 `/2023533024/users/zhangmq/condaenvs/naivetorch`，正式训练入口通常为 `scripts/big_train_v1.sh`。

Windows 本地机可用 1 张 RTX 3060 和 16 核 CPU，环境启动方式为 `conda activate torchenv`。

Apple Silicon 本地机为 MacBook Air `Mac17,3`（Apple M5、10 核 CPU、8 核 GPU、16 GiB 统一内存），`torchenv` 当前为 Python 3.10.20、PyTorch 2.9.1，MPS 可用但 CUDA 和 `v0_core` 不可用。该机器使用 `portable` V1 路径做 CPU 搜索/控制和 MPS 推理训练；不得把其 8 simulations 结果与 H20/CUDA 正式配置直接比较。

性能优化通常针对 V1 pipeline；V0 既是 V1 的底层依赖，也是功能/性能对照，Legacy 用作规则和行为参考。在其他机器上不得假定 CUDA、`v0_core`、checkpoint 或上述 Conda 环境可用。

如需测试运行速度，先查看当前系统有无正在运行的任务，选择低占用资源并记录竞争负载。可以通过 `git log` 和 `git status` 判断对话中的修改是否已经提交。所有判断和计划从当前代码实现出发；进行代码优化时先找影响最大的模块或流程，不通过改变实验条件制造收益。

## 开工前检查与信息来源

1. 先运行 `git status --short`，并按需查看 `git diff`、相关文件历史和最近提交。工作区中的既有改动默认属于用户；不覆盖、不回退、不顺手整理。
2. 找到真实入口、调用链、测试和配置，再制定方案。优先使用 `rg` 定位符号及所有调用者，不只阅读报错行或单个函数。
3. 信息权威顺序：
   - 六洲棋规则：`docs/rules.md`；
   - 当前行为与接口：代码、测试及实际运行结果；
   - V1 设计背景与历史验收：`v1/Design.md`；
   - 项目结论与待办：`MEMORY.md`、`TODO.md`；
   - `rule_description.md` 仅为兼容入口，不作为规则依据。
4. 涉及 PyTorch、FastAPI、PyBind11、CUDA、LibTorch 等版本相关 API 时，先从环境或依赖中确认实际版本，再查对应版本的官方文档；无法确认的结论必须标成未验证，不得静默套用记忆中的 API。
5. 如果任务只要求分析或诊断，交付根因、证据和建议，不擅自扩大为实现、训练或发布。
6. 多文件、跨层、规则/编码/训练语义或性能改动，编码前先写下本文末尾的计划要素；单纯文档或显然局部的修改可以简化计划。

## 项目级不变量与跨层契约

以下内容是评审和测试时的硬约束。若任务确实要改变其中任何一项，必须显式说明迁移范围、兼容策略和验证证据，不能把它当作局部实现细节。

- **规则与阶段语义**：保持 `PLACEMENT`、`MARK_SELECTION`、`REMOVAL`、`MOVEMENT`、`CAPTURE_SELECTION`、`FORCED_REMOVAL`、`COUNTER_REMOVAL` 等原子阶段及其合法流转、行动方切换、胜负与和棋判定一致。规则变更必须同步 Python 参考实现、C++/CUDA 路径、动作生成及回归用例。
- **动作编码**：默认动作空间为 220 维（36 placement + 144 movement + 36 selection + 4 auxiliary）。movement 索引采用 cell-major：`placement_dim + cell_idx * 4 + dir_idx`。`src/policy_batch.py`、`v0/python/move_encoder.py`、C++/CUDA legal mask、策略投影和 policy target 必须对齐。
- **价值视角**：训练 target 维持黑方为正、白方取反的约定；MCTS 子值只在 side-to-move 发生切换时翻转，同一玩家连续标记时不得机械翻转。终局判定、搜索回传与轨迹 finalize 必须使用同一语义。
- **张量契约**：显式检查 shape、dtype、device、合法动作 mask、有限值和空批次。CPU/CUDA、eager/graph、单卡/DP/DDP 等双路径优先做同输入对拍，不能仅验证“能运行”。
- **数据与接口兼容**：staged 入口、checkpoint、self-play payload 关键字段和读取端保持兼容。新增 metadata 可以是加法式变更；重命名、删除或改变字段语义必须同步所有生产者、消费者、旧数据读取测试与文档。
- **评估口径**：`vs_random` 只作为健康度探针；模型强度结论以固定条件下的 tournament/Elo/BT 或约定 gating 为主。不得用单次随机结果代替强度结论。
- **实验可比性**：随机种子、checkpoint、硬件、设备、batch/concurrency、MCTS simulations、warmup、运行时长及环境负载属于实验条件。比较前后结果时必须固定并记录，不能通过改变实验条件宣称代码优化有效。

如果某个契约目前只能靠文字维持且反复发生漂移，优先补充低成本的自动检查或对拍测试；历史债务与本次新增回归要分开报告，不制造“假绿”。

### 训练日志文件名与时间（UTC / UTC+8）

- 训练日志文件名中的 run_tag（如 `20260223_173954`、`20260309_170346`）为 **UTC** 时间（与当前运行机器一致，本机即按 UTC 生成）。
- 当前机器本地时间为 **UTC+8**（北京时间）。换算关系：**本地时间 = UTC + 8**；小时上可记为 **UTC 17 点 = 次日凌晨 1 点**（17+8=25→次日 01:xx）。
- 示例：
  - `logs/big_train_v1_20260223_173954.log` → UTC 2026-02-23 17:39:54 = 北京时间 2026-02-24 01:39:54（凌晨启动）。
  - `logs/big_train_v1_20260309_170346.log` → UTC 2026-03-09 17:03:46 = 北京时间 2026-03-10 01:03:46（凌晨启动）。
- 如需做“修改前后”归因分析，先用上述关系把 run_tag 换成本地时间再对照操作时间或 git 提交时间。

## 项目速览

- 六洲棋 AI 系统：规则引擎 + MCTS + 强化学习训练 + 人机对战前后端
- 规则权威来源：`docs/rules.md`（`rule_description.md` 仅保留为兼容入口）
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
  - `long_train_portable_mps.py`：Apple Silicon portable MPS 可恢复长训编排（deadline、checkpoint/optimizer、replay、周期评估和 best retention）
  - `run_long_train_mps.sh`：以项目 `torchenv` 和 `caffeinate -ims` 启动本地 MPS 长训
  - `export_torchscript.py`：TorchScript 模型导出
  - `filter_decisive_jsonl.py`：过滤决定性样本（value ≠ 0）
  - `monitor_resources.sh`：资源监控；`kill_top_io.sh`：高 IO 进程清理
- `docs/`：项目文档主树
  - `index.md`：项目文档首页与阅读导航
  - `quickstart.md`：构建、训练、评估与人机对战快速开始
  - `architecture.md`：系统组成、目录职责与三代实现演进
  - `method.md`：项目方法论、模型输出、训练与评估口径
  - `gameplay_system.md`：人机对战前后端说明
  - `rules.md`：正式规则文档
  - `results.md`：里程碑与阶段成果
  - `faq.md`：常见问题

## 规则一致性

- 规则解释以 `docs/rules.md` 为准；`rule_description.md` 仅作兼容入口。
- 规则或动作编码变更时同步更新：
  - `src/` 中的规则与动作生成
  - `v0/` 中的参考实现/核心逻辑
  - `v1/` 中相关自博弈与训练数据路径（尤其是 target 语义）
  - 相关测试或对照脚本
  - 规则文档、设计记录和确实受影响的待办

## 实现纪律：测试先行、小步交付

1. 先定义可观察行为和验收条件，再修改实现。Bug 修复应先构造最小复现，并尽量补一条会因原 Bug 失败的回归测试；确认失败原因正确后再修复。
2. 每次只处理一个逻辑问题。优先交付能独立验证的薄切片；跨 Python/C++/CUDA 时，可先固定契约和参考用例，再逐层实现与对拍。
3. 先写最简单、显然正确的实现。不要为假想需求提前抽象；不要在功能修改中夹带格式化、邻近重构、参数清理或“顺手优化”。发现无关问题只记录，不扩大当前 diff。
4. 每个切片完成后先跑最相关的快速测试，再进入下一片。重构只能在行为测试为绿后进行，且重构前后输出与契约保持一致。
5. 测试优先覆盖行为、边界和跨实现一致性，少测试内部调用次数。只有在真实依赖昂贵、不可控或无法构造时才使用 mock；硬件路径不能用 mock 结果替代真实 CUDA 验证。
6. 文档、纯配置、探索性原型不强制机械执行红绿循环，但仍要做与风险匹配的验证。探索代码若要进入主线，必须补齐正式契约和回归测试。
7. 失败、回退与兼容路径必须可观察。不得通过吞异常、静默 CPU fallback、过滤坏样本或降低测试强度制造成功；如确需 fallback/过滤，必须记录触发次数、比例和原因。

## 系统化调试

遇到 Bug、测试失败、训练异常、NCCL/CUDA 问题或性能回退时，按以下顺序处理：

1. **复现并保存症状**：记录最小命令、输入/seed、环境、完整错误和首次异常位置。不能稳定复现时先增加诊断信息，不猜修复。
2. **检查变化与边界**：查看相关 diff、提交、依赖和环境差异；在 `shell → v1/train.py → worker → v0_core/CUDA → payload → train_bridge` 等边界核对输入、输出、shape/device/dtype、配置和状态。
3. **反向追踪数据流**：从坏值、错误状态或慢点向上追到最初产生处；优先修根因，不在下游堆补丁。
4. **对比正常路径**：寻找仓库内可工作的参考实现或旧基线，完整列出差异，尤其关注 CPU/CUDA、v0/v1、eager/graph、single/DDP 的分叉。
5. **一次验证一个假设**：明确写出“根因可能是 X，因为证据 Y”，做能证伪它的最小实验；保持其他变量固定。实验失败后回到证据，不叠加多个猜测性修改。
6. **修复并防回归**：在源头实施单一修复，运行原始复现、对应回归测试和必要的邻近测试。

若连续三次修复尝试都未命中，应停止继续打补丁，重新审视共享状态、接口边界或架构假设，并向用户报告已验证/已排除的内容。最终将问题分类为代码回归、环境不可用、依赖/硬件问题、数据问题或证据不足，不把环境阻塞误报为代码已坏，也不把无法运行误报为已通过。

## 验证矩阵与完成标准

验证按改动影响面选择，先小后大；下面是常用入口，不要求对每个改动无差别全跑：

- 文档/脚本静态检查：`git diff --check`，并人工核对命令、路径、链接和平台语法。
- Python 状态、阶段、价值桶和策略损失：`python -m pytest tests/test_game_state_phase_gate.py tests/test_value_bucket_encoding.py tests/test_batched_policy_loss.py -q`
- V1 张量/self-play/train bridge：`python -m pytest tests/v1/test_v1_tensor_pipeline_smoke.py -q`
- Legacy/V0 行为：`python -m tools.run_test_matrix --group legacy --group v0 --fail-fast`
- 动作编码或 legal mask：至少运行 `python -m pytest tests/v0/test_actions.py -q`；若 `v0_core` 未构建而测试被 skip，不能声称跨层对齐已验证。
- CUDA kernel：在可用 GPU 上运行对应的 `tests/v0/cuda/` 测试，并先做 CPU/CUDA 正确性对拍；带 `slow` 的性能用例按需显式执行。
- V1 GPU 验收或性能声明：使用 `tools/smoke_v1_gpu_pipeline.py`、`tools/validate_v1_claims.py` 或 `tools/run_v1_acceptance_suite.py` 中与任务匹配的入口，保留输出 JSON/日志锚点。
- 后端/UI：运行相关 Python 测试后启动 `uvicorn backend.main:app --reload`，手工覆盖新建对局、合法/非法动作、阶段切换、AI 返回和错误响应；模型或 GPU 不可用时明确未覆盖项。
- 影响广或准备集成时：在环境支持的前提下运行 `python -m pytest -m "not slow"`，再按改动补 CUDA、集成或基准测试。

完成声明必须有本轮修改后的新鲜证据：说明执行的准确命令、退出码或通过/失败/skip 数、测试环境和未运行原因。部分测试通过只能证明对应范围；lint、import 成功、dry-run 或旧日志不能替代构建、行为测试和实际运行。测试失败时如实报告，不为了收尾修改测试期待、放宽阈值或隐藏失败。

## 性能、训练与运行证据

- 性能工作遵循“基线 → profile 定位瓶颈 → 单点修改 → 同条件复测 → 正确性回归”。先优化实测热点和主流程，不凭代码观感做微优化。
- 基准前检查 GPU/CPU/IO 占用，固定环境与实验条件，完成 warmup，并尽量多次重复；至少报告原始数值、聚合口径（如 median/p95）和波动，而不只报告倍率。
- 吞吐提升必须同时通过语义对拍和数值健康检查；更快但规则、legal mask、policy target、value 符号、有效样本率或模型输出改变，不算等价优化。
- 训练/评估改动需记录 run_tag、commit、checkpoint、seed、完整命令/配置、硬件、核心指标和 artifact 路径。分析日志时按前述 UTC/UTC+8 规则归因。
- 大型 checkpoint、JSONL、trace、profile 和临时日志不提交到 Git。交付中引用其路径和关键摘要，并确认不含密钥或敏感环境信息。
- 修改训练脚本时优先提供 smoke/toy 或 stage 级验证方案；正式 H20 长跑属于独立验收，不因本地代码测试通过而宣称训练效果已验证。

### 冒烟训练与 `vs_random` 验收

- 训练型冒烟按“短 pilot 验证链路和信号 → 小规模调参 → 冻结配置 → 从头正式运行 → 独立评估”的顺序执行。正式计时开始后不再改参数；若因错误重启，保留失败证据并明确重新计时边界。
- 正式运行前预先指定验收 checkpoint，默认使用到达时间/样本预算后的最后一个完整 checkpoint。若评估多个中间 checkpoint 后再选最优，必须标明为 post-hoc screening，并另用独立样本复验，不能与预注册的 final-checkpoint 结果混报。
- `vs_random` 阈值必须写清分母和判定式；默认报告 `wins / total_games`，平局属于未胜。固定并记录总局数、双方颜色分配、MCTS simulations、温度、随机开局、并发、设备、fallback 和耗时；条件允许时同时给出逐颜色 W/L/D。
- 1000 局等有限样本除点估计外应报告二项比例置信区间（默认 Wilson 95%）。只有点估计超过阈值时称“观察值通过”；若要主张有统计余量，置信区间下界也应超过阈值。
- 评估 seed 必须可设置并写入报告。若当前入口使用时间 seed 或未持久化 seed，交付时必须明确标为复现限制，不能把一次随机结果描述为完全可复现。
- 外层循环反复调用 staged `selfplay`/`train` 时，显式持久化并审计 checkpoint 与 optimizer state；首轮后每轮都应成功加载。外层迭代号、输入 checkpoint、输出 checkpoint 和配对 metrics 必须有可追踪映射，不能只依赖可能在每次进程启动时重置的 checkpoint 内部 `iteration` 字段。
- 正式训练健康检查至少汇总：游戏数、位置数、决定性/和棋比例、loss 首尾与近期窗口、optimizer 连续性、非有限值、过滤样本和设备/MCTS fallback。训练内自博弈胜负不能替代独立 `vs_random` 或 tournament 结果。

### Apple Silicon portable MPS 长训

- 当前 M5 的测量结果、冻结参数、checkpoint 和命令以 `MEMORY.md` 最新日期段为准；这些参数是该机器的本地基线，不自动推广到其他 Apple Silicon、CUDA 或 H20 环境。正式长训开始后保持 self-play games/concurrency、batch、simulations、replay、LR、温度和开局日程不变。
- 调整 MCTS simulations 时分别报告 self-play 数据吞吐/和棋率与固定模型评估收益，并用同 checkpoint、seed、局数、并发做倍率对照。100 局 screening 只能决定是否扩大实验；若验收口径为 500 局，参数结论必须由 500 局复核，训练参数还需 equal-wall-clock A/B，不能用更昂贵的评估搜索冒充训练改进。
- 使用 `scripts/long_train_portable_mps.py` 运行长任务时，保留 `state.json`、`events.jsonl`、`final_summary.json`、逐阶段 metrics、当前/best checkpoint、optimizer state 和滚动 replay。重启用同一 run directory 加 `--resume`，恢复后审计外层 iteration、checkpoint SHA、optimizer 加载和 replay 输入，不能只确认进程重新启动。
- `best_vs_random.pt` 与 `best_model.pt` 语义不同：前者按固定 RandomAgent 口径筛选，后者由候选对 incumbent 的独立 head-to-head gate 更新。不要用随机对手最高点替代 incumbent gating，也不要将训练内胜负当作两者之一。
- 500 局评估必须使用偶数局、固定并持久化 seed、精确均分 challenger 黑白并报告逐颜色 W/L/D。若目标为 `99%` raw wins，则判定式为 `wins >= 495/500`，平局不计胜；筛选后还要用独立 seed 再跑 500 局确认，并保留 Wilson 95% 区间。
- macOS `caffeinate -ims` 只在进程存活时抑制 idle/system/disk sleep，不抑制 display sleep，也不能绕过合盖硬件条件。Apple silicon 合盖长跑必须接交流电、外接显示器及外接键盘/鼠标，并用 `--require-external-display` 做启动前检查；缺任一条件时停止，不使用持久化 `sudo pmset disablesleep` 规避。
- 用户明确选择开盖长跑时，可以不要求外接显示器，但必须接交流电并保持上盖打开；需记录该选择，且不得把结果表述为合盖耐久验证。Codex 执行环境中的直接 `nohup` 可能被进程回收器清理，跨断联运行优先使用 `RunAtLoad=true`、`KeepAlive=false` 的一次性用户 LaunchAgent，并核对任务进程、日志、状态文件和 `caffeinate` 断言后再交付。
- 用户要求在已确认目标后继续同一长训时，只移除 `--stop-on-target`，保持原 run directory、模型/optimizer 对、replay、冻结配置和原 deadline；恢复后必须确认 `run_resumed` 事件、下一轮提交、optimizer/replay 连续性、SHA、fallback 和非有限样本，不能重新初始化或偷偷延长墙钟预算。
- 长训产物保留在 ignored 的 `tmp/`/`logs/` 路径，不纳入 Git。短 smoke 的 4/20 局结果只验证控制流、seed/颜色统计、best retention、恢复与 fallback 审计，不得作为 500 局目标或模型强度证据。

## Git、分支收尾与交付

- 修改前后检查 `git status --short`；交付前使用 `git diff -- <exact-paths>` 逐文件阅读本次路径的 diff，确认没有用户改动、临时产物或无关格式化混入。
- 若用户要求提交，按一个逻辑行为组织原子提交，使用精确路径暂存，避免在混合工作区执行 `git add .`。提交前运行与该提交匹配的验证，提交后核对 `git show --stat --oneline HEAD` 和工作区状态。
- 测试失败时不进入合并/PR 收尾。合并后需在合并结果上重新验证；未经明确授权不 push、不 force-push、不创建 PR、不合并。
- 删除分支、worktree、checkpoint、数据或运行产物前，先解析精确目标并取得明确确认。只清理由自己创建且确定归属的 worktree/临时文件。
- 稳定、跨任务可复用的工程规则可以回写本文件；单次实验参数、临时结论和日志细节应写入 `v1/Design.md`、`MEMORY.md`、`TODO.md` 或专门报告，避免 `AGENTS.md` 膨胀为流水账。

最终回复至少包含：改动摘要、验证证据、未验证/受阻项、风险或待测功能、修改文件，以及一条 commit 建议。没有新鲜验证证据时，不使用“已完成、已修复、全部通过”等表述。


## Vibe Coding 计划内容

1. 目标与范围：本次要解决的问题/不做的部分。
2. 影响面：计划改动的模块/文件。
3. 不变量：需保持一致的规则/编码/行为（如阶段流转、动作索引）。
4. 风险点：可能破坏的边界条件或历史兼容。
5. 验证方式：具体要跑的测试/脚本或手工验收步骤。
6. 产出清单：新增或修改的文件列表。
7. 实现闭环：实现内容和总结结果回归至哪些工作记录/计划文件。
8. 结果回归：只将稳定、可复用的原则写入 `AGENTS.md`；运行特定结论写入对应设计、记忆或待办文档。
