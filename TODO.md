# TODOS

TODO以产生时间为准。最近审查：2026.02.20

### 2026.02.20

#### 已完成加速里程碑（v1 主线）

- [x] 完成 v1 staged 流水线（`v1/train.py`，`--stage all/selfplay/train/infer`）
- [x] 完成多 GPU 自博弈 process-per-GPU 分片与 shard 合并（`v1/python/self_play_worker.py`）
- [x] 完成训练阶段 DataParallel/DDP 路径与按 rank 预分片 H2D（`v1/python/train_bridge.py`）
- [x] 完成统一训练入口与大规模训练脚本（`scripts/train_entry.py`、`scripts/big_train_v1.sh`）
- [x] 完成 v1 评估后端与并发参数贯通（`scripts/eval_checkpoint.py --backend v1`）
- [x] 完成自博弈有效性信号输出（`decisive_game_ratio`、`draw_game_ratio`、`value_target_summary`）

#### RWD 奖励机制待办（下一阶段主目标）

- [ ] RWD-0（基线冻结）：固定当前配置跑 3 轮 selfplay-only，记录 `decisive_game_ratio`、`draw_game_ratio`、`value_target_summary.nonzero_ratio`、`games_per_sec`、`positions_per_sec`
- [ ] RWD-1（固定公式首版）：在 `v1/python/self_play_gpu_runner.py` + `v1/python/trajectory_buffer.py` 定义并回填逐步奖励回报，在 `v1/train.py` 接入统计；公式固定为  
      `terminal=result_from_black`，`dense_piece=0.20*(soft_t-soft_{t-1})`，`anti_draw=-0.002 if moves_since_capture>=24 else 0`，`step_reward=clip(dense_piece+anti_draw,-0.2,0.2)`，`target_return_t=terminal+sum(step_reward from t to T-1)`，`white_target=-black_target`
- [ ] RWD-2（门槛验收，3 轮均值）：`decisive_game_ratio >= 1%`，`value_target_summary.nonzero_ratio >= 1%`，`draw_game_ratio <= 99%`，`positions_per_sec >= baseline * 85%`
- [ ] RWD-3（放量验收）：接入 `scripts/big_train_v1.sh` 后，RWD-2 指标不回退，且分片/显存稳定性不引入新的 capture 类崩溃

### 2025.10.20

- [x] 重构逻辑：将形成square和line后capture等action拆分为原子化的move+capture action

### 2025.10.22

- [x] batch推理叶子节点
- [x] 调整log verbose
- [x] 策略头调整，对应各策略头语义
- [x] 试跑，时间分析
- [x] 分析强化学习原理

### 2025.10.23

- [x] 添加batch_K：自博弈和eval阶段批量推理不同的节点 

### 2025.10.25

- [x] 预留虚损

### 2025.10.26
- [x] 置换策略头 12 在 move动作上的功能 
- [x] 修复 no_legal_moves 的异常
- [x] 多进程推理，添加numworkers功能
- [x] 实现 RandomAgent 的对局，并获取胜率数据
- [x] 合并参数：self_play_games、self_play_games_per_worker
- [x] 整理test file
- [x] 张量化实现自博弈过程 —— 已由 v0 C++ 管线替代：MCTSCore (C++) 实现批量树搜索，fast_legal_mask/fast_apply_moves 提供 CPU+CUDA 内核
- [ ] 并行化DDP训练或实现"自对弈集群+训练服务器"模式
- [x] ~~确认代码实现逻辑~~ （早期探索性条目，已通过 tests/ 体系充分验证）
- [ ] 模型结构和超参调优（持续性任务）
- [ ] 在 v1 自博弈中实现中间奖励并检查效果 —— 当前 `v1/python/self_play_gpu_runner.py` + `v1/python/trajectory_buffer.py` 以终局目标回填为主，尚未按步累计奖励回报

### 2025.10.27

- [x] 实现前端，人机对弈功能
- [x] 修补bug：同时square和line只能提掉一个

### 2025.10.28

- [x] eval log添加 胜负和 信息
- [x] 前端：添加敌方上一步信息

### 2025.10.29

- [x] 添加前端对弈value值显示（胜率信息）

### 2025.10.30

- [x] 创建 v1 张量化重写骨架目录
- [x] 实现 v1/game 张量化状态转换与规则逻辑
- [x] 完成 v1/game/move_encoder.py 动作编码与解码
- [x] 写出 VectorizedMCTS 主流程并替换自博弈调度
- [x] 打通 v1/self_play -> v1/train 张量化训练闭环（当前由 `v1/train.py` + `v1/python/train_bridge.py` 承接）
- [x] 为 v1 分支新增测试与对照验证脚本

### 2025.10.31

- [x] 对拍传统管线和新管线（MCTS部分） —— tests/v0/test_mcts.py (move coverage 对照)、test_actions.py (动作编码对照)、tools/benchmark_mcts.py 等已覆盖

### 2025.11.1

- [x] 梳理 legacy 流程关键函数并整理 inputs/outputs 对照表
- [x] 设计并实现 states_to_model_input 及 _board_planes/_mark_planes/_phase_planes
- [x] B=1 parity test matches legacy state encoding
- [x] 复核 ActionEncodingSpec/encode/decode 索引约定并补齐批量掩码 helper
- [x] 在 project_policy_logits 中完成 logits 拆分、掩码与兜底归一化逻辑
- [x] 如有需要新增 batched gather/reshape 等张量工具函数
- [x] 编写 cross-check 脚本对比两套管线的输入张量与动作分布
- [x] 更新自博弈/MCTS/训练脚本调用路径切换至 v1 张量化实现
- [x] README/TODO 记录切换方式与验证状态（当前主线以 v1 流水线为准）
- [ ] Future: unify policy heads (head1=place/move-to, head2=move-from, head3=mark+all removals) and drop phase one-hot channels when legacy parity is secured.

### 2025.11.2

- [x] ~~debug: python -m tools.cross_check_mcts~~ （一次性调试任务，对拍功能已由 tests/v0/test_mcts.py、test_actions.py 覆盖）

### 2025.11.3 — MCTS 性能优化

> 原始 profile：waves=64 sel=0.67% moves=0.78% encode=11.96% fwd=3.09% pri=35.44% apply=48.05%

- [x] 优化 project_policy_logits 的实现 —— C++ 实现 `v0/src/net/project_policy_logits_fast.cpp`，仅对合法动作做 masked softmax
- [x] 通过 pybind11 把棋规核心搬到 native —— `v0/src/rules/rule_engine.cpp` + `v0/src/moves/move_generator.cpp` 完整 C++ 原生实现
- [ ] 在节点缓存 legal action indices 和 child state，避免重复调用 action_to_index/apply_move（MCTSCore Node 当前仅存 GameState + children indices）
- [ ] 预热并复用 TensorStateBatch / states_to_model_input 的工作缓冲降低 encode 成本（当前每次调用均创建新张量）

### 2025.11.09

- [x] C++ 层重写 batch_apply_moves，并补充准确性测试与性能脚本

### 2025.11.10

- [x] GPU 版本重写 apply 部分，打通 fast apply kernel 的 CUDA 支持 —— `v0/src/game/fast_apply_moves_cuda.cu` 已实现，tests/v0/cuda/test_fast_apply_moves_cuda.py 通过

### 2025.11.13

- [ ] 训练时可以random训练或者是先训练后面那部分，再逐渐往前训练直至收敛（课程学习，尚未实现）
- [ ] v0 CUDA benchmarking: MCTSCore 的 `project_policy_logits_fast` 要求 legal masks 和 logits 在同一设备，需在 `ExpandBatch` 中添加设备对齐逻辑以支持 "forward on CUDA + tree on CPU" 基准测试

### 2025.11.14

- [x] 新增 `v0/data/` 目录，统一存放离线自博弈样本（JSONL + meta）。
- [x] 实现 `python v0/generate_data.py`，可指定模型 checkpoint/算力配置批量生成样本并落盘。
- [x] `v0/train.py` 支持 `--data_files` 离线训练模式，或用 `--save_self_play_dir` 将在线自博弈数据自动写入 `v0/data/self_play`。
- [ ] 后续：给数据生成/训练脚本补详细 README 片段（`v0/v0_pipeline_notes.md` 有基本 JSONL 格式，但缺少字段级文档和典型命令）
- [x] 排查：动作空间是否相同？ —— `tests/v0/check_v0_policy_index_alignment.py` 已验证 legacy 与 v0 动作索引对齐
- [x] ~~排查：verify_v0速度优化12倍，benchmark速度为何只优化3倍？~~ （早期调查已过时，README.md 显示端到端 ~4.5x 加速，符合预期）

### 2025.12.27

- [x] 将 v0 C++ 扩展的 MSVC 编译支持扩展到 Linux（CMake + Ninja）
- [x] 清理测试代码，整理测试目录结构
- [x] 梳理并重建 v1 目录（从早期试验版演进为当前主线）
- [x] 合并 v0/TODO.md 到根目录
- [x] 整理说明文件（MD文档）
- [ ] 进行大规模训练（当前 toy_train.sh 规模：40 iterations × 6400 games/iter）

### 2026.01.13

- [x] batching in eval phase —— `v0/src/mcts/eval_batcher.cpp` 实现 EvalBatcher：异步批量推理、超时合并、统计直方图

### 2026.02.07

#### 无吃子判和规则 (方案 C)

- [x] Python `GameState` 添加 `moves_since_capture` 和 `NO_CAPTURE_DRAW_LIMIT = 36`
- [x] `move_generator.apply_move` 通过棋子数量变化检测吃子，自动重置/递增计数器
- [x] `evaluate.play_single_game` 使用 `is_game_over()` 统一判终
- [x] C++ `GameState` 添加 `moves_since_capture` 和 `kNoCaptureDrawLimit`
- [x] `state_io.py` 序列化/反序列化兼容
- [x] C++ tensor-level batch apply (CPU/CUDA) 添加 `moves_since_capture` 追踪——已修改 `BatchInputs`/`BatchOutputs`、所有函数签名、CUDA kernel 参数和 Python 绑定

#### GPU 性能优化调查与改进

观察到两个异常现象需要调查和优化：

**现象 1：自博弈时功耗低但显存占用高**
- 现状：GPU 功耗 185W/500W，显存占用 72500MiB/97871MiB，GPU-Util 99%
- 疑点：功耗低表明计算密集度不足，大量显存可能仅用于存放节点缓存
- [ ] 分析 MCTS 节点缓存的显存占用分布（`tools/analyze_node_cache.py` 已就绪，需运行分析）
- [ ] 排查是否存在冗余的张量副本或未释放的中间结果
- [ ] 优化节点缓存策略：考虑 LRU 淘汰、压缩存储或 CPU 卸载
- [ ] 提升 GPU 计算密集度：增加批量推理大小或优化 kernel 并行度

**现象 2：训练时 GPU 利用率低**
- 现状：GPU 功耗 132W/500W，显存占用 1201MiB，GPU-Util 仅 11%
- 疑点：训练效率极低，GPU 大部分时间在等待数据或 CPU 计算
- [x] 使用 profiler 定位训练瓶颈（数据加载 / CPU 预处理 / GPU 计算）→ 策略损失为逐样本 Python 循环
- [x] 排查 DataLoader 配置：num_workers、prefetch_factor、pin_memory
- [ ] 优化 batch size 和梯度累积策略
- [ ] 检查是否存在不必要的 CPU-GPU 同步操作
- [x] **策略损失批量化**（见下方「策略损失批量化重构方案」）

---

### 策略损失批量化重构方案（高效训练必要保证）—— ✅ 已完成

**目标**：去掉训练阶段「对 batch 内每个样本 for 循环 + 逐样本调用 `get_move_probabilities`」的瓶颈，改为在 GPU 上对整批做一次 combined logits 构建 + 掩码 log_softmax + 批量化 KL 损失，从而提升 GPU 利用率与训练吞吐。

<details>
<summary>详细重构步骤（已全部完成，点击展开）</summary>

**现状简述**：
- 策略头输出：`log_p1`(B,36)、`log_p2`(B,36)、`log_pmc`(B,36)，与 v0 的 placement / movement-from / selection 语义一致。
- 当前：对每个样本 i 调用 `get_move_probabilities(log_p1[i], log_p2[i], log_pmc[i], legal_moves[i], ...)`，在 Python 里按合法着法循环求 score 再 `stack`，再对整批做逐样本 KL。batch=4096 即 4096 次 Python 循环，GPU 大量空闲。
- v0 已有：`project_policy_logits_fast`（C++）按 (placement_dim, movement_dim, selection_dim, auxiliary_dim) 从 log_p1/log_p2/log_pmc 构建 combined logits (B, total_dim)，并在 legal_mask 上做 masked softmax；`v0/python/move_encoder.py` 有 `action_to_index(move, board_size, spec)` 将 legacy 的 move 字典映射到 flat 索引，与 v0 动作空间一致。

**不变量**：
- 规则与阶段语义以 `README.md`、`rule_description.md` 为准。
- 动作索引与 v0 的 `ActionEncodingSpec`（placement_dim=36, movement_dim=144, selection_dim=36, auxiliary_dim=4, total_dim=220）保持一致，便于日后与 v0 推理/自博弈共用或对拍。
- 策略损失仍为「仅在合法动作上的 KL」，draw 样本的 policy 权重用 `policy_draw_weight` 控制。Value loss 使用 WDL cross-entropy，和棋样本无需额外降权。

**重构步骤（细化）**

1. **动作空间与索引约定**
   - [x] 在 `src` 中引入与 v0 一致的「单动作 → flat 索引」约定（`src/policy_batch.py`：`action_to_index`、TOTAL_DIM 等），保证 placement / movement(dir-major) / selection / process_removal 与 v0 C++ 一致。
   - [x] 在文档或注释中写明：训练用的 flat 索引与 v0 自博弈/推理的 `legal_mask`、`project_policy_logits_fast` 使用同一套 spec，避免两套编码。

2. **数据侧：legal_mask + target_dense**
   - [x] 定义「每样本」产出：`legal_mask_i` (1, total_dim) bool，`target_dense_i` (1, total_dim) float；实现 `legal_mask_and_target_dense()`。
   - [x] 集成到 `ChessDataset`：`use_batched_policy=True` 时返回 `(state_tensor, legal_mask, target_dense, value, soft_value)`；保留旧路径，collate 支持两种形状（`mcts_collate_fn` / `mcts_collate_batched_policy`）。

3. **模型侧：batch combined logits**
   - [x] Python 实现 `build_combined_logits(log_p1, log_p2, log_pmc, board_size)`，逻辑与 v0 C++ 一致（placement / movement / selection / auxiliary）。

4. **损失侧：masked log_softmax + 批量化 KL**
   - [x] 实现 `masked_log_softmax(logits, mask, dim)`、`batched_policy_loss(...)`，按样本 policy_draw_weight 加权平均；value 损失与总损失形式不变。

5. **训练入口与开关**
   - [x] `train_network(..., use_batched_policy=True)`，True 时走批量化路径，False 时保留逐样本 `get_move_probabilities` 循环。
   - [x] 单测：`tests/test_batched_policy_loss.py`（action_to_index、build_combined_logits、masked_log_softmax、batched_policy_loss 梯度、与 legacy 的 log-prob 一致性及损失量级）。

6. **验证与收尾**
   - [x] 用现有 `tests/` 或小规模训练跑 1 个 iteration，确认 loss 曲线与旧实现同量级、无 NaN（已跑 smoke：batched_policy=True，Avg Policy Loss 有限）。
   - [x] 在 AGENTS.md 中注明：训练策略损失已批量化（`src/policy_batch.py`），与 v0 动作编码一致；若修改动作空间或 spec，需同步更新此处与 v0。
   - [x] 更新本 TODO：将「策略损失批量化」项勾选完成，并保留上述步骤为历史记录。

</details>

- [ ] 前端展示胜率时没说是黑棋胜率还是白棋胜率
