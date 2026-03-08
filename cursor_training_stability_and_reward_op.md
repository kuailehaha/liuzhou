# Training stability and reward optimization
_Exported on 3/6/2026 at 12:05:00 GMT+8 from Cursor (2.6.11)_

---

**User**

/2023533024/users/zhangmq/liuzhou/logs/big_train_v1_20260223_173954.log
/2023533024/users/zhangmq/liuzhou/logs/big_train_v1_20260228_055847.log

查看这两个训练文件，以及必要的训练相关文件。我想要确定为什么目前的训练不够稳定，以及有哪些方式可以让模型随着自博弈数据的增加棋力稳定上升？

具体来说，这个项目：

传统棋类“六洲棋”AI，涵盖规则建模、MCTS 强化学习训练与人机对战系统实现。【项目高强度更新中】
• 实现规则引擎与合法动作生成器，原子化拆解对弈流程。
• 搭建 AlphaZero 风格自博弈训练流水线。
– 搭建 ChessNet，以三个策略头（给出下一步动作概率）和一个价值头（评估当前局面价值）作为网络输出。自
博弈生成对局。使用当前模型驱动的 MCTS 产生 (s, π, z) 序列，模拟结果累积为训练样本。
– 使用 C++/CUDA 重写自博弈引擎，通过适当的算子设计和通信优化，极大提升了训练效率（∼ 448.6× 于原
python 功能性框架）。在 4 卡 H20 上训练 24 小时后，当前模型可在严苛条件下以 99.8% 胜率（2000 局）击败
RandomAgent，显著超越基线（∼ 1.3% 胜率）。
– 当前模型暂时无法在小算力下（1024simu）击败精通级别的人类。下一步的方向在于训练强人工智能：优化奖
励机制、引入剪枝策略、保证训练稳定可持续。


我在这里，着重希望可以调整：训练稳定可持续，以及奖励机制的优化，从而实现比人类更强的人工智能。

请你为我查看代码和训练日志，并最终形成可行的优化方案。


---

**Cursor**

I'll start by examining the training logs and key training-related source files to understand the current state and identify stability issues.

Let me start by reading both training logs and key source files in parallel.

Now let me read more of both logs to understand the full trajectory, and also examine the core training code.

Let me read a few more critical files to complete the picture before formulating the plan.

Now let me check a few more things - the policy_batch loss, the MCTS value handling, and the self-play runner's soft value computation.

Now I have a thorough understanding of both training runs, the full pipeline, and all relevant code. Here is my complete analysis and optimization plan.

---

## 诊断分析

### 1. 两次训练运行的关键指标对比

**Run 1 (20260223, 从零开始, 80轮)**

| 指标 | 趋势 | 问题 |
|---|---|---|
| vs_random 胜率 | iter1: 27.7% → iter5: 92.1% → peak 99.8% (iter34附近) → 后期波动在 80%~99% | 非单调上升，后期有回落 |
| vs_previous gating | 80轮中仅少数 accept (iter4,5,9,16,17,19...)，大量 reject (0-0-1000 全和或 0-500-500) | **gating 机制几乎失效** |
| 自博弈和棋率 | iter1: 98.77% → iter2: 51.15% → 后期 60%~90% 波动 | 后期模型趋于保守，和棋率反复升高 |
| 训练 loss | 3.53 → 2.92 → 在2.86~3.18间波动 | **Loss 不收敛，在plateau振荡** |
| value targets nonzero | iter1: 0.11% → iter2: 47.88% → 后期波动在 15%~50% | 有效训练信号不稳定 |

**Run 2 (20260228, 从 iter_034 热启动, lr=1e-4 + cosine)**

| 指标 | 趋势 | 问题 |
|---|---|---|
| vs_random 胜率 | iter1: 63.5% → iter7: 88.3% (上升中) | 从 iter_034 开始反而 vs_random 下降了（原本99.8%） |
| vs_previous gating | 7轮全部 reject（始终输给 iter_034） | **模型在持续退化** |
| 自博弈和棋率 | 33%~70% 波动 | 波动极大 |
| loss | 1.83 → 1.62 → 1.97 → 2.43 → 1.78 → 2.06 | **Loss 不降反升** |
| 自博弈黑白不均 | iter4: black_win=236013 vs white_win=21070 | 11:1的先后手失衡 |

### 2. 核心问题根因

我从代码和日志中识别出以下 **五个根本性问题**：

#### 问题 A：每轮训练只用当轮数据，没有经验回放（Replay Buffer）

```58:63:v1/python/train_bridge.py
    if strategy_raw in {"dp", "data_parallel"}:
        strategy = "data_parallel"
    elif strategy_raw in {"none", "single"}:
        strategy = "none"
    elif strategy_raw == "ddp":
        strategy = "ddp"
```

每轮 selfplay 产生 ~5000 万 positions，训练 4 epochs 后就丢弃。下一轮用新模型生成全新数据。这意味着：
- 模型对旧经验没有记忆，容易**灾难性遗忘**
- 若某轮自博弈产生了高和棋率的数据（有效信号不足），该轮训练几乎无效
- 训练 target 在 z=0（和棋）时几乎没有 value 梯度，导致 value head 学到"预测 0 最安全"

#### 问题 B：optimizer 每轮重新初始化

```217:217:v1/python/train_bridge.py
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
```

每轮 `train_network_from_tensors` 调用都创建新的 Adam optimizer，这意味着：
- Adam 的一阶/二阶动量估计每轮都从零开始
- 前几个 batch 的更新步长不稳定（Adam warmup 效应）
- 有效学习率在每轮开头极大（因为二阶矩估计小），容易过冲

#### 问题 C：和棋奖励信号太弱，soft value 混合策略不够

当 `soft_label_alpha=1.0` 时：
```271:275:v1/python/train_bridge.py
                mixed_values = torch.clamp(
                    (1.0 - alpha) * batch_values + alpha * batch_soft,
                    min=-1.0,
                    max=1.0,
                )
```

虽然 alpha=1.0 完全使用 soft target（material delta），但 soft value 的计算方式有问题：

- `soft_value = tan(2.0 * material_delta / 18.0)`
- 大多数和棋结束时 material delta 接近 0，所以 soft value 也接近 0
- 这相当于告诉 value head："这些和棋局面的价值是 0"——这没有提供任何关于**谁在和棋中占优势**的信息
- 大量 z=0 且 soft_value 也 ≈ 0 的样本稀释了训练信号

#### 问题 D：Gating 机制在确定性评估下失效

从日志可以看到大量 `0-0-1000`（全和）和 `500-0-500`（恰好各赢一半）的 vs_previous 结果。这是因为：
- 评估时 temperature 过低导致两个相似模型走出完全相同的棋路
- 当两个模型差异不大时，确定性评估完全无法区分强弱
- Gating reject 积累导致 best_model 长期不更新，但 latest_model 在 diverge

#### 问题 E：训练没有 LR warmup，且缺少 intra-iteration scheduler

LR 只在 iteration 间通过 cosine 下降，但每轮内部 4 epochs（数千个 batch）用固定 LR 训练，结合问题 B（optimizer 重置），导致训练极不稳定。

---

## 优化方案

以下方案按**优先级从高到低**排列，每个改动都有明确的代码修改位置和预期效果。

### 方案 1（最高优先级）：引入滑动窗口经验回放

**原理：** AlphaZero 原论文和后续工作（KataGo, Leela）都使用最近 N 轮的数据混合训练，而非仅用当轮数据。这是训练稳定性的基石。

**改动范围：**
- `scripts/big_train_v1.sh`：增加 `REPLAY_WINDOW` 参数（建议 4~8 轮），在 train stage 传入最近 N 轮的 selfplay 文件列表
- `v1/train.py`：接收多个 selfplay input 文件，合并加载
- `v1/python/train_bridge.py`：从合并后的样本中随机采样训练

**建议实现：**
```python
# v1/train.py 中，stage=train 时
selfplay_files = args.self_play_input.split(",")  # 支持逗号分隔多文件
merged_samples = merge_selfplay_batches(selfplay_files)
```

**预期效果：**
- 消除灾难性遗忘
- 平滑每轮数据质量波动（某轮高和棋率 ≈ 被其他有效轮稀释）
- 预计 vs_random 波动幅度降低 50%+

### 方案 2（高优先级）：持久化 optimizer state

**原理：** Adam 的动量状态是训练连续性的关键。重置 optimizer 等于每轮从头学习。

**改动范围：**
- `v1/python/train_bridge.py`：增加 `optimizer_state_path` 参数
- checkpoint 中同时保存/加载 optimizer state dict
- `v1/train.py`：在 stage=train 时传入/输出 optimizer 状态

**建议实现：**

```python
# train_bridge.py
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
if optimizer_state_path and os.path.exists(optimizer_state_path):
    opt_state = torch.load(optimizer_state_path, map_location=device_obj)
    optimizer.load_state_dict(opt_state)
    # 更新 LR
    for pg in optimizer.param_groups:
        pg['lr'] = lr

# 训练结束后保存
torch.save(optimizer.state_dict(), optimizer_state_path)
```

**预期效果：**
- 消除每轮开头的 loss spike
- 训练 loss 曲线更平滑
- 配合 cosine LR 真正实现连续退火

### 方案 3（高优先级）：改进 value target，引入 anti-draw 惩罚

**原理：** 当前和棋 target = 0（hard）或 ≈ 0（soft material delta），导致 value head 学到"预测 0 是安全的"。需要让模型有动机去避免和棋。

**改动方式 A — anti-draw 常量偏移：**

```python
# self_play_gpu_runner.py 中 finalize 时
ANTI_DRAW_PENALTY = -0.05  # 和棋对双方都是小负值
result_from_current_player = where(is_draw, ANTI_DRAW_PENALTY, z)
```

**改动方式 B — 基于 material 的更精细 soft value：**

当前 `soft_value_k=2.0` 太保守。对于 `material_delta / 18.0` 这个归一化值，`tan(2 * x)` 在 x 接近 0 时几乎是线性的，没有放大效果。建议：

```python
# 提高 k 到 4.0~6.0，让 material 优势在 value 上有更强的体现
SOFT_VALUE_K = 5.0
```

**改动方式 C — value target 中加入游戏长度惩罚：**

```python
# 更长的和棋 → 更负的 value（鼓励快速结束）
game_length_penalty = -0.1 * (game_length / MAX_GAME_PLIES)
mixed_value = where(is_draw, game_length_penalty + soft_material, z)
```

**预期效果：**
- Value head 学到"和棋不如赢"，主动追求胜利
- 自博弈的和棋率应该稳步下降
- 有效训练样本比例从 15%~50% 提升到 50%~80%

### 方案 4（中优先级）：增加 LR warmup + intra-iteration scheduler

**改动范围：**
- `v1/python/train_bridge.py`：在每轮训练内增加 linear warmup + cosine decay

**建议实现：**

```python
total_steps = synced_batch_count * epochs
warmup_steps = min(100, total_steps // 10)

def get_lr_scale(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)
# 在每个 step 后调用 scheduler.step()
```

**预期效果：**
- 消除每轮/每 epoch 开头的 loss 跳跃
- 配合 optimizer state 持久化效果更佳

### 方案 5（中优先级）：修复 gating 评估的区分度

**问题：** 确定性评估 (T=0) 导致两个相似模型走相同棋路 → 全和。

**改动范围：**
- `scripts/big_train_v1.sh` 中 `EVAL_VS_PREVIOUS_*` 参数

**建议调整：**

```bash
EVAL_VS_PREVIOUS_TEMPERATURE=0.3          # 从 1.0 降低，但不为 0
EVAL_VS_PREVIOUS_SAMPLE_MOVES=1
EVAL_VS_PREVIOUS_V1_OPENING_RANDOM_MOVES=4  # 增加开局随机性
EVAL_GAMES_VS_PREVIOUS=2000                  # 增加对局数提高统计显著性
```

另外，gating 条件可以放宽（不要求严格 wins > losses，改为置信区间检验）：

```python
# 使用 binomial test 或简单的 win_rate > 0.48 条件
accept = (wins / max(1, wins + losses)) > 0.48  # 而非 wins > losses
```

**预期效果：**
- vs_previous 不再出现 `0-0-1000` 全和
- Best model 更新频率从 ~20% 提升到 ~40%

### 方案 6（中低优先级）：policy draw weight 渐进下调

**原理：** 当前 `policy_draw_weight=1.0`，和棋样本对 policy loss 的贡献和胜负样本相同。但和棋的 policy target 本质上是"双方都没犯错"的均匀搜索策略，信噪比低。

**建议：** 随训练进度将 `policy_draw_weight` 从 1.0 下调到 0.3~0.5：

```bash
POLICY_DRAW_WEIGHT_INIT=1.0
POLICY_DRAW_WEIGHT_FINAL=0.3
```

**预期效果：**
- Policy head 更专注于从胜负局学习决定性着法
- 减少和棋样本对 policy 的噪声干扰

### 方案 7（低优先级但长期重要）：增加网络容量

当前 10 blocks * 128 channels 对于复杂棋类可能偏小。AlphaZero 原版用 20 blocks * 256 channels。

**建议分阶段增加：**
- Phase 1: 15 blocks * 128 channels
- Phase 2: 15 blocks * 192 channels（如果 Phase 1 见效）

这需要更多 GPU 显存和训练时间，但对于超越人类水平是必要的。

---

## 实施优先级总结

| 优先级 | 方案 | 修改文件 | 预期影响 |
|---|---|---|---|
| P0 | 滑动窗口经验回放 | `big_train_v1.sh`, `v1/train.py`, `train_bridge.py` | 消除遗忘，稳定训练 |
| P0 | 持久化 optimizer state | `train_bridge.py`, `v1/train.py` | 消除每轮 loss spike |
| P0 | Anti-draw + 增强 soft value | `self_play_gpu_runner.py`, `big_train_v1.sh` | 增加有效训练信号 |
| P1 | Intra-iteration LR warmup | `train_bridge.py` | 平滑 loss 曲线 |
| P1 | 修复 gating 评估 | `big_train_v1.sh`, gating 逻辑 | 合理更新 best model |
| P2 | Policy draw weight schedule | `big_train_v1.sh`, `train_bridge.py` | 提升 policy 质量 |
| P3 | 增加网络容量 | `neural_network.py` | 提高棋力上限 |

---

## 不变量

任何改动都需要保证：
- 动作编码与 v0 一致（`ActionEncodingSpec`，total_dim=220）
- 阶段流转逻辑不变
- 规则引擎行为不变
- 现有 checkpoint 兼容性（网络结构不变时可热加载）

## 验证方式

1. P0 改动完成后，运行一次 10~15 轮的 aggressive 训练
2. 观察指标：
   - loss 是否单调下降（不再振荡）
   - vs_random 是否单调上升
   - 自博弈和棋率是否稳步下降
   - gating accept 比例是否 > 30%
3. 如果以上指标改善，继续运行到 80 轮观察长期趋势

需要我开始实现这些优化方案吗？建议从 P0 三个改动一起推进，因为它们是相互协同的。

---

**User**

我的训练入口在scripts/big_train_v1.sh

针对你的调整，我有如下想法：
我十分支持引入滑动窗口经验回放，但是由于数据量太大，回放会导致读入时处理时间过长或者是容易崩溃。
我十分支持持久化optimizer state
我十分支持改进value target 引入 anti draw惩罚，并且那个alpha=1.0我知道应该调整，但是我担心引入多个级别不一致的量导致训练混乱。同时我需要考虑参数的设置级别。
我支持增加LR warm up和intra-iteration scheduler
我也支持policy draw weight 渐进下调
Gating 机制在确定性评估下失效这一点，现在应该是通过调整vs_previous的temperature实现了
我还支持增加网络容量。
因为eval_vs_previous以及vs random其实很快的（大概在四卡上2000局只需要20秒）

我还想要考虑，是否应当取消开局随机步的退火？

请你严肃评议我的想法。然后列出具体的最终版调整计划。在这之后，我们将开始实现。

---

**Cursor**

Let me re-examine the specific code and parameters you're concerned about before giving my assessment.

Now let me look at the Run 1 log data more carefully to understand the opening random schedule behavior.

Now let me check how checkpoint saving works and the `_load_self_play_payload` function to understand the replay buffer implications.

Now let me check the `_save_checkpoint` function to understand what's saved.

Now I have a thorough understanding of every detail. Let me formulate the final assessment.

---

## 严肃评议

### 1. 滑动窗口经验回放 — 支持，但需控制规模

你的担忧是对的：每轮 selfplay 产生 ~5000 万 positions，直接加载 4~8 轮的数据（2~4 亿 positions）确实会在 `torch.load` + CPU 拼接阶段耗时极长且有 OOM 风险。

**但这不意味着要放弃 replay，而是换一种实现方式。** 关键洞察是：不需要加载全部历史数据，只需要从历史轮中**随机采样一个子集**。具体做法：

- 在 `big_train_v1.sh` 中，train stage 传入最近 N 轮的 selfplay manifest 路径列表
- `_load_self_play_payload` 扩展为支持多文件输入，对每个历史文件只加载其中 1/N 的 shard（利用已有的 sharded_manifest 机制，每轮本身就有 4 个 shard 文件）
- 这样总数据量恒等于约 1 轮的体量，但混合了多轮的样本

**采样策略：** 窗口内最新轮占 50%，其余轮平分剩余 50%。这样既保证了最新数据的权重，又避免了数据量膨胀。

**结论：方案保留，实现方式调整为"多源采样加载"而非"全量拼接"。**

### 2. 持久化 optimizer state — 完全支持

没有需要修正的点。当前每轮创建新 Adam optimizer 是训练不稳定的主要原因之一。实现方式：checkpoint 中增加 `optimizer_state_dict` 字段，train_bridge 接收并加载。

**注意事项：** 当 LR 通过 cosine schedule 在 iteration 间变化时，加载 optimizer state 后需要覆盖 `param_groups` 中的 `lr`，否则会用上一轮的 LR。当前代码中已经是每轮传入 `CUR_LR`，只要在加载 state dict 后更新即可。

### 3. 改进 value target / anti-draw — 支持，但你关于"参数层级一致性"的担忧需要正面回应

你担心引入多个量级不一致的量导致训练混乱，这个担忧是合理的。当前体系中存在三个"层"：

1. **hard target z**: {-1, 0, +1}
2. **soft target**: tan(k * material_delta/18)，范围 [-1, 1] 但实际值远小于 1
3. **混合**: (1-α) * z + α * soft

问题在于当 α=1.0 时，你完全抛弃了 z，只用 soft target。而 soft target 在和棋局面（material delta ≈ 0）时值接近 0，在胜负局面时也因为 k=2.0 太小而信号偏弱。

**我的建议是简化为单一 target 语义，而不是叠加更多层：**

```
value_target = z + anti_draw_penalty * is_draw
```

其中 `anti_draw_penalty = -0.05`（一个小负值），`is_draw = (z == 0)`。

然后将 soft_label_alpha 设回 0.0（或一个很小的值如 0.1），让 hard target 主导训练信号，soft target 仅作微调。这样：
- value target 的量级始终在 [-1, 1]，语义清晰
- anti-draw 是一个纯增量偏移，不涉及新的尺度变换
- soft_value_k 提高到 5.0 只影响 soft channel，不改变主信号

**结论：anti-draw penalty 直接作用在 hard target 上，soft_label_alpha 从 1.0 降回 0.1。** 这比在 soft channel 上做文章更干净。

### 4. LR warmup + intra-iteration scheduler — 支持

每轮 ~5000 万 positions / 16384 batch_size / 4 GPU = ~760 steps * 4 epochs = ~3040 steps。在这个步数规模下，100 步的 linear warmup + cosine decay 是合理的。

配合 optimizer state 持久化，warmup 主要在第一轮起作用（后续轮因为 optimizer 有历史动量，warmup 的影响会减小，这正是期望的行为）。

### 5. Policy draw weight 渐进下调 — 支持，但建议保守

当前 `policy_draw_weight=1.0` 意味着和棋样本的 policy loss 权重与胜负样本相同。考虑到你的和棋率在 40%~85% 波动，和棋样本占据了大量训练比重。

**但不建议下调到 0.3 那么低。** 原因：和棋局面中的 policy target（MCTS 搜索策略）并非完全无信息——它们反映了"如何维持均势"的着法分布，对开局和中局的策略塑形有价值。

**建议：从 1.0 渐进到 0.5，不低于 0.5。**

### 6. 开局随机步退火 — 建议保留退火，但调整初始值

你问是否应该取消开局随机步的退火（即保持固定值而非从 6→0）。

**分析当前两次训练：**
- Run 1: opening_random_schedule=6→0，从零训练
- Run 2: opening_random_moves=0→0（固定为 0），从 iter_034 热启动

Run 2 完全取消了开局随机，但训练表现更差（7 轮全部 reject）。这并非仅因为开局随机的问题（Run 2 的退化有多个原因），但完全取消开局随机会导致：

1. **自博弈多样性严重不足**：强模型在确定性开局下会走出极其相似的棋路，导致数据集中在极少数局面附近
2. **value head 只见到"正确开局"的局面**，无法评估"犯错后的局面"——而对手（特别是人类）会走出非最优开局

**我的建议：不退火，固定 opening_random_moves=4。**

理由：
- 退火到 0 会在训练后期丧失开局多样性
- 但 6 在后期偏多（6 步随机 ≈ 整个落子阶段的约 1/3 都是随机的）
- 固定 4 步随机是一个好的平衡点：保证开局多样性的同时不会过度扰动
- 这也符合 KataGo 的做法（training 中始终保持少量开局随机）

### 7. 增加网络容量 — 支持但延后

这是正确的方向，但应该在前面的稳定性改动验证通过后再做。原因：
- 增大网络会改变训练动力学，不利于判断其他改动是否生效
- 需要验证当前 10x128 在稳定训练下能达到什么棋力上限
- 增大网络后 selfplay 吞吐会下降，需要重新平衡 concurrent_games 等参数

**建议：本轮暂不调整网络容量，在稳定性改动验证后作为下一阶段独立推进。**

### 8. Gating 机制 — 你说已调整 vs_previous temperature

看 Run 2 的日志，确实 `eval_vs_previous temperature=1.0 sample_moves=1`，不再有 `0-0-1000` 全和的情况（Run 2 的 vs_previous 都有明确胜负分布）。所以这点已经解决。

但 gating 条件 `wins > losses` 过于严格——在 1000 局中，模型只需要稍微弱一点就永远 reject。**建议放宽到 wins >= losses * 0.9 或使用更宽松的统计检验。** 不过这是小调整，可以在主要改动落地后再微调。

---

## 最终调整计划

以下是经评议后的确认改动清单，按实施顺序排列：

### Phase 1：训练稳定性核心改动（一起实施和验证）

| # | 改动 | 文件 | 具体内容 |
|---|---|---|---|
| 1 | 持久化 optimizer state | `v1/python/train_bridge.py`, `v1/train.py`, `scripts/big_train_v1.sh` | checkpoint 中保存/加载 optimizer_state_dict；train_bridge 接收 optimizer_state_path；big_train_v1.sh 传递路径 |
| 2 | Intra-iteration LR warmup + cosine | `v1/python/train_bridge.py` | 每轮内 100 步 linear warmup + cosine decay；配合外层 cosine LR |
| 3 | 滑动窗口经验回放（采样式） | `v1/train.py`, `scripts/big_train_v1.sh` | 支持 `--self_play_input` 接受逗号分隔多文件；每个历史文件只加载部分 shard；最新轮 50% + 历史轮 50% |
| 4 | Anti-draw penalty on hard target | `v1/python/self_play_gpu_runner.py` 或 `v1/python/train_bridge.py` | 和棋 value target 从 0 改为 -0.05；soft_label_alpha 默认 0.1 |
| 5 | soft_value_k 提高 | `scripts/big_train_v1.sh` | `SOFT_VALUE_K=5.0`（从 2.0 提高） |
| 6 | 开局随机固定 4 步 | `scripts/big_train_v1.sh` | `SELF_PLAY_OPENING_RANDOM_MOVES=4`, `SELF_PLAY_OPENING_RANDOM_MOVES_FINAL=4`（不退火） |
| 7 | Policy draw weight 渐进 | `scripts/big_train_v1.sh` | `POLICY_DRAW_WEIGHT` 从 1.0→0.5 cosine 退火 |

### Phase 2：验证后的后续改动

| # | 改动 | 条件 |
|---|---|---|
| 8 | 增加网络容量 (15x128 或 15x192) | Phase 1 验证通过后 |
| 9 | 放宽 gating 条件 | 如果 accept 率仍低于 30% |

### 参数总表（Phase 1 最终值）

| 参数 | 当前值 | 调整后 |
|---|---|---|
| `SOFT_LABEL_ALPHA` | 1.0 | 0.1 |
| `SOFT_VALUE_K` | 2.0 | 5.0 |
| `ANTI_DRAW_PENALTY` | (无) | -0.05 |
| `SELF_PLAY_OPENING_RANDOM_MOVES` | 0 (Run2) / 6→0 (Run1) | 4 (固定) |
| `SELF_PLAY_OPENING_RANDOM_MOVES_FINAL` | 0 | 4 |
| `POLICY_DRAW_WEIGHT` | 1.0 (固定) | 1.0→0.5 (cosine) |
| `LR` | 1e-4 | 1e-4 (不变) |
| `LR_COSINE_FINAL_SCALE` | 0.5 | 0.5 (不变) |
| Optimizer state | 每轮重置 | 持久化 |
| Intra-iter scheduler | 无 | 100 步 warmup + cosine |
| Replay window | 1 轮 | 4 轮 (采样式) |

### 不变量清单

- 动作编码 `TOTAL_DIM=220` 不变
- 网络结构 10x128 不变（Phase 1）
- 阶段流转、规则引擎不变
- 评估参数不变（vs_random T=0, vs_previous T=1.0 sample_moves=1）
- Gating 条件 `wins > losses` 暂不变

### 验证方式

Phase 1 完成后运行 15 轮 aggressive 训练，对比观察：
1. Loss 曲线：是否单调下降（不再振荡）
2. vs_random 胜率：是否 >= 90% 且稳定
3. 自博弈和棋率：是否稳步下降到 40% 以下
4. Gating accept 比例：是否 > 30%
5. vs_previous：候选模型是否能逐轮缩小与 best 的差距

---

准备好开始实施了。要我按上面 1~7 的顺序逐项推进吗？

---

**User**

这也符合 KataGo 的做法（training 中始终保持少量开局随机）

这个会不会导致最终的模型的前4步的走法比较随机？

因为eval_vs_previous以及vs random其实很快的（大概在四卡上2000局只需要20秒）
我想在eval时多加几个模型，这样的话可以保证测试稳定性。



---

**Cursor**

Good questions. Let me address both precisely.

## 开局随机步是否导致模型前 4 步随机？

**不会。** 这里需要区分两个完全不同的东西：

1. **自博弈时的 `opening_random_moves=4`**：在生成训练数据时，前 4 步跳过 MCTS 搜索，直接从合法动作中均匀随机选一个。但关键是——**这 4 步的 policy target 也是均匀分布**，而模型从这些样本中学到的是"开局前几步有很多合法选择"。

2. **评估和部署时的行为**：评估时 `opening_random_moves=0`，模型完全按自己的 policy head 输出 + MCTS 搜索来走棋。前 4 步的走法取决于模型学到了什么。

**实际效果是：** 因为训练数据中前 4 步是随机的，模型在这些步的 policy target 基本是均匀分布。但模型在这些位置的 **value head** 会学到"从各种不同开局出发的局面价值"。这意味着：

- 模型部署时，前 4 步仍然会通过 MCTS 搜索选择 value head 评估最高的着法——这不是随机的
- 模型见过了各种非最优开局后的局面，所以即使对手走了"怪招"，value head 也能正确评估

**但你的担忧有一个合理内核：** 前 4 步的 policy head 确实会被训练成比较"平"的分布（因为 target 就是平的）。如果你不想让 policy head 在开局阶段失去辨别力，有一个折中方案：

**对开局随机步的样本，设 policy_weight=0（只训练 value head，不训练 policy head）。**

这样 value head 从多样化开局中学习评估能力，而 policy head 只从 MCTS 搜索步中学习着法选择。这是最干净的做法。

实现方式：在 `self_play_gpu_runner.py` 中给 `force_uniform_random` 的步骤标记一个 flag，让 `train_bridge.py` 对这些步的 policy loss 权重置 0。

**不过这会增加实现复杂度。** 一个更简单的替代方案是：保持 `opening_random_moves=4`，但把 `temperature_threshold` 从 10 提到 12 或 14。这样前 4 步虽然是随机的，但第 5~14 步仍然用高温 MCTS 采样（保证多样性），而 policy head 从第 5 步起就有有意义的搜索 target。这样 policy head 在开局阶段（步 5+）仍然有充分的训练信号。

**我的最终建议：保持 `opening_random_moves=4`，暂不做额外标记处理。** 理由是：
- 六洲棋的落子阶段有 36 步（每人 18 子），4 步随机只占 11%
- 开局的 policy 精度对整体棋力的影响远小于中盘和走子阶段
- 增加标记机制的复杂度不值得，除非后续验证发现开局确实是瓶颈

---

## 评估时多加几个模型

既然评估很快（20 秒/2000 局），增加评估维度是性价比极高的改进。具体建议：

在 `big_train_v1.sh` 的 eval stage 中，除了 `vs_random` 和 `vs_previous`，增加：

| 评估对手 | 用途 | 局数 |
|---|---|---|
| `vs_random` | 健康度探针（已有） | 2000 |
| `vs_best_previous` | Gating 判定（已有） | 2000 |
| `vs_iter_001` | 绝对基线锚点 | 1000 |
| `vs_self` | 自对弈和棋率监控 | 1000 |

**`vs_iter_001`（固定基线锚点）：** 始终用训练起始模型作为对手。这给出一个绝对的进步度量——如果连最初的模型都打不过，说明训练严重退化。

**`vs_self`（自对弈）：** 用当前候选模型自己打自己，监控和棋率。这个指标不受对手强度影响，纯粹反映模型的和棋倾向。如果 self-play 的和棋率在评估时也很高（>80%），说明模型确实在"学习和棋"。

**实现方式：** 在 `big_train_v1.sh` 的 eval 阶段增加 2 次 `eval_checkpoint.py` 调用，每次约 10 秒（1000 局），总额外开销 ~20 秒/轮，完全可以接受。

不过我建议把这两个额外评估的结果输出到**独立的 JSON 文件**，不影响 gating 逻辑。Gating 仍然只看 `vs_best_previous`。

---

## 更新后的最终计划

将上述两点整合到 Phase 1 计划中：

| # | 改动 | 说明 |
|---|---|---|
| 1 | 持久化 optimizer state | 不变 |
| 2 | Intra-iteration LR warmup + cosine | 不变 |
| 3 | 滑动窗口经验回放（采样式） | 不变 |
| 4 | Anti-draw penalty + soft_label_alpha=0.1 | 不变 |
| 5 | soft_value_k=5.0 | 不变 |
| 6 | opening_random_moves=4（固定不退火） | 不变 |
| 7 | Policy draw weight 1.0→0.5 cosine | 不变 |
| **8** | **增加 eval 维度：vs_iter_001 + vs_self** | **新增** |

确认这个计划后，我开始按顺序实施。是否可以开始？

---

**User**

是的，开始。

---

**Cursor**

Let me start by reading all the files that need modification to have the full current state.

Now let me read the remaining sections I need to modify in `v1/train.py` — the argument parser and the `train_pipeline_v1` function signature.

Good. Now I have complete context. Let me implement all changes. I'll work on `train_bridge.py` first (items 1, 2, and the anti-draw penalty in the value mixing), then `v1/train.py` (items 1, 3, 4 — plumbing), and finally `big_train_v1.sh` (items 3, 5, 6, 7, 8).

**Item 1: Persist optimizer state + Item 2: LR warmup + cosine scheduler + Item 4 (partial): anti-draw in train_bridge**

Now update the function signature to accept new parameters:

Now replace the optimizer creation + training loop section to add optimizer persistence, LR scheduler, and anti-draw penalty:

Now update the return section to save optimizer state and include new metrics:

Now update `v1/train.py` — plumbing new parameters through the staged pipeline. First, the function signature and argument parser:

Good, now I need to update the function signature and the train stage call site. Let me update the function signature:

Now update the train stage call site to pass the new parameters and add replay buffer support:

Now add the new CLI arguments to the parser:

Now update the `__main__` call to pass the new args:

Now update `big_train_v1.sh` — this is the big one with all remaining items (3, 5, 6, 7, 8):

This is a large file. I'll rewrite it entirely with all changes applied — parameter defaults, curriculum function, replay window, train stage plumbing, new eval stages, and logging:

Now update the `compute_curriculum_values` to include policy_draw_weight:

Now update the logging section to include new parameters:

Now update the iteration loop header to read the new curriculum field and set up replay + optimizer paths:

Now update the train stage commands (both DDP and non-DDP) to pass the new parameters:

Now add the extra eval stages (vs_baseline and vs_self) after the existing eval block. I need to insert them after the `fi` that closes the gating block and before the gating decision logic:

Now I need to make sure the `scripts/train_entry.py` forwards the new arguments. Let me check it:

I need to add the new arguments to `train_entry.py`:

Now update the `_run_v1` call to pass the new args:

Now let me check all files for lint errors:

Let me verify the shell script is syntactically valid:

Good. Now let me update all the remaining todos and verify the complete set of changes is consistent:

Let me check the `instruction.sh` to see if it also needs updating:

This is an instrumentation script, not part of the training pipeline. No changes needed.

All 8 planned changes are implemented. Let me do a final verification by checking that the key parameter flow is correct end-to-end.

All parameters propagate correctly through the full chain: `big_train_v1.sh` → `train_entry.py` → `v1/train.py` → `train_bridge.py`.

Let me verify there are no git diff issues with inconsistent changes:

The `instruction.sh` diff is from a pre-existing modification (shown in the git status at the start). Let me verify it wasn't changed by my edits:

That's a pre-existing change (mode change only). My edits only touched the 4 intended files.

---

## 实施完成总结

以下是所有修改的文件和对应改动：

### 修改的文件

| 文件 | 改动 |
|---|---|
| `v1/python/train_bridge.py` | +54 行：optimizer 持久化、LR warmup+cosine scheduler、anti-draw penalty |
| `v1/train.py` | +67 行：新参数传递、replay buffer 采样加载逻辑 |
| `scripts/train_entry.py` | +20 行：新 CLI 参数转发 |
| `scripts/big_train_v1.sh` | +144/-5 行：新默认参数、curriculum 扩展、replay 窗口、额外 eval 阶段 |

### 具体改动对照

**1. Optimizer state 持久化** — `train_bridge.py`
- 新增 `optimizer_state_path` 参数
- 训练前加载 optimizer state dict（如存在），训练后保存
- 加载后用当前 LR 覆盖 param_groups

**2. Intra-iteration LR warmup + cosine** — `train_bridge.py`
- 新增 `warmup_steps` 参数（默认 100）
- 每轮内：前 100 步 linear warmup，之后 cosine decay 到 0.5x
- 使用 `LambdaLR` scheduler，每个 gradient step 后调用

**3. 滑动窗口经验回放** — `v1/train.py` + `big_train_v1.sh`
- `big_train_v1.sh` 自动收集最近 `REPLAY_WINDOW` 轮的 selfplay 文件列表
- `v1/train.py` 接收 `--self_play_replay_inputs`，对每个历史文件随机采样子集
- 总训练数据量 ≈ 当前轮的 2x（当前轮 100% + 历史轮按比例采样）

**4. Anti-draw penalty** — `train_bridge.py`
- 新增 `anti_draw_penalty` 参数（默认 0.0，big_train 设为 -0.05）
- 在 batch 级别：检测 `|value_target| < 1e-8` 的和棋样本，将其 value 替换为 penalty 值
- 作用于 mixed value 计算之前，与 soft_label_alpha 协同

**5. 参数默认值调整** — `big_train_v1.sh`
- `SOFT_VALUE_K`: 2.0 → **5.0**
- `SOFT_LABEL_ALPHA`: 1.0 → **0.1**
- `ANTI_DRAW_PENALTY`: (新增) **-0.05**
- `SELF_PLAY_OPENING_RANDOM_MOVES`: 0 → **4**（固定不退火）
- `POLICY_DRAW_WEIGHT`: 1.0 → **1.0→0.5** cosine 退火

**6. 额外评估维度** — `big_train_v1.sh`
- `eval_vs_baseline`：与 iter_001（或指定锚点）对战，1000 局
- `eval_vs_self`：自对弈监控和棋率，1000 局

### 待测功能清单

1. Optimizer state 在 DDP 模式下正确保存/加载（仅 rank 0 保存）
2. Replay buffer 在 sharded_manifest 格式下正确加载和采样
3. Anti-draw penalty 正确应用于 value target（和棋样本 value 从 0 变为 -0.05）
4. LR scheduler 在 epoch 边界处连续（不重置 global_step）
5. Policy draw weight curriculum 在 iteration 间正确变化
6. 额外 eval 阶段不影响 gating 逻辑
7. `vs_baseline` 在 iter_001 尚未生成时优雅跳过

### 建议 commit

`[feature/train]: stabilize training with replay buffer, optimizer persistence, anti-draw penalty, LR warmup, and multi-dimensional eval`

---

**User**

此外，我还想确认一下导入数据过程中，即在处理selfplay的数据到train的过程中。我们能不能做到并行？我们有很多的内存和很多的cpu，但是现在往往四卡只有4个cpu是满占用的。我非常需要一个可以并行导入数据的流程。
同时，这个在运行时，容器经常容错。我想增加捕获机制或者是避免报错的机制。（因为我现在甚至不知道它的具体出错是什么原因？我猜是读入读出带宽过大导致的容错？）


---

**Cursor**

Let me first understand the current data loading path and identify the bottlenecks.

Now let me look at the `_concat_self_play_batches_cpu` function and the selfplay save/shard flow to understand the full picture:

Now I see the full picture. Each load takes ~70 seconds for ~13M positions per rank (52M global / 4 DDP ranks). And shards are loaded **sequentially** within the manifest. Let me also check where the crash might happen:

Let me look at any recent terminal errors:

Now I have the full picture. Key observations:

1. **Load time: ~70 sec** per iteration just for `torch.load` of 1 shard per DDP rank (~13M positions)
2. **DDP mode: `ddp_pre_sharded=True`** — each rank loads only 1 shard (good), but the load is single-threaded
3. **Replay adds more sequential loads** on top of the primary
4. **The concat (`torch.cat`) of multiple batches** doubles peak CPU memory transiently
5. **Train time: ~107 sec** — so load is ~40% of the total cycle, which is huge

Let me now implement parallel loading and robust error handling.

Good, `ThreadPoolExecutor` is already imported. Now I'll implement two changes:

1. **Parallel shard loading** in `_load_self_play_payload` — load shards via ThreadPool
2. **Parallel replay loading** in the train stage
3. **Robust error handling with retries** throughout

Now let me update the replay loading section to use parallel loading:

Now let me replace the sequential replay loading with a parallel approach:

Now let me add robustness to the overall train stage — wrapping the entire train-stage data load + train call with error handling, and adding a GC + memory logging mechanism. Let me first look at the full train stage ending:

Now let me add explicit memory management after data load and GC calls, plus better timing/logging. Also, let me add a `try/except` wrapper around the data load section for more informative error messages:

Now let me also look at the `big_train_v1.sh` to see how the train stage invocation handles errors — this is likely where the container crash occurs:

I see — currently there's no error handling at all for the train/selfplay stages in the shell script. If the Python process crashes (e.g. OOM), the script just `exit 1`s. Let me check the top of the shell script for `set -e`:

`set -e` is on — any non-zero exit instantly kills the entire script. Now let me also check the selfplay shard saving to understand how many shards exist per file:

Now let me also check how many shards are saved per selfplay iteration (to know the parallelism we can get):

Good. Now I have a complete picture. In DDP mode with 4 GPUs, shards are already pre-split per rank (so each rank loads 1 shard). The bottleneck is:

1. **Shard loading** is single-threaded `torch.load` (~70 sec per rank for 13M positions = ~7GB)
2. **Replay loading** is sequential file-by-file, now parallelized
3. **No error handling** at any level — container crashes are invisible

The `torch.load` itself is essentially I/O bound (reading a huge file from disk). Using threads for multiple shards already helps. But for the single-shard DDP case, I should add more robust approaches. Let me now also add robust error handling in the shell script:

Now let me add a retry wrapper and crash diagnostics to the shell script. I'll add a `run_with_retry` function at the top, and use it to wrap the critical selfplay and train stages:

Now let me find the right spot to insert the retry function — right after the helper functions at the top:

I'll add a `run_with_retry` function after `build_local_cuda_list`, before the EVAL_WORKERS block:

Now I need to wrap the selfplay and train invocations with `run_with_retry`. Let me update both:

Now wrap the train invocations:

Now let me verify the Python code for linting issues and verify the shell script syntax:

Both clean. Let me also verify the key code paths are consistent — check that the `del payload` is properly placed and the gc import is available:

Good. Let me also quickly check what the `_load_single_shard` function does with the retry:

Now let me also verify the full shard parallel loading section is correct:

Now verify the replay loading section is correct:

Let me also verify the retry wrapper in the shell script looks correct:

Let me verify the selfplay invocation:

And the train invocations:

All three invocations (selfplay, train-ddp, train-non-ddp) are wrapped. Now let me do a final end-to-end check by running the complete shell syntax check and also verifying the Python import chain works:

All imports work correctly. Let me do the final diff check:

All changes are complete. Here is a summary of what was done:

---

## Changes Summary

### Problem Analysis

From the training logs: each iteration's `selfplay_payload_load_sec` is **~70 seconds** (loading ~13M positions / ~7GB per DDP rank), which is **~40% of the total iteration cycle** (~70s load + ~107s train). With replay buffer added, load time would scale linearly. Additionally, crashes are silent — `set -e` in the shell script kills everything with no diagnostics.

### 1. Parallel Shard Loading (`v1/train.py`)

When a sharded manifest contains multiple shards for one rank, they are now loaded **concurrently** via `ThreadPoolExecutor` (up to 8 workers). Previously this was a sequential loop. Each shard is loaded by `_load_single_shard()` which also explicitly `del payload` after extracting tensors to reduce peak memory.

### 2. `torch.load` Retry with Backoff (`v1/train.py`)

New `_torch_load_with_retry()` function wraps every `torch.load` call with:
- Up to **3 attempts** (configurable via `_LOAD_RETRY_ATTEMPTS`)
- Exponential backoff: 2s, 4s, 6s between attempts
- `gc.collect()` between retries to reclaim memory
- Detailed warning logs on each retry attempt

This directly addresses transient IO failures that may cause container crashes.

### 3. Parallel Replay Loading (`v1/train.py`)

Replay files (from sliding window) are now loaded **concurrently** via `ThreadPoolExecutor` instead of sequentially. Each replay file is loaded and sampled in its own thread. Failed files are logged but don't crash the run.

### 4. Memory Diagnostics (`v1/train.py`)

After data loading + `gc.collect()`, the log now prints `rss_mb=` (process RSS via `resource.getrusage`). This helps identify if memory pressure is the crash root cause.

### 5. Stage Retry Wrapper (`big_train_v1.sh`)

New `run_with_retry` function wraps selfplay and train stages:
- Up to **3 total attempts** (configurable via `STAGE_MAX_RETRIES=2` = 1 original + 2 retries)
- On failure, logs: exit code, timestamp, `free -h`, `dmesg -T | tail -20`
- Before retry: `sync` + `drop_caches` to reclaim OS page cache
- Backoff: 5s, 10s between retries

### 6. Explicit `del payload` Everywhere

All `_load_*` paths now `del payload` after extracting tensors, which avoids holding 2x the data in memory during the conversion step.

---

### Quantitative Impact Estimate

| Metric | Before | After |
|---|---|---|
| Shard load (multi-shard) | Sequential | Parallel (up to 8x) |
| Replay load (N files) | Sequential | Parallel (up to 8x) |
| Transient IO crash | Fatal | Auto-retry (3 attempts) |
| Container crash visibility | Exit code only | `dmesg`, `free`, exit code, timestamp |
| Peak memory during load | 2x payload size | ~1x (explicit `del`) |

---

Suggested git commit: `[feature/train]: parallel data loading with retry and crash diagnostics`

### Pending Test Checklist
1. Run a single iteration with replay to confirm parallel loading works and timing improves
2. Verify `rss_mb` appears in the timing log line
3. Test `run_with_retry` by temporarily forcing a failure (e.g. corrupt input path)
4. Confirm `STAGE_MAX_RETRIES=0` disables retry (single attempt only)

---

**User**

我想我们应当streaming load训练。（摒弃我刚才的多线程load想法，按照最优的设计来说，应当流式读入的）我们应当怎么修改呢？列一个具体的修改计划，然后我们基于修改计划来进行完善。

---

**Cursor**

Let me first thoroughly understand the current data flow before proposing a streaming design.

Now let me also check the TensorSelfPlayBatch structure to understand the tensor shapes:

Now I have a complete picture. Let me switch to plan mode since this is a significant architectural change with real trade-offs to discuss.

Understood. Let me lay out the concrete plan directly here.

---

## Current Bottleneck Analysis

From the log data, each training iteration cycle looks like:

```
selfplay_payload_load_sec ≈ 70s   (torch.load 13M positions, ~7GB per DDP rank)
cpu_shard_sec ≈ 0s
h2d_copy_sec ≈ 5.5s               (full dataset CPU→GPU)
train_time ≈ 107s                  (4 epochs over 13M positions)
```

The fundamental problem: **全量加载** — the entire dataset is `torch.load`'d into CPU RAM as one blob, then `torch.cat`'d for replay, then **copied in its entirety to GPU**, then trained. This causes:
1. ~70s dead time for disk→CPU (single-threaded `torch.load` on one giant file)
2. ~5.5s dead time for CPU→GPU bulk copy
3. Peak CPU memory = 2x dataset (original + GPU copy coexist briefly)
4. With replay, peak = (1 + N) x dataset

## Streaming Design Plan

The core idea: **never hold the entire dataset in memory at once**. Instead, use PyTorch `DataLoader` with `num_workers` to stream chunks from disk in parallel with GPU training.

### Architecture Overview

```
[Disk: .pt shard files]
      ↓ (num_workers=8..16, each loads a chunk)
[DataLoader workers: torch.load chunk → CPU tensors]
      ↓ (prefetch_factor=2, pin_memory=True)
[Training loop: batch → GPU, forward, backward]
```

### Concrete Changes (4 files)

#### 1. New: `v1/python/streaming_dataset.py` — Streaming Dataset

A `torch.utils.data.IterableDataset` that:
- Takes a list of `.pt` shard/payload file paths + optional sample budget per file
- Each worker gets a subset of files (via `worker_info`)
- Loads one file at a time, yields individual samples (or small chunks)
- Implements reservoir sampling for replay files (to respect `per_file_budget`)
- Handles retry on IO errors (reusing `_torch_load_with_retry` logic)

#### 2. Modify: `v1/python/train_bridge.py` — Accept DataLoader instead of TensorSelfPlayBatch

Current `train_network_from_tensors` takes a monolithic `TensorSelfPlayBatch`. Change to:

- New function `train_network_streaming(model, dataloader, ...)` that iterates over the DataLoader
- Each batch from DataLoader arrives as CPU tensors → immediate `.to(device, non_blocking=True)` per batch
- The per-batch training loop (forward/backward/step) stays identical
- Keep `train_network_from_tensors` as a fallback path (for small datasets or compatibility)
- DDP synchronization: `synced_batch_count` determined at start via `dist.all_reduce` on estimated total

#### 3. Modify: `v1/train.py` — Build DataLoader from file list, pass to train_bridge

Current: `_load_self_play_payload` → monolithic `samples` → `train_network_from_tensors(samples)`.

Change to:
- Collect file paths (primary + replay) without loading
- Build `StreamingSelfPlayDataset(file_paths, budgets)` 
- Build `DataLoader(dataset, batch_size=batch_size, num_workers=N, pin_memory=True, prefetch_factor=2)`
- Call `train_network_streaming(model, dataloader, ...)`
- Keep the monolithic path as fallback when `STREAMING_LOAD=0`

#### 4. Modify: `scripts/big_train_v1.sh` — Pass file list instead of loading

- Add `STREAMING_LOAD="${STREAMING_LOAD:-1}"` and `STREAMING_WORKERS="${STREAMING_WORKERS:-8}"` parameters
- Pass `--streaming_load --streaming_workers N` to train_entry.py
- No need to change the selfplay stage (it still saves the same shard format)

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Dataset type | `IterableDataset` | Shard files are variable-size; `Map`-style needs upfront total count |
| Worker file assignment | Round-robin over shards | Simple, balanced |
| Replay sampling | Reservoir per shard | Bounded memory per worker |
| DDP sharding | Each rank gets distinct shard subset | Same as current manifest logic |
| Epoch boundary | Re-shuffle file order per epoch, within-file shuffle via randperm | Deterministic enough |
| Batch count sync | Pre-compute from file metadata (shard_sizes) | Avoid mid-epoch DDP hangs |
| Fallback | `STREAMING_LOAD=0` → old monolithic path | Safety net |

### Expected Performance Impact

| Metric | Before | After (Streaming) |
|---|---|---|
| Load wait before training | ~70s | ~0s (first batch ready in <2s) |
| H2D copy time | 5.5s (bulk) | Amortized into training (~0s visible) |
| Peak CPU memory | ~14GB per rank | ~2-3GB per rank (workers + prefetch) |
| Replay loading | Sequential/parallel 70s×N | Amortized, overlapped |
| Total iteration time | ~183s | ~115s (train-bound) |

**~37% reduction** in wall-clock per iteration, plus massively reduced memory pressure (fixing container crashes).

### Verification

1. Compare `total_seen` samples and `avg_loss` between streaming and monolithic paths (should be identical for same data, same seed)
2. Timing: `selfplay_payload_load_sec` should drop to near zero
3. Memory: `rss_mb` should drop significantly
4. DDP: all ranks must see same number of batches (via pre-computed `synced_batch_count`)

---

This is the plan. Shall I proceed with implementation?

---

**User**

好的，请执行修改。

---

**Cursor**

Let me start by reading all the files I need to modify to ensure I have the latest state.

Now let me read the `train_pipeline_v1` signature and the shard save logic to understand the metadata available in manifest files:

Now let me read the shard manifest structure:

Now I have the complete picture. The manifest stores `shard_sizes` — I can use this to pre-compute `total_samples` without loading data. Let me implement all the changes.

Now add the `train_network_streaming` function to `train_bridge.py`. I'll append it after the existing `train_network_from_tensors`:

Now append the streaming function at the end of the file:

Now I need to modify `v1/train.py` — the train stage section. I need to add the streaming path alongside the existing monolithic path. Let me first read the full train stage:

Now I have the full picture. I need to:
1. Add `streaming_load` and `streaming_workers` parameters to `train_pipeline_v1`
2. Add a streaming branch in the `stage_norm == "train"` section
3. Update the imports

Let me add the parameters to the function signature:

Now add the `streaming_load` and `streaming_workers` parameters to `train_pipeline_v1`:

Now I need to replace the entire train stage data loading + training section with branching logic. Let me re-read the exact block to replace:

Now I'll replace the entire train stage block with branching logic:

Now update `scripts/train_entry.py`:

Now update the `_run_v1` call to pass the new args:

Now update `big_train_v1.sh`. I need to add the parameters and pass them to both DDP and non-DDP train commands:

Now find the DDP train command and add the new args:

Now do the same for the non-DDP train command:

Now add logging for the new params. Let me find the logging section:

Now verify everything:

Now let me verify the imports work end-to-end:

Now let me verify the parameter flow through `train_entry.py`:

Let me do a final diff to confirm the scope of changes:

All changes are complete. Here is the summary:

---

## Changes Summary

### New File: `v1/python/streaming_dataset.py`

Core streaming infrastructure:

- **`ShardSpec`** dataclass — describes one shard file (path, num_samples, sample_budget)
- **`resolve_shard_specs()`** — reads only the manifest metadata (not tensor data) from primary + replay files. Returns file paths and estimated sample counts. DDP-aware: each rank gets a disjoint subset of shards. This replaces the ~70s `torch.load` of the entire dataset with a <1s metadata scan.
- **`StreamingSelfPlayDataset(IterableDataset)`** — each DataLoader worker gets a disjoint subset of shards, loads them one at a time, applies reservoir sampling for replay budgets, shuffles within each shard, and yields individual `(state, mask, policy, value, soft_value)` tuples. After each shard is consumed, tensors are freed + `gc.collect()`.
- **`build_streaming_dataloader()`** — wraps the dataset with `DataLoader(num_workers=8, pin_memory=True, prefetch_factor=2)`.
- Retry logic built in via `_torch_load_retry`.

### Modified: `v1/python/train_bridge.py`

Added `train_network_streaming()` — a new entry point that:

- Accepts a `DataLoader` instead of a monolithic `TensorSelfPlayBatch`
- Each batch arrives as CPU pinned tensors → immediate `.to(device, non_blocking=True)` per batch
- The training loop (forward/backward/optimizer step/LR scheduling/AMP/DDP gradient sync/non-finite guards) is identical to `train_network_from_tensors`
- `total_samples` is passed in from the metadata-only scan to compute `est_batches_per_epoch` and LR schedule
- DDP batch count synchronization via `dist.all_reduce(MIN)`
- The existing `train_network_from_tensors` is fully preserved as fallback

### Modified: `v1/train.py`

The `stage_norm == "train"` section now branches:

- **`streaming_load=True`** (default): calls `resolve_shard_specs` → `build_streaming_dataloader` → `train_network_streaming`. No monolithic load ever happens. Only manifest metadata is read upfront.
- **`streaming_load=False`**: falls through to the existing monolithic path (preserved unchanged).

### Modified: `scripts/train_entry.py`

Added `--streaming_load` (default 1) and `--streaming_workers` (default 8) arguments, propagated to `train_pipeline_v1`.

### Modified: `scripts/big_train_v1.sh`

- `STREAMING_LOAD="${STREAMING_LOAD:-1}"` and `STREAMING_WORKERS="${STREAMING_WORKERS:-8}"` at top
- Both DDP and non-DDP train commands pass `--streaming_load` and `--streaming_workers`
- New params logged at startup

### Data Flow Comparison

**Before (monolithic):**
```
torch.load(7GB) → CPU RAM [70s] → torch.cat replay [+70s×N] → .to(GPU) [5.5s] → train [107s]
Peak memory: 2× dataset. Total wall: ~250s with replay.
```

**After (streaming):**
```
resolve metadata [<1s] → DataLoader(workers=8) streams batches in parallel with train
Per-batch: worker reads chunk → pin_memory → .to(GPU, non_blocking) → forward/backward
Peak memory: ~workers × shard_size. Total wall: ≈ train time only (~107s).
```

### Fallback

Set `STREAMING_LOAD=0` in the environment to revert to the old monolithic path.

---

Suggested git commit: `[feature/train]: streaming DataLoader for selfplay data — eliminates 70s blocking load per iteration`

### Pending Test Checklist

1. Run 1 iteration with `STREAMING_LOAD=1` and verify training completes, loss is reasonable
2. Compare loss from `STREAMING_LOAD=1` vs `STREAMING_LOAD=0` on same data (should be similar)
3. Verify `first_batch_sec` timing in log — should be <2s instead of 75s
4. Verify memory (`rss_mb` in fallback path or `htop`) — should be significantly lower
5. Verify DDP: all ranks see same `est_batches_per_epoch` via the `all_reduce(MIN)` sync
6. Test with replay: `REPLAY_WINDOW=4` should stream replay files without memory explosion

