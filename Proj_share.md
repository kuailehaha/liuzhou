# 六洲棋 AI 算法分享：从 AlphaZero 闭环到当前 V1 的真实实现

> 面向推荐算法同学的 30 分钟分享材料
> 文档基于当前仓库 `main`（`e3e9fdd`）的代码、测试和设计记录梳理，重点讨论算法与训练动力学，不展开 GPU、DDP、数据落盘等工程细节。

## 0. 先给结论

这套系统可以概括为：

> 用规则生成合法候选，用策略价值网络给候选先验和长期价值，用搜索生成比网络更强的策略标签，再用自博弈数据训练同一个网络。

它属于 AlphaZero 风格的广义策略迭代，但当前 V1 主线与标准 AlphaZero 有一个非常重要的区别：

> 默认 `SPARSE_PLY=1` 时，V1 不是逐层扩展并回传的完整蒙特卡洛树搜索，而是“评估根节点全部合法子节点，再在固定叶值上用 PUCT 分配访问次数”的根节点搜索。

因此，本项目当前最准确的算法名称是：

> **神经网络引导的 Root-PUCT 自博弈策略迭代**。

这不是文字游戏。它直接影响我们如何理解 simulations：标准 AlphaZero 增加 simulations 通常能搜索得更深、修正更多网络误差；当前根搜索增加 simulations 主要是在同一批固定子节点价值上更精细地重新分配访问次数，并不会自动看得更远。

训练不稳定也不是单一学习率问题，而是多个反馈环叠加：

1. 模型决定下一轮看见什么状态；
2. 模型又参与生成自己的监督标签；
3. 当前搜索对 value 误差的纠偏能力有限；
4. 六洲棋有大量原子子阶段和同一玩家连续动作，价值符号、探索退火和状态表示都更容易发生错配；
5. 高平局使硬价值标签趋近于零，而长平局还会按“局面数”占据更多训练权重；
6. 当前 11 通道输入没有覆盖完整规则状态，存在状态混叠；
7. 训练脚本中有若干“声明了但主线没有真正启用”的稳定化参数。

如果只记住三个改进优先级：

1. **先保证问题是 Markov 的**：把影响合法转移和终局距离的计数器编码进网络输入；
2. **再保证搜索真的是有效的 policy improvement**：优先验证更深搜索是否正确，而不是继续堆 root simulations；
3. **把胜负目标与局面优势辅助目标解耦**：主 value 学 W/D/L，子力差等 dense signal 放辅助头，不要默认混成同一个标量目标。

---

## 1. 30 分钟分享建议

| 时间 | 内容 | 希望听众带走什么 |
|---|---|---|
| 0–3 min | 六洲棋为什么难 | 它不是普通的“落一子就换手”游戏 |
| 3–7 min | 算法主干 | 规则 → 网络 → 搜索 → 自博弈 → 训练 → 再搜索 |
| 7–12 min | 状态、动作和网络 | 11 通道、220 动作、三策略头、101 桶价值头 |
| 12–18 min | 当前 Root-PUCT | PUCT 如何生成策略标签，以及它为什么不是完整 MCTS |
| 18–22 min | 训练目标 | `(s, π, z)`、软价值、视角变换、损失函数 |
| 22–25 min | 与 AlphaGo 系列对比 | 我们继承了什么，又为六洲棋改了什么 |
| 25–29 min | 不稳定和平局 | 非平稳闭环、状态混叠、目标冲突、长平局过采样 |
| 29–30 min | 下一步 | 先修语义和测量，再做逐项 A/B |

给推荐算法同学的类比：

| 棋类系统 | 推荐系统类比 |
|---|---|
| 规则引擎与 legal mask | 候选生成和业务约束 |
| 策略网络 `p(a|s)` | 排序模型给候选打分 |
| 价值网络 `v(s)` | 长期价值/LTV 估计 |
| Root-PUCT | 带长期反馈和探索项的在线重排 |
| 访问分布 `π` | 比原模型更强的 teacher distribution |
| 自博弈 | 模型自己产生曝光日志，典型的 endogenous data |
| replay / 历史对手池 | 时间窗日志和历史策略流量 |
| 对局胜负 | 稀疏、延迟的最终业务指标 |

最大的共同难点也是一样的：**模型改变数据分布，数据再改变模型**。这不是固定数据集上的普通监督学习。

---

## 2. 算法主干：一个自我改进闭环

先忽略所有实现细节，一轮训练只有六步：

```mermaid
flowchart LR
    A[当前网络 f_theta] --> B[规则约束下的搜索]
    B --> C[自博弈动作与轨迹]
    C --> D[样本 s, pi, z]
    D --> E[更新策略头与价值头]
    E --> F[候选网络 f_theta_new]
    F --> G[固定条件评估]
    G --> A
```

### 2.1 先用一条训练样本说清楚：`(s, π, z)` 是什么

先不看公式。这个系统实际上在学两件事：

1. **面对当前局面，下一步应该怎么走？**
2. **面对当前局面，最后大概会赢、平还是输？**

自博弈中的每一步，最终会形成一条训练样本：

\[
(s,\pi,z).
\]

三个量先用人话解释：

- **`s` —— state，当前局面**：网络作出判断时看到的完整状态。它不只是 6×6 棋盘上的棋子，还包括当前行动方、当前规则阶段、标记等信息。输入网络时，它们被编码成多个 6×6 特征平面，类似一张有多个通道的小图像。
- **`π` —— pi，搜索后的下法分布**：搜索完成后，把每个合法动作的访问次数归一化得到的概率分布。某个动作的 `π` 越大，表示搜索综合了网络判断、探索和规则后，越愿意选它。它是教网络“应该怎么下”的策略标签，不是网络原始输出，也不是实际落子的 one-hot 标签。
- **`z` —— 终局结果**：这盘棋真正结束后才知道的结果。从样本 `s` 中当前行动方的视角看，通常记为胜 `+1`、平 `0`、负 `-1`。它不是网络在当时的估计，而是对局结束后沿整条轨迹回填的监督信号。六洲棋有同一玩家连续执行多个原子动作的阶段，因此只有行动方真正改变时才改变价值视角，不能每走一个原子步骤就机械取反。

> 一条 `(s,π,z)` 可以读成：“当时看到局面 `s`，搜索认为应当按 `π` 分配下法概率，而这盘棋最后的结果是 `z`。”

实际数据还会保存 `legal mask` 和软价值等字段，但 `(s,π,z)` 是理解主算法时最重要的抽象。

#### 网络看到 `s` 后输出什么

对状态 `s`，网络输出两类信息：

\[
f_\theta(s) = \bigl(p_\theta(\cdot\mid s), v_\theta(s)\bigr).
\]

- **`pθ(a|s)` —— 策略先验**：网络根据状态 `s` 给每个合法动作的初始概率。可以理解为网络的“第一直觉”：**它来自网络输出，尚未加入 PUCT 的探索奖励，也尚未根据访问次数重新分配**。在实际计算中，非法动作会先被 `legal mask` 屏蔽，再在合法动作上形成先验分布。
- **`vθ(s)` —— 长期价值估计**：从状态 `s` 的当前行动方视角出发，如果之后沿当前网络所诱导的策略继续下去（实际自博弈中还会经过搜索改进），最终结果的期望估计。它不是“下一步的好坏”，而是对整盘棋最终结果的压缩判断：越接近 `+1` 越看好当前行动方，越接近 `-1` 越看衰，接近 `0` 则偏向均势或和棋。

这里最需要分清的是 `p`、`π` 和 `z`：

| 符号 | 什么时候得到 | 通俗理解 | 在训练中做什么 |
|---|---|---|---|
| `p` | 网络刚看完 `s` | 未加搜索探索的“第一直觉” | 策略头当前的预测 |
| `π` | 对 `s` 搜索完之后 | 搜索修正后的“教师答案” | 策略头要拟合的目标 |
| `z` | 整盘棋结束之后 | 事后得知的真实胜平负 | 价值头要拟合的目标 |

整条关系可以读成：

```text
当前局面 s
    → 网络给出先验策略 p 和长期价值 v
    → 搜索结合 p、v、规则后得到改进策略 π
    → 根据 π 选择动作，继续对局
    → 对局结束后得到终局结果 z
    → 用 (s, π, z) 训练网络：p 学 π，v 学 z
```

因此，网络策略 `p` 不是最终的下法标签。搜索使用策略先验 `p` 和价值评估 `v` 做一次策略改进（policy improvement），得到访问分布 `π`。训练再让网络拟合 `π` 和终局结果 `z`：

\[
\theta_{t+1}
\leftarrow
\arg\min_\theta
\mathbb{E}_{(s,\pi,z)\sim D(\theta_t)}
\left[
L_\text{policy}(p_\theta,\pi)
+L_\text{value}(v_\theta,z)
\right].
\]

公式用人话说就是：**用当前模型 `θt` 自己下棋生成一批 `(s,π,z)`，然后调整新模型，让它输出的 `p` 更接近搜索给出的 `π`，输出的 `v` 更接近最终结果 `z`**。

其中 `D(θt)` 表示“由当前模型 `θt` 产生的自博弈数据分布”。它强调这不是一个固定数据集：模型一变，下出来的棋会变，下一轮看到的训练数据也会跟着变。

### 2.2 为什么搜索和网络要互相教

只用网络：推理快，但一个前向很难精确处理战术后果。

只用搜索：每次都从零搜索，缺少局面先验，计算量巨大。

二者结合：

- 网络把搜索预算集中到更有希望的动作；
- 搜索把规则后果和局部 lookahead 融入策略标签；
- 网络将搜索结果“蒸馏”回一次前向；
- 下一轮更强的网络又生成更强的搜索标签。

理想状态下，这是一个正反馈；value 偏差大、搜索弱或数据覆盖差时，它也会变成自我确认的负反馈。

### 2.3 PPT 可直接使用：当前 V1 端到端流程伪代码

先说明版本口径：示例中的 `v0/train.py` 和 `src/train.py` 在当前检出版本中已经不存在。当前训练主线是：

```text
scripts/big_train_v1.sh
    → scripts/train_entry.py
        → v1/train.py::train_pipeline_v1()
```

下面这张调用树可以直接放在 PPT。它刻意保留算法节点和关键函数名，省略 DDP 通信、文件重试和 CUDA Graph 等工程细节。

```text
V1 训练迭代开始
入口: scripts/big_train_v1.sh
  │
  ├─ 初始化
  │   ├─ latest_model = 初始模型或上一轮 candidate
  │   ├─ best_model   = 当前通过评估的最好模型
  │   └─ optimizer_state = 上一轮 Adam 状态（若启用连续性）
  │
  ├─ for iteration = 1 ... T
  │   │
  │   ├─ [1. 自博弈阶段]
  │   │   v1/train.py → v1/python/self_play_worker.py
  │   │   │
  │   │   ├─ 每张 GPU 启动一个 self-play worker
  │   │   └─ run_self_play_worker()
  │   │       └─ self_play_v1_gpu(latest_model)
  │   │           │
  │   │           ├─ 初始化一批并行对局
  │   │           │   states = GpuStateBatch.initial(batch_games)
  │   │           │
  │   │           └─ while 仍有未结束对局:                 ← 每个原子 ply
  │   │               │
  │   │               ├─ search = V1RootMCTS.search_batch(states)
  │   │               │   ├─ 状态编码: (B, 完整规则状态)
  │   │               │   │              → model_input (B, 11, 6, 6)
  │   │               │   ├─ 规则生成 legal_mask (B, 220)
  │   │               │   ├─ 根网络前向:
  │   │               │   │     policy heads + value head
  │   │               │   ├─ 只保留合法动作，得到根 prior P(s,a)
  │   │               │   ├─ 枚举并生成所有一步子状态 s_a
  │   │               │   ├─ 批量评估所有子状态 value v(s_a)
  │   │               │   ├─ 若行动方切换则 value 取负，否则保持符号
  │   │               │   ├─ Root-PUCT 在固定 P_a、q_a 上分配访问次数 N_a
  │   │               │   ├─ π(a|s) ∝ N_a^(1/temperature)
  │   │               │   └─ 从 π 采样动作
  │   │               │       （opening curriculum 可覆盖为均匀随机动作）
  │   │               │
  │   │               ├─ 保存一步训练信息:
  │   │               │     (model_input, legal_mask, π, current_player)
  │   │               ├─ 批量执行动作，更新规则状态
  │   │               └─ 判断 win / loss / draw / max_plies
  │   │
  │   │           对局结束后:
  │   │           ├─ hard_value = 终局 W/D/L × 每步 player_sign
  │   │           ├─ soft_value = 终局子力差映射 × 每步 player_sign
  │   │           └─ 输出 TensorSelfPlayBatch:
  │   │               state_tensors      (N, 11, 6, 6)
  │   │               legal_masks        (N, 220)
  │   │               policy_targets     (N, 220)
  │   │               value_targets      (N,)
  │   │               soft_value_targets (N,)
  │   │
  │   ├─ [2. 训练阶段]
  │   │   v1/train.py → v1/python/train_bridge.py
  │   │   │
  │   │   ├─ 加载 latest_model 和本轮 TensorSelfPlayBatch
  │   │   ├─ streaming DataLoader / DDP 切分 batch
  │   │   └─ for epoch; for batch:
  │   │       ├─ 网络前向:
  │   │       │     log_p1, log_p2, log_pmc, value_logits = model(states)
  │   │       ├─ 三策略头 → 220 维 combined_logits
  │   │       ├─ legal-mask log-softmax
  │   │       ├─ policy_loss = KL(π_search || p_network)
  │   │       ├─ mixed_value = (1-λ)·hard_value + λ·soft_value
  │   │       ├─ mixed_value → 101 桶 two-hot target
  │   │       ├─ value_loss = bucket cross-entropy
  │   │       ├─ total_loss = policy_loss + value_loss
  │   │       └─ backward → gradient clip → Adam step
  │   │
  │   │   保存 candidate_model
  │   │
  │   ├─ [3. 评估阶段]
  │   │   scripts/eval_checkpoint.py
  │   │   ├─ candidate vs RandomAgent          ← 早期健康度
  │   │   ├─ candidate vs best_previous        ← gating
  │   │   ├─ 可选: candidate vs fixed baseline
  │   │   └─ 可选: candidate vs itself         ← 平局监控
  │   │
  │   ├─ [4. 模型推进]
  │   │   ├─ if candidate wins > losses against best_previous:
  │   │   │      best_model = candidate
  │   │   └─ latest_model = candidate
  │   │          注意: 即使 gating 未通过，latest 仍会推进
  │   │
  │   └─ 进入下一 iteration
  │
  └─ 输出 final latest checkpoint + final best checkpoint
```

如果 PPT 只能放一页，可以压缩成下面这段算法级伪代码：

```text
model ← InitializeOrLoad()
best  ← model

for iter = 1 ... T:
    trajectories ← []

    for batched self-play games:
        s ← InitialState()
        while not Terminal(s):
            legal ← Rules(s)
            P, V  ← Network(s)
            q[a]  ← PerspectiveAlign(Network(Apply(s, a)).value)
            N[a]  ← RootPUCT(P[a], q[a], legal[a])
            π[a]  ← Normalize(N[a]^(1/τ))
            trajectories.append(s, legal, π, PlayerToMove(s))
            a ← Sample(π)                 # 前期可用均匀随机探索覆盖
            s ← Apply(s, a)

    D ← Finalize(trajectories, WDL(s), MaterialMargin(s))

    candidate ← Train(
        model,
        policy_target = π,
        value_target  = Mix(WDL, MaterialMargin)
    )

    report ← Evaluate(candidate, Random, best)
    if report.wins_vs_best > report.losses_vs_best:
        best ← candidate
    model ← candidate                     # 当前 latest 始终前进
```

### 2.4 一步搜索的输入输出：PPT 上建议单独画一页

```text
完整规则状态 S_t
    │
    ├─ Rules(S_t) ─────────────────────────────→ legal_mask: (B, 220)
    │
    └─ Encode(S_t) → X_t: (B, 11, 6, 6)
                           │
                           ▼
                    ChessNet(X_t)
                      ├─ 三张 6×6 policy heatmap
                      └─ 101-bin value distribution
                           │
                           ▼
                 合法 prior P(a|S_t) + value
                           │
             枚举合法动作并批量生成一步子状态
                           │
                           ▼
                 q_a = Value(S_{t+1}^{(a)})
                           │
                           ▼
                   Root-PUCT visits N_a
                      ├─ policy target π_a
                      └─ sampled action a_t
```

这里要向听众强调：完整规则状态 `S_t` 比网络图像 `X_t` 信息更多。规则和 legal mask 使用完整状态，当前神经网络只看到 11 个图像平面；这正是后文“状态混叠”的来源。

---

## 3. 六洲棋首先不是一个普通棋类状态

### 3.1 棋盘和目标

- `6×6` 棋盘；
- 黑白各 18 枚棋子；
- 先落子，再处理标记和统一移除，之后进入走子与提吃；
- 当前代码在走子相关阶段判定某方棋子数少于 4 时失败；
- 总动作数达到 144，或连续 36 个动作无吃子，判平；
- 自博弈另有 `max_game_plies` 保护，主脚本默认 512。

正式规则解释以 [docs/rules.md](docs/rules.md) 为准；代码中的终局阈值见 `GameState.LOSE_PIECE_THRESHOLD = 4`。规则文档中“无子”措辞与代码阈值并不完全等价，分享时建议按“少于 4 枚即失去继续竞争能力”的当前实现口径描述，并另行做规则文档一致性核对。

### 3.2 七个原子阶段

六洲棋一个人类语义上的“回合”可能包含多个原子决策：

1. `PLACEMENT`：落子；
2. `MARK_SELECTION`：形成方/洲后，逐个选择要标记的对手棋；
3. `REMOVAL`：统一移除已标记棋；
4. `MOVEMENT`：移动一枚棋；
5. `CAPTURE_SELECTION`：形成方/洲后，逐个选择要提吃的棋；
6. `FORCED_REMOVAL`：满盘且无标记时的强制移除；
7. `COUNTER_REMOVAL`：无棋可动时，对手执行反移除。

把复合动作拆成原子阶段的好处是：

- 每个状态只需要预测一个清晰的合法动作集合；
- 多次标记或提吃不必组合成指数级大动作；
- 规则、动作 mask、自博弈样本可以逐步对齐。

代价是：**动作后不一定换手**。例如落子后进入自己的 `MARK_SELECTION`，同一玩家继续行动。这破坏了普通棋类实现里“每走一步 value 就取负”的默认假设。

### 3.3 正确的价值视角

仓库约定黑方视角终局结果为：

\[
z_\text{black}\in\{-1,0,+1\}.
\]

每个训练状态记录当时行动方符号：黑为 `+1`，白为 `-1`，因此：

\[
z(s_t)=\text{playerSign}(s_t)\cdot z_\text{black}.
\]

搜索从子状态价值转回父状态视角时，也只在行动方真的切换时翻转：

\[
q_\text{parent}(s')=
\begin{cases}
v(s'), & player(s')=player(s),\\
-v(s'), & player(s')\ne player(s).
\end{cases}
\]

这是六洲棋相对标准 AlphaZero 最重要的语义改造之一。测试 `test_v1_child_value_perspective_alignment` 专门覆盖了这一点。

---

## 4. 状态表示：网络看到了什么

### 4.1 当前 11 个输入平面

网络总是以当前行动方为“自己”编码：

| 通道 | 含义 |
|---:|---|
| 0 | 当前行动方棋子 |
| 1 | 对手棋子 |
| 2 | 当前行动方被标记棋 |
| 3 | 对手被标记棋 |
| 4–10 | 七个阶段 one-hot，全棋盘广播 |

因此输入形状为 `(B, 11, 6, 6)`。

这种相对视角编码有两个好处：

- 黑白共享同一套特征和网络参数；
- value 天然可以解释为“当前行动方价值”。

### 4.2 当前输入不是完整马尔可夫状态

规则状态实际上还保存：

- `pending_marks_required / remaining`；
- `pending_captures_required / remaining`；
- `forced_removals_done`；
- `move_count`；
- `moves_since_capture`。

这些量参与合法动作生成、阶段流转或平局判定，但没有进入 11 通道网络输入。

legal mask 能防止策略选择非法动作，却不能完全修复 value 的信息缺失。两个棋盘、标记和阶段完全相同的状态，可能因为剩余任务数或无吃子计数不同而具有不同的后继和终局距离。网络却会收到同一个 `s`，被要求预测不同标签。

这就是状态混叠：

\[
\phi(S_1)=\phi(S_2),\qquad V^*(S_1)\ne V^*(S_2).
\]

它对高平局游戏尤其危险：同样的盘面可能距离“36 步无吃子判平”还有 1 步或 30 步，value 完全看不见这个差别。

**结论：补全状态输入不是模型 trick，而是恢复 MDP/Markov 假设。** 这是优先级高于调学习率的算法修正。

---

## 5. 动作空间与策略网络

### 5.1 为什么是 220 维：计算固定动作 ID 空间

对，这里计算的就是**动作空间**。更准确地说，是给所有阶段可能出现的动作建立一个固定长度的 ID 字典，方便网络、MCTS 和训练数据使用同一套坐标。

`220` 不表示当前局面有 220 个合法动作。网络始终产生一个 220 维动作向量，而规则生成的 `legal_mask` 决定本局面其中哪些 ID 可以使用：

```text
固定动作向量 a[0 ... 219]

0                    35 36                              179 180        215 216  219
├──── placement 36 ────┼──────── movement 144 ────────────┼─ selection 36 ─┼─ aux 4 ─┤
```

推导过程如下。

#### 第一段：落子动作 `36`

棋盘共有：

\[
6\times6=36
\]

个位置，所以“在第几个格子落子”有 36 个固定 ID：

\[
cell(r,c)=6r+c,\qquad
action_\text{place}=cell(r,c).
\]

对应索引区间 `[0,35]`。

#### 第二段：移动动作 `144`

每个移动只允许从一个起点向上、下、左、右移动一格。因此只需要表示：

\[
36\text{ 个起点}\times4\text{ 个方向}=144.
\]

方向顺序固定为：

| `dir_idx` | 方向 | `(dr,dc)` |
|---:|---|---|
| 0 | 上 | `(-1,0)` |
| 1 | 下 | `(1,0)` |
| 2 | 左 | `(0,-1)` |
| 3 | 右 | `(0,1)` |

移动 ID 前面要跳过 36 个 placement ID：

\[
action_\text{move}
=36+cell(r_{from},c_{from})\times4+dir\_idx.
\]

对应索引区间 `[36,179]`。

这里没有再乘 36 个终点，因为终点已经由“起点 + 方向”唯一决定。144 个槽位中会包含一些永远不合法或在当前状态不合法的组合，例如从顶边向上、起点不是己方棋子、终点被占用；这些槽位由 legal mask 设为 false。

#### 第三段：选择目标动作 `36`

标记、提吃、强制移除、反移除和 `no_moves_remove` 都是在棋盘上“选择一个位置”。它们共享同一段 36 维位置 ID：

\[
action_\text{select}=180+cell(r,c).
\]

对应索引区间 `[180,215]`。

为什么不分别分配 `mark 36 + capture 36 + remove 36`？因为当前 `phase` 已经说明这次选择的语义：同一个 ID `195` 在 `MARK_SELECTION` 阶段表示标记，在 `CAPTURE_SELECTION` 阶段表示提吃。共享位置编码可以避免重复扩大动作空间。

#### 第四段：辅助动作 `4`

最后保留 4 个不带棋盘坐标的辅助槽位 `[216,219]`。当前实际只有：

```text
216 = process_removal
217, 218, 219 = 当前未使用，解码为 None，也不会被 legal mask 置为合法
```

所以总维度为：

\[
36+144+36+4=220.
\]

#### 四个具体例子

设位置 `(2,3)`，其 cell index 为：

\[
cell(2,3)=2\times6+3=15.
\]

那么：

| 动作 | 计算 | 最终 ID |
|---|---|---:|
| 在 `(2,3)` 落子 | `15` | 15 |
| 从 `(2,3)` 向下移动 | `36 + 15×4 + 1` | 97 |
| 在当前选择阶段选择 `(2,3)` | `180 + 15` | 195 |
| 处理统一移除 | 固定辅助 ID | 216 |

最重要的一句话是：

> **220 是跨所有阶段统一的输出坐标系；legal mask 才是当前状态真正的合法动作集合。**

例如处于落子阶段时，通常只有 `[0,35]` 中尚为空且满足规则的位置为 true，移动段和选择段全部被屏蔽。统一编码的价值在于每条样本都能保存固定 shape 的 `(220,)` policy target，并在 GPU 上批量训练。

movement 使用 **cell-major**：先起点 cell，再方向。这个约定必须与 Python、C++、CUDA、legal mask 和 policy target 全部一致。

### 5.2 三个 36 维策略头

共享 ResNet trunk 后，策略头输出：

- `p1(x)`：落点或移动终点；
- `p2(x)`：移动起点；
- `pmc(x)`：标记、提吃、强制移除等目标位置。

具体动作 logit 为：

\[
\ell(a)=
\begin{cases}
\ell_{p1}(x), & a=\text{place}(x),\\
\ell_{p2}(x_{from})+\ell_{p1}(x_{to}), & a=\text{move}(x_{from},x_{to}),\\
\ell_{pmc}(x), & a=\text{select/remove}(x),\\
0, & a=\text{auxiliary}.
\end{cases}
\]

然后只在 legal mask 内做 softmax。

为什么这样设计：

- 复用“某个位置适合落入/移入”的空间语义；
- 36+36 参数化比直接预测 144 个移动更省；
- 不同阶段共享棋盘空间特征。

它也有明确的表达能力代价：移动打分是 `起点分数 + 终点分数` 的可分形式，不能自由表达每个 `(from,to)` 特有的交互偏好。虽然共享 trunk 已编码全局局面，输出层仍然受到低秩约束。对 6×6 棋盘来说，直接 220 维统一 policy head 其实并不大，值得作为独立 A/B，而不应默认三头一定更优。

同一个 `pmc` 头同时承担标记、提吃、强制移除和反移除，优点是共享“选对方棋子”的知识；风险是不同阶段目标语义相反或强弱不同，只能依赖 phase plane 来消歧。

---

## 6. 网络结构与价值头

### 6.1 共享残差主干

从 CV 视角看，六洲棋状态就是一张尺寸很小、通道有明确语义的“多光谱图像”：

```text
输入 X: (B, 11, 6, 6)
          │
          ├─ 4 个局部棋盘平面
          │    self pieces / opponent pieces / self marks / opponent marks
          │
          └─ 7 个全局阶段平面
               每个 phase one-hot 被广播为整张 6×6 常数图
```

它不是 RGB 图像，而是 11 通道 binary feature map。卷积在棋盘空间共享权重，适合提取：

- 相邻上下左右关系；
- 2×2 的“方”结构；
- 横向/纵向“洲”结构；
- 空位、己方棋、对方棋和标记之间的局部组合。

当前行动方没有额外颜色通道，而是通过交换 self/opponent 通道完成相对视角编码：轮到白方时，channel 0 放白棋、channel 1 放黑棋。

两个 CV 细节值得在分享中点出来：

- 7 个 phase plane 是空间上恒定的条件信号，作用类似给整张图注入“当前任务类型”；
- stem 加第一个 residual block 已包含 3 层 3×3 卷积，理论感受野达到 7×7，已经覆盖整个 6×6 棋盘；后续 block 主要增加特征变换深度，而不是解决“看不到远处”的问题。

### 6.2 PPT 可直接使用：`ChessNet` 逐层结构与 shape

```text
X                                           (B, 11, 6, 6)
│
├─ Stem
│   Conv 3×3, padding=1, 11 → 128           (B, 128, 6, 6)
│   BatchNorm + ReLU                        (B, 128, 6, 6)
│
├─ Residual Trunk × 10
│   每个 PreActResBlock:
│      x ───────────────────────────────────────────────┐
│      └─ BN → ReLU → Conv 3×3, 128 → 128               │
│         → BN → ReLU → Conv 3×3, 128 → 128             │
│         → Add skip connection ◀───────────────────────┘
│   输出始终保持                              (B, 128, 6, 6)
│
├─ Trunk BN + ReLU                           (B, 128, 6, 6)
│
├──────────────────── Policy Head ────────────────────────────────┐
│                                                                 │
│   Conv 1×1, 128 → 64                        (B, 64, 6, 6)       │
│   BN + ReLU                                 (B, 64, 6, 6)       │
│      │                                                          │
│      ├─ GlobalPool(mean, max, std over 36 cells) → (B, 192)     │
│      ├─ Linear 192 → 64                        → (B, 64)        │
│      └─ reshape/broadcast + local feature      → (B, 64, 6, 6)  │
│                                                                 │
│   BN + ReLU                                 (B, 64, 6, 6)       │
│      ├─ Conv 1×1, 64 → 1 → flatten → log-softmax → p1  (B, 36)  │
│      ├─ Conv 1×1, 64 → 1 → flatten → log-softmax → p2  (B, 36)  │
│      └─ Conv 1×1, 64 → 1 → flatten → log-softmax → pmc (B, 36)  │
│                                                                 │
│   p1/p2/pmc + 动作编码 + legal mask
│      └─ combined policy over 220 actions                         │
│                                                                 │
└───────────────────── Value Head ────────────────────────────────┐
                                                                  │
    Conv 1×1, 128 → 64                        (B, 64, 6, 6)       │
    BN + ReLU                                 (B, 64, 6, 6)       │
    GlobalPool(mean, max, std)                (B, 192)            │
    Linear 192 → 128 + ReLU                   (B, 128)            │
    Linear 128 → 101                          (B, 101)            │
       ├─ 训练: 101-bin logits → bucket cross-entropy             │
       └─ 推理: softmax 后对 [-1,1] 桶中心求期望 → scalar value   │
```

这张网络图有四个讲解重点：

1. **全程不下采样**：棋盘只有 6×6，所有残差块保持空间分辨率，避免丢掉精确落点；
2. **卷积 trunk 共享**：策略和价值共享对“方/洲/棋子关系”的视觉表征；
3. **policy 是三张热力图**：最终不是直接输出 220 图，而是按动作语义组合三张 6×6 heatmap；
4. **value 是全局分类**：通过全局池化丢弃具体坐标，预测整个局面的 101 桶价值分布。

### 6.3 网络输入与输出总表

| 张量 | Shape | 数值/语义 | 消费者 |
|---|---|---|---|
| `state_tensors` | `(B,11,6,6)` | 0/1 多通道棋盘图像 | `ChessNet` |
| `log_p1` | `(B,36)` | 落子位置/移动终点 heatmap | 动作投影 |
| `log_p2` | `(B,36)` | 移动起点 heatmap | 动作投影 |
| `log_pmc` | `(B,36)` | 标记/提吃/移除位置 heatmap | 动作投影 |
| `value_logits` | `(B,101)` | `[-1,1]` 上的 bucket logits | value loss / 搜索 |
| `legal_mask` | `(B,220)` | 完整规则生成的合法动作 | masked softmax |
| `combined_logits` | `(B,220)` | 三头投影后的统一动作分数 | policy loss / search prior |
| `policy_target` | `(B,220)` | Root-PUCT 访问分布 `π` | policy loss |
| `value_targets` | `(B,)` | 当前行动方视角的 hard W/D/L | value target 混合 |
| `soft_value_targets` | `(B,)` | 当前行动方视角的终局子力软值 | value target 混合 |
| `mixed_value` | `(B,)` | `(1-λ)·hard + λ·soft`，训练时计算 | value loss |

一个容易讲错的点是：`legal_mask` 不是网络输出，它来自规则引擎。网络只负责给动作打分，规则负责决定哪些动作可以进入 softmax。这与推荐系统中“模型打分 + 业务候选约束”非常相似。

### 6.4 101 桶 distributional value

价值头不直接回归一个标量，而是输出 101 个等距桶的 logits，桶中心覆盖 `[-1,1]`：

\[
c_i=-1+\frac{2i}{100},\quad i=0,\ldots,100.
\]

连续 target `y` 被线性插值为相邻两个桶的 two-hot 分布。训练使用交叉熵，推理时取期望：

\[
v_\theta(s)=\sum_i \operatorname{softmax}(u_\theta(s))_i c_i.
\]

这样做的动机是让 bounded value 学习更稳定，并保留一定分布形状。需要注意：搜索最终仍只使用期望标量；“确定平局”和“胜负各半但期望为零”在搜索决策上仍会坍缩到相同的 `v=0`。

当前代码还计算 WDL auxiliary loss，但权重 `_WDL_AUX_LOSS_WEIGHT = 0.0`，因此它只作为指标，不参与梯度。

---

## 7. 当前 V1 搜索：Root-PUCT 到底做了什么

先用人话概括：**Root-PUCT 像是在每个合法动作之间反复分配一批“考察票”。它既要把票投给当前看起来价值高的动作，又要给网络觉得有希望但还没被充分考察的动作一些机会。**

前者叫**利用**（exploitation），后者叫**探索**（exploration）。PUCT 的核心就是平衡两者。

### 7.1 一次 Root-PUCT 搜索先准备什么

给定当前局面 `s`，当前 V1 默认路径先做三件事：

1. **找出候选动作**：规则引擎生成 220 维 `legal mask`，只保留当前阶段的合法动作集 `A(s)`。
2. **给每个动作一个第一印象 `P_a`**：根状态的策略网络对合法动作输出先验概率 `P(a|s)`。`P_a` 越大，表示网络越倾向于“这一步值得优先考察”。
3. **试走每个动作，得到结果估计 `q_a`**：对所有合法动作 `a` 各执行一次，得到一步子状态 `s_a`，再用价值网络估计子状态的最终结果。该价值会被转换到根状态当前行动方的视角，得到 `q_a`。

`q_a` 可以读成：

> “如果我现在选动作 `a`，试走一步后，网络从我的视角认为最终结果大约有多好？”

`q_a` 越接近 `+1` 越有利于当前行动方，越接近 `-1` 越不利，接近 `0` 则表示网络认为偏向均势或和棋。如果执行 `a` 后行动方换了，子状态价值才需要取反；同一玩家继续标记、移除时不能机械取反。

然后 Root-PUCT 并不立即只选 `q_a` 最大的动作，而是虚拟分配 `num_simulations` 张考察票。一次完整的根搜索可以写成：

```text
输入：当前状态 s

legal_actions = 规则引擎生成的合法动作
P[a] = 策略网络给动作 a 的先验概率
q[a] = 试走 a 后，子状态价值转换到当前行动方视角

对每个合法动作 a：
    N[a] = 0       # 目前分到了多少张考察票
    W[a] = 0       # 历次考察累加的价值

重复 num_simulations 次：
    Q[a] = W[a] / N[a] （未访问时记为 0）
    U[a] = c_puct * P[a] * sqrt(1 + sum(N)) / (1 + N[a])
    a_star = argmax_a (Q[a] + U[a])
    N[a_star] += 1
    W[a_star] += q[a_star]

π[a] = 将 N[a] 经温度处理后归一化
按 π 采样或选择实际动作
输出：搜索策略 π
```

### 7.2 PUCT 公式：一项负责“看好”，一项负责“还没看够”

每次虚拟访问都会选择 `PUCTScore` 最大的动作：

\[
\underbrace{a^*=\arg\max_a\left[Q_a+U_a\right]}_{\text{选择总分最大的动作}},
\]

\[
\underbrace{U_a}_{\text{探索分}}
=
\underbrace{c_\text{puct}}_{\text{探索强度}}
\underbrace{P_a}_{\text{网络先验}}
\frac{\sqrt{1+\sum_bN_b}}{1+N_a}.
\]

- **`a`**：当前考察的某个合法动作。
- **`P_a` —— 先验概率**：网络未加搜索探索时，对动作 `a` 的初始偏好。所有合法动作的 `P_a` 之和为 1；它决定“尚未充分考察时，谁更值得先看”。
- **`N_a` —— 访问次数**：到目前为止，动作 `a` 在虚拟访问中被选中了多少次。它不是真的把棋局重新运行了 `N_a` 遍，而是 PUCT 分配的计数器。
- **`∑_bN_b` —— 总访问次数**：根节点已经分配出去的考察票总数。`b` 只是用来遍历所有合法动作的符号。
- **`W_a` —— 累积价值**：每当动作 `a` 被访问一次，就把这次得到的价值加进 `W_a`。
- **`Q_a=W_a/N_a` —— 平均价值，即利用分**：已有证据认为动作 `a` 最终有多好。`Q_a` 越大，PUCT 越愿意利用这个已知好动作。未访问动作的 `Q_a` 在当前实现中记为 `0`。
- **`U_a` —— 探索分**：动作 `a` 还值不值得继续考察的奖励。先验越高、自身访问越少，`U_a` 通常越大。
- **`c_puct` —— 探索权重**：唯一直接缩放整个探索项的系数，代码中叫 `exploration_weight`，当前默认值为 `1.0`。

总分 `Q_a+U_a` 的含义是：

> **`Q_a` 回答“目前看起来有多好”，`U_a` 回答“这个动作是不是还没看够”。**

还要注意量纲：`Q_a` 是大致落在 `[-1,1]` 的结果价值；`P_a` 是概率；`U_a` 则是为了参与排序而构造的正向奖励，并不是胜率，也不要求落在 `[0,1]`。它们能够相加，是因为 `c_puct` 负责把探索奖励缩放到能和 `Q` 比较的尺度。也正因此，一旦 value 的尺度发生变化，原来合适的 `c_puct` 就未必还合适。

### 7.3 为什么这些量要变大或变小：PUCT 如何自动找平衡

先澄清一点：算法并不是人为地把 `P`、`Q` 或 `N` 都调大。它们的来源不同：

- `P` 由策略网络给出；
- `q` 由子状态价值评估给出，并决定 `Q`；
- `N` 由搜索过程自动累加；
- 我们可直接调节的主要是 `c_puct`、`num_simulations`、温度 `τ` 和探索噪声。

各个量变大的影响如下：

| 哪个量变大 | 公式中的直接影响 | 带来的行为 | 过大的风险 |
|---|---|---|---|
| `Q_a` 变大 | 直接抬高 `Q_a+U_a` | 更多利用当前被评估为有利的动作 | value 估错时，会反复强化错误动作 |
| `P_a` 变大 | 按比例抬高 `U_a` | 让网络第一直觉看好的动作更早、更多地被考察 | 策略网络过度自信时，会压缩其他动作的机会 |
| `N_a` 变大 | 分母 `1+N_a` 变大，该动作 `U_a` 下降 | 一个动作看得越多，额外探索奖励越小，机会让给其他动作 | `N_a` 本身不是需要人为追求的目标；它也会在最后决定策略标签 `π` |
| `∑_bN_b` 变大 | 分子的平方根缓慢增大 | 随搜索进行，尚未充分访问的动作又逐渐变得值得考察 | 用平方根而非线性增长，就是为了防止总探索压力增长过快 |
| `c_puct` 变大 | 同比例放大所有 `U_a` | 相对更相信先验 `P`，更愿意扩大动作覆盖 | 过大会让 `Q` 难以主导选择，搜索结果过度接近网络先验 |
| `q/Q` 的数值间距变大 | 利用项之间的差距扩大 | 更快集中到 value 评分最高的少数动作 | 即使 `c_puct` 不变，有效探索强度也会变弱 |

#### 一个数值例子：为什么价值稍低的动作也会被选中

假设现在已经分配了 15 次访问，`c_puct=1`，且有两个候选动作：

| 动作 | 平均价值 `Q` | 先验 `P` | 已访问 `N` | 探索分 `U=P·√16/(1+N)` | 总分 `Q+U` |
|---|---:|---:|---:|---:|---:|
| A | 0.40 | 0.50 | 10 | 0.182 | 0.582 |
| B | 0.20 | 0.30 | 2 | 0.400 | 0.600 |

A 的当前价值更高，但它已经被考察得很多；B 的价值稍低，但先验不差且访问很少，因此 B 本轮总分反而更高。

如果本轮选了 B，`N_B` 从 2 增加到 3。即使考虑总访问次数从 15 增加到 16，B 下一轮的探索分也会变成 `0.30×√17/4≈0.309`，明显低于原来的 `0.400`。它不会因为“没看过”而永远占便宜。这就是 PUCT 的自平衡机制：

```text
Q 高             → 值得利用
P 高且 N 低      → 值得探索
某动作 N 增加    → 它的探索加成递减
其他动作相对更有机会
```

#### `c_puct` 到底在平衡什么

- `c_puct` **过小**：搜索很快被 `Q` 最高的动作占据，策略标签会很尖。优点是利用充分；缺点是 value 早期不准时容易锁死错误动作。
- `c_puct` **过大**：搜索更多按 `P` 和“少访问”分配次数，策略标签更广。优点是覆盖更多动作；缺点是搜索可能无法充分使用价值信息，最后只是近似复制网络先验。
- **合适的 `c_puct` 不是独立常数**：它取决于 `Q` 的尺度和可信度、合法动作数、先验的尖锐程度、simulations 以及训练阶段。例如改变 soft-value 混合会改变 `Q` 分布；即使 `c_puct` 数值没变，利用/探索比例也已经变了。

因此，调参时不应只问“`c_puct` 是多少”，而应同时看：

- `Q` 的均值、方差和饱和比例；
- 根策略 `P` 和搜索策略 `π` 的熵及差异，例如 KL；
- 多少合法动作真正获得了访问；
- 搜索后的对局强度是否比直接使用网络策略更好。

### 7.4 当前 Root-PUCT 的特殊性：`Q_a` 其实很快就等于固定 `q_a`

在通用完整 MCTS 中，同一根动作的不同次访问可能沿树走到不同的更深叶子，因此每次回传的价值不同，`Q_a=W_a/N_a` 是逐渐稳定的经验平均。

但当前默认 `SPARSE_PLY=1` 的 Root-PUCT 中，每次访问动作 `a` 时累加的都是同一个预先计算好的 `q_a`：

\[
N_a\leftarrow N_a+1,
\qquad
W_a\leftarrow W_a+q_a.
\]

所以动作 `a` 只要被访问过一次，就一直有：

\[
Q_a=\frac{W_a}{N_a}=q_a.
\]

这意味着，后续数千次乃至数万次“simulation”并没有获得新价值信息，只是在固定的 `P_a`、`q_a` 和 `c_puct` 上重复分配访问次数。所以这里的 simulation 更像“PUCT 分票迭代”，不是“向下多模拟一条新棋路”。

也因为如此，在当前默认实现中：

- `Q_a` 大，不代表经过了许多不同未来分支的反复验证；它本质上还是网络对一步子状态的单次判断。
- 增大 `N_a` 只会降低该动作的探索项，并改变最终 `π`；不会让 `Q_a` 因新证据而更准。
- 如果 `q_a` 的排序错了，增加 simulations 可能只是更稳定地放大错误排序。

### 7.5 从访问次数 `N` 到训练策略 `π`

PUCT 结束后，并不直接把 `Q` 最大的动作当作唯一标签，而是将访问次数转成概率分布：

\[
\pi(a\mid s)
=
\frac{N(a)^{1/\tau}}{\sum_bN(b)^{1/\tau}}
=
\operatorname{softmax}\left(\frac{\log N(a)}{\tau}\right).
\]

`N` 同时整合了价值 `Q`、网络先验 `P` 和探索压力，所以 `π` 是搜索给出的“教师分布”。

#### 温度 `τ` 调节的不是搜索过程，而是访问结果最后有多尖

- `τ=1`：`π` 正比于访问次数，保留原始访问比例。
- `τ<1`：访问多的动作被进一步放大，`π` 变尖，实际下法更确定。当 `τ→0` 时接近只选访问最多的动作。
- `τ>1`：访问比例被拉平，实际动作更多样，但也会执行更多次优动作。

当前自博弈默认：

- 前 10 个原子 ply：`τ=1.0`，保留较多开局多样性；
- 之后：`τ=0.1`，更集中地利用搜索结果。

#### Dirichlet 噪声：在搜索开始前故意扰动先验

自博弈时，根节点先验可被替换为：

\[
P'_a=(1-\varepsilon)P_a+\varepsilon\eta_a,
\qquad \eta\sim\operatorname{Dirichlet}(\alpha).
\]

- **`ε` —— 噪声混合比例**：`ε` 越大，原网络先验被改写得越多，状态覆盖更广，但搜索也更容易偏离当前强策略。
- **`α` —— 噪声的形状**：在维度固定时，`α` 较小时噪声更尖、更像随机重点推几个动作；`α` 较大时更均匀。

当前默认 `α=0.3, ε=0.25`。也就是根先验保留约 75% 的网络分布，混入约 25% 的随机 Dirichlet 分布。

#### opening random：直接改变实际执行的动作

前若干原子 ply 还可强制从合法动作中均匀采样，主脚本默认将这个长度从 6 线性退火到 0。它只替换实际执行动作，保存的 policy target 仍然是搜索访问分布 `π`。

几种探索手段不要混在一起：

| 机制 | 作用在哪里 | 变大时的主要影响 |
|---|---|---|
| `c_puct` | PUCT 分配访问的过程 | 更愿意访问先验较高且尚未充分访问的动作 |
| 温度 `τ` | 将访问次数变成 `π` 的阶段 | 当 `τ` 变大时，最终动作分布更平 |
| Dirichlet `ε` | 搜索开始前的先验 `P` | 噪声比例更大，搜索起点更多样 |
| opening random 长度 | 实际执行的 behavior action | 前期轨迹更随机，但训练分布更可能偏离部署分布 |

注意，温度阈值和 opening random 都按**原子动作数**计算，不是按完整的双方回合计算。一次落子后若紧接着还有连续标记或移除动作，会更快消耗退火预算。这是六洲棋阶段拆分与探索参数之间的项目特定耦合。

### 7.6 为什么当前实现中 simulations 不是越大越好

在标准完整 MCTS 中，增加 simulations 可以继续向树的更深层扩展，有机会发现新叶子和更深战术。

当前默认 Root-PUCT 中，增加 simulations 只会让访问分配更接近由固定 `P,Q,c_puct` 决定的稳定分布：

- **好处**：降低访问计数的离散噪声，让 `π` 更稳定；
- **做不到的事**：不会看得更深，不会获得新叶子证据，也不会降低神经网络 `q_a` 的系统偏差；
- **风险**：当 `q_a` 排序错误时，更多 simulations 可能只是更确信地放大错误排序；
- **计算代价**：次数越大，PUCT 分票循环越长，但边际信息增益很快下降。

所以 `65536` 次 Root-PUCT simulation 不能和 AlphaZero 的 65536 次树模拟等同。前者是对固定一层价值的更精细分票，后者才可能包含大量不同的树路径与叶节点评估。

当前 `SPARSE_PLY>1` 有一个 top-K lookahead 实验路径，但它是固定层的 max refinement，不是通用递归树扩展；默认关闭。其换手视角、minimax 聚合和多层递归语义应单独对拍后，才能把它当成完整 policy-improvement 能力。

---

## 8. 自博弈样本与训练目标

### 8.1 每一步保存什么

每个原子状态保存：

\[
(s_t,\; legalMask_t,\; \pi_t,\; playerSign_t).
\]

对局结束后，再把同一终局结果回填到整条轨迹，形成：

- hard value target；
- soft value target。

### 8.2 硬价值

\[
z_t=playerSign_t\cdot z_\text{black},
\quad z_\text{black}\in\{-1,0,1\}.
\]

没有折扣，整局所有状态共享同一终局结果。这与 AlphaZero 的 Monte Carlo return 相同。

### 8.3 当前软价值不是 `tanh`，而是裁剪后的 `tan`

终局统计黑白棋子差：

\[
\Delta=n_\text{black}-n_\text{white}.
\]

黑方视角软值为：

\[
m_\text{black}
=\operatorname{clip}
\left(
\tan\left(
\operatorname{clip}\left(k\frac{\Delta}{18},-\frac\pi2+\epsilon,\frac\pi2-\epsilon\right)
\right),-1,1
\right).
\]

然后按每一步行动方变换符号：

\[
m_t=playerSign_t\cdot m_\text{black}.
\]

主脚本默认 `k=2`。由于外层在 `|tan(x)|≥1` 时饱和，约在 `|Δ|≥8` 后就达到 `±1`。它给平局中的子力优势提供非零监督，但并不是严格胜率，也没有证明与最优策略完全一致。

### 8.4 混合目标

训练 value 使用：

\[
y_t=\operatorname{clip}\left((1-\lambda)z'_t+\lambda m_t,-1,1\right).
\]

其中 `λ = soft_label_alpha`。若启用 anti-draw，硬平局会先从 `0` 改成常数 `β`，即 `z'_t=β`。

当前大训练脚本默认：

\[
\lambda=1.
\]

这意味着默认 value **完全不学习硬 W/D/L，而是学习终局子力差的 tan 映射**。同时 anti-draw 即使传入，其有效幅度也只有 `(1-λ)β`；当 `λ=1` 时严格为零。

更需要注意的是，搜索遇到终局子节点时也用软子力值覆盖，而不是直接使用硬胜负值。因此当前默认搜索与训练在“优化子力代理目标”上是一致的，但它们与真正比赛效用 `win/draw/loss` 并不完全一致。

### 8.5 损失函数

策略损失只在合法动作上计算。当前实现使用：

\[
L_\pi
=D_{KL}(\pi\Vert p_\theta)
=-\sum_a\pi_a\log p_{\theta,a}-H(\pi).
\]

`H(π)` 对梯度是常数，所以优化梯度与交叉熵一致。

value 将 `y` 转为 101 桶 two-hot target `b(y)`：

\[
L_v=-\sum_i b_i(y)\log q_{\theta,i}(s).
\]

总损失：

\[
L=L_\pi+L_v.
\]

优化器是 Adam，带 weight decay、AMP、梯度裁剪和线性 warmup；warmup 后当前实现保持常数学习率。

---

## 9. 一轮训练中模型、数据和评估如何流转

当前大训练脚本按：

1. `latest model` 生成新一轮自博弈；
2. 训练得到 candidate；
3. candidate 对 Random 做健康度评估；
4. candidate 对 `best_previous` 做 gating；
5. `wins > losses` 才更新 best；
6. 无论 gating 是否通过，只要 candidate 数值有限，`latest model` 都会推进到 candidate；
7. 下一轮继续从 latest 训练和生成数据。

所以 gating 当前只维护一个“最好评估锚点”，并不会阻止被拒绝模型继续进入训练主链。其优点是接近 AlphaZero 的持续更新，不被高方差 gate 卡住；缺点是一次退化可以继续污染后续自博弈分布和优化器动量。

`vs_random` 适合判断“是否学会基本规则和战术”，但在模型较强后区分度和与模型间相对强度的相关性较弱。项目已有历史记录表明，最终选择应看固定条件的模型间对局、Elo/Bradley–Terry 趋势，而不是只看 Random 胜率。

高平局场景下，`wins > losses` 忽略 draws，但有效样本量其实是 `wins + losses`。例如 2000 局中只有 20 局分胜负，哪怕结果是 11:9，也远不足以证明稳定提升。更稳妥的 gate 应报告置信区间，或使用成对开局、SPRT/贝叶斯后验等序贯判定。

---

## 10. 与 AlphaGo、AlphaGo Zero、AlphaZero 的差异

先区分三个常被混用的名字：

- 2016 AlphaGo：专家棋谱监督策略 + 自博弈强化策略 + 独立 value + rollout policy + MCTS；
- 2017 AlphaGo Zero：只用规则和自博弈，单个 policy-value ResNet，去掉专家数据和 rollout；
- 2018 AlphaZero：把 AlphaGo Zero 泛化到围棋、国际象棋和将棋，持续使用最新网络训练。

原论文：

- [AlphaGo: Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961)
- [AlphaGo Zero: Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
- [AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)

### 10.1 总体对比

| 维度 | AlphaGo 2016 | AlphaGo Zero / AlphaZero | 当前六洲棋 V1 |
|---|---|---|---|
| 初始知识 | 人类专家棋谱 + 规则 | 只有规则 | 只有规则 |
| 网络 | 多个 policy/value 网络 | 单个 policy-value ResNet | 单个三策略头 + bucket value ResNet |
| 搜索 | 完整 MCTS，含 rollout/value | 完整神经 MCTS | 默认 Root-PUCT；固定评估全部一步子节点 |
| policy target | 搜索访问次数 | 搜索访问次数 | 根 PUCT 访问次数 |
| value target | 自博弈结果 | W/D/L 终局结果 | 默认终局子力软值；hard WDL 可混合 |
| 动作结构 | 围棋单点落子 | 各游戏统一合法动作 | 七阶段、220 编码、同方连续动作 |
| 换手语义 | 每步换手 | 通常每步换手 | 只在行动方实际切换时 value 取负 |
| 探索 | 搜索与自博弈噪声 | 根 Dirichlet + 温度 | 同样机制 + opening uniform curriculum |
| 模型更新 | 分阶段、多网络 | Zero 有 gate；AlphaZero 持续更新 | best 做 gate，但 latest 始终推进 |
| 平局处理 | 围棋通常有明确胜负 | 国际象棋/将棋包含 draw | 两个规则平局上限 + 软子力代理 |

### 10.2 六洲棋必须做的改造

#### 改造一：复合回合原子化

避免把“落子 + 选两个标记对象”等组合编码成巨大动作空间，使规则和 policy target 可管理。

#### 改造二：条件式价值翻转

标准两人交替实现常写 `parent_value = -child_value`。六洲棋必须检查 `current_player` 是否真的变化。

#### 改造三：阶段化动作头

placement、movement、selection 的结构不同，用共享棋盘特征加分头降低参数量。

#### 改造四：显式 draw protection

总步数和无吃子上限避免循环对局无限消耗自博弈算力。

#### 改造五：软局面优势

高平局使 `z=0` 过多，因此加入终局子力差。但目前是把它混入主 value，而不是辅助任务。

#### 改造六：GPU 友好的根搜索

一次性批量评估所有子节点，显著提高吞吐；代价是牺牲标准 MCTS 的动态深搜和误差纠偏能力。

所以我们不是简单“复刻 AlphaZero”，而是在规则复杂度、样本稀疏度和吞吐之间做了一组明确取舍。

---

## 11. 为什么训练动力学这么不稳定

### 11.1 第一层：自博弈本来就是非平稳闭环

普通监督学习近似是：

\[
D\text{ 固定},\quad \theta\rightarrow L_D(\theta).
\]

自博弈是：

\[
\theta_t
\rightarrow D(\theta_t)
\rightarrow \text{SearchLabels}(\theta_t)
\rightarrow \theta_{t+1}.
\]

一个参数同时改变：

- 行为策略访问哪些状态；
- 搜索 teacher 给什么 policy target；
- 对局胜负与平局比例；
- 每局长度和各阶段样本比例；
- 下一轮训练集难度。

因此小改动可能经过两三轮反馈后变成完全不同的数据分布。这也是“某个参数一换就训练不出来”的根本背景。

### 11.2 第二层：当前 policy improvement 对 value 偏差很敏感

完整 MCTS 可以沿多个深度重新评估叶子，并用回传缓和单步 value 偏差。当前 Root-PUCT 的 `q_a` 是一次网络前向给出的固定值。

于是训练链变成：

\[
v_\theta(s_a)
\rightarrow N_a
\rightarrow \pi_a
\rightarrow p_{\theta'}(a|s).
\]

value 的排序误差会直接被蒸馏进 policy。更多 root simulations 会让 `π` 更稳定，却不一定更正确。

### 11.3 第三层：超参数并不独立

#### `soft_value_k / soft_label_alpha` 与 `c_puct`

改变 value target 会改变 `Q` 的尺度和饱和程度。`c_puct` 不变时，探索项 `U` 相对于 `Q` 的比重已经变了。也就是说，改 soft label 同时隐式改了搜索的 exploitation/exploration 平衡。

#### simulations 与 temperature

simulations 决定访问计数差异，temperature 再对 `log N` 做缩放。增加 simulations 后不重新看 temperature，policy target 和实际行为都可能突然变尖。

#### opening random 与阶段原子化

随机步数按原子 ply 计数。相同的 `6` 在不同规则触发轨迹中可能对应不同数量的完整回合和标记选择，实际课程强度不是恒定的。

#### Dirichlet `alpha` 与合法动作数

固定 `α=0.3` 用在落子阶段几十个候选和选择阶段少量候选上，噪声总浓度 `nα` 不同，实际稀疏程度也不同。阶段分布一变，探索强度随之改变。

#### 数据量、epochs、batch 和 LR

真正决定优化强度的是每轮有效 optimizer steps、每个新样本被重复看到的次数、旧数据比例和 LR，而不是单独某一个参数。

### 11.4 第四层：目标本身可能互相拉扯

当前默认 `soft_label_alpha=1`，模型主 value 学的是终局子力差代理，而评估关心 W/D/L。两者通常相关，但不是同一目标：

- 为了多吃一子承担输棋风险，soft target 可能鼓励、胜率目标可能反对；
- 已经必胜时继续扩大子力差，可能浪费终局效率；
- 平局但子力领先会得到正 value，搜索可能把“有利平局”当成接近胜局。

anti-draw 常数也有语义风险：若双方视角都把 draw 设为负值，目标不再严格零和。它会鼓励冒险，但可能提升对 Random 胜率、同时降低最坏情况下的博弈强度。

更稳的设计是：

- 主头学习真实 W/D/L；
- 子力差、剩余棋子、终局 margin 作为辅助头；
- 搜索首先最大化胜率，在主价值接近时再小权重使用 margin tie-break。

这样既保留 dense representation learning，又不改变主博弈目标。KataGo 的经验也表明 score/ownership 辅助目标可以显著提高自博弈学习效率，但不需要把所有信号粗暴混成一个 scalar value。参考 [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565)。

### 11.5 第五层：状态混叠制造不可消除的标签噪声

如果输入没有 `moves_since_capture`，网络无法区分即将判平和刚刚吃过子的盘面。继续调 LR 只能改变模型如何折中冲突标签，不能消除冲突。

同理，remaining task 影响阶段内还要连续选择几次。legal mask 能约束当前动作，却没有告诉 value“完成当前选择后是否换手”。这会直接影响前述条件式 value 翻转后的长程回报。

### 11.6 第六层：长平局按 position 过采样

训练单位是局面，不是对局。一局 144 步平局产生的样本数可能远多于一局 40 步胜负局。

因此：

\[
\text{draw game ratio}=50\%
\not\Rightarrow
\text{draw position ratio}=50\%.
\]

后者可能高得多。当前 policy draw weight 是逐样本权重；若为 1，长平局会同时在 policy 和 value 中占据更大质量。只记录对局 draw rate 不足以判断训练信号。

### 11.7 第七层：优化器和模型分布一起移动

Adam 状态跨 iteration 延续可以减少每轮重启，但历史二阶矩来自旧数据分布。若 opening random、soft alpha、阶段比例或模型行为快速变化，旧动量可能造成过冲或响应迟钝。

这不意味着一定要重置 optimizer，而是应该把“模型连续、优化器连续、数据分布连续”作为一组共同假设验证。三者中任何一个发生突变，都应监控梯度范数、有效步长和按头损失。

BatchNorm 也有类似问题：自博弈阶段/棋局长度分布变化会改变各 phase 的混合比例，running statistics 可能跨轮漂移。

### 11.8 第八层：评估本身有噪声和口径错位

- `vs_random` 初期有用，后期容易饱和；
- 高平局时模型间对局的有效 decisive 数很少；
- vs-random 使用近确定性温度，vs-previous 当前默认采样，方差口径不同；
- gating 失败不阻止 latest 推进。

所以单轮胜率抖动不能直接解释为参数更新失败。至少要同时看固定 anchor、黑白分侧、置信区间、自博弈分布和 value calibration。

---

## 12. 当前主脚本中“看起来存在但实际没有启用”的能力

这一节非常重要，因为否则我们会对一个没有真正发生的实验做归因。

| 配置/能力 | 底层是否支持 | `scripts/big_train_v1.sh` 当前实际状态 |
|---|---|---|
| replay 输入 | `v1/train.py --self_play_replay_inputs` 支持 | `REPLAY_WINDOW` 只被声明、打印和基线检查；脚本未组装历史文件，也未传该参数 |
| cosine LR | bridge 有 warmup scheduler | `LR_COSINE_FINAL_SCALE` 只声明和打印；实际 scheduler 是 warmup 后常数 1.0 |
| anti-draw | train bridge 支持 | 大训练的 train 命令未传 `--anti_draw_penalty`，使用入口默认 0 |
| policy draw weight | policy loss 支持 | curriculum 计算并打印，但 train 命令未传 `--policy_draw_weight`，实际入口默认 1 |
| policy draw final schedule | 脚本声明 final | curriculum 函数只读取起始值，没有使用 final 值插值 |
| WDL auxiliary loss | 代码计算指标 | 权重固定为 0，不参与训练 |
| sparse multi-ply | 搜索代码支持实验路径 | 默认 `SPARSE_PLY=1`，主线是 root-only |

这意味着，当前稳定性分析应以“无 replay、无 cosine、draw policy 全权重、无 anti-draw、无 WDL aux、root-only”为真实基线，而不是以日志里的配置说明文字为准。

这不是建议现在顺手修全部参数。正确做法是先增加配置生效断言或 resolved-config 输出，再一次只启用一个算法变量做 A/B。

---

## 13. 高平局游戏应该怎么处理

### 13.1 先区分三种平局

#### A. 博弈论平局

双方足够强时，最优策略本来就可能走向平局。此时高平局不是 bug，也不应该为了“数据好看”强行惩罚。

#### B. 学习塌缩平局

模型只学会“不犯错”，但没有学会制造对手难题；或者 value/search 缺少区分平局内部状态的能力。这是需要解决的训练问题。

#### C. 行政性平局

因为 `144` 总动作、`36` 无吃子或 `max_game_plies` 达限而结束。它可能混合了循环、搜索保守、规则计数不可见等原因。

当前统计主要把三者合成一个 draw 数。第一步不是设计更复杂 reward，而是记录 `draw_reason`、终止阶段、终局子力差、最后一次吃子距离和对局长度。

### 13.2 为什么平局样本不是完全没用

即使 hard value 为 0，平局状态的搜索访问分布仍能训练 policy：它告诉网络“在这个局面下，搜索更偏好哪些合法动作”。所以不应简单丢弃所有平局。

真正的问题是：

- value 的区分信号弱；
- 长平局位置数过多；
- policy teacher 也可能只是当前 value 的自我确认；
- `v=0` 混淆确定平局与高不确定胜负局。

### 13.3 推荐的目标结构

#### 主任务：WDL 分布

显式预测：

\[
(p_W,p_D,p_L),\qquad p_W+p_D+p_L=1.
\]

搜索主价值：

\[
v=p_W-p_L.
\]

显式 WDL 能区分：

- 确定平局：`(0,1,0)`；
- 高风险均衡：`(0.5,0,0.5)`。

二者 scalar expectation 都是 0，但决策风险含义完全不同。

#### 辅助任务：局面优势和终局 margin

可预测：

- 黑白剩余棋子差；
- 是否接近形成方/洲；
- 距离下一次提吃的结构性特征；
- 剩余无吃子步数；
- 终局 piece-delta distribution。

辅助损失帮助 trunk 学密集特征，但搜索仍以真实 WDL 为主。若确实需要打破等价平局，可使用很小的 lexicographic/tie-break 权重，而不是让 material 完全替代 WDL。

### 13.4 数据侧处理

建议按以下顺序：

1. 记录 game-weighted 和 position-weighted 的 W/D/L；
2. 按 `outcome × phase × draw_reason × game_length` 分桶采样；
3. 保留平局 policy 样本，但限制超长平局对 batch 的垄断；
4. decisive 与 draw 的 value 样本可分层配额，而不是永久删除 draw；
5. replay 采样按数据年龄和有效信号加权，避免旧弱 policy target 等权淹没新数据；
6. 引入历史模型/对手池，减少当前模型自我镜像造成的循环策略。

仓库中的 `filter_decisive_jsonl.py` 可以用于诊断或构造对照集，但“只训练 decisive”会造成选择偏差，不宜直接作为默认方案。

### 13.5 探索侧处理

- opening random 退火：早期扩状态覆盖，后期接近部署分布；
- 从历史中间局面或多样起始局面继续自博弈，减少所有轨迹都从同一初始局面出发的强相关；
- 对不同阶段使用与合法动作数匹配的 Dirichlet 参数；
- 使用多种搜索预算：少量高预算局面提供强 teacher，大量低预算对局提供独立性和覆盖。

KataGo 的 playout-cap randomization 和多辅助目标是值得借鉴的方向；Go-Exploit 也展示了从多样中间状态启动自博弈可提高 value 覆盖和样本独立性，见 [Targeted Search Control in AlphaZero for Effective Policy Improvement](https://arxiv.org/abs/2302.12359)。

### 13.6 不建议首先做什么

- 不要先缩短判平规则来制造更多 decisive；这改变了游戏；
- 不要把所有平局统一设成负奖励后就认为解决了稀疏监督；
- 不要只增加 root simulations；它不增加搜索深度；
- 不要一次同时改 random moves、soft alpha、LR、replay 和 search depth；无法归因；
- 不要只看 train loss 或单次 vs-random。

若要增加逐步奖励，优先考虑独立辅助预测；必须进入 return 时，应研究 potential-based shaping：

\[
r'(s,a,s')=r(s,a,s')+\gamma\Phi(s')-\Phi(s),
\]

并严格处理当前行动方视角。直接累加“吃子奖励 + anti-draw”容易改变最优策略，模型可能学会刷子力而不是赢棋。

---

## 14. 一个可验证的稳定性研究计划

这里不要求现在训练出强模型，目标是先让每次实验都可解释。

### 阶段 0：语义审计，优先于调参

1. **补全或证明状态表示**：对 counters/remaining tasks 构造同盘面异状态，验证网络输入是否冲突；
2. **配置生效表**：运行时输出 requested/resolved/effective value，并对主线关键参数做断言；
3. **目标对拍**：抽样轨迹检查 hard/soft target、黑白符号、同方连续动作；
4. **搜索对拍**：小状态空间与穷举 minimax 或 legacy full MCTS 比较；
5. **平局原因拆分**：144、36、max ply 分开计数。

### 阶段 1：冻结一个真正的基线

固定：

- commit、seed、初始 checkpoint；
- self-play games 和实际 position 数；
- root simulations、温度、噪声；
- optimizer steps、batch、LR；
- 评估开局、颜色和 search budget。

至少跑 3 个 seed 的小规模曲线。单 seed 的“成功配方”可能只是幸运进入了一个正反馈盆地。

### 阶段 2：一次只检验一个算法假设

建议顺序：

| A/B | 只改变什么 | 回答的问题 |
|---|---|---|
| A | 完整状态输入 vs 11 通道 | 状态混叠贡献多大 |
| B | hard WDL 主头 + material aux vs 当前混合 scalar | 目标错位贡献多大 |
| C | root-only vs 语义验证后的 2/3-ply 或 full MCTS | 搜索纠偏是否是瓶颈 |
| D | 固定 opening random vs 退火 | 覆盖和部署分布如何权衡 |
| E | latest self-play vs 历史对手池 | 循环/遗忘是否减轻 |
| F | 无 replay vs 真正生效的短 replay | 方差和 stale target 如何变化 |

### 阶段 3：统一监控面板

至少记录：

#### 数据分布

- game/position 两种口径的 W/D/L；
- draw reason、局长、phase 占比；
- black/white 分侧胜率；
- legal action count 和 policy entropy；
- piece delta 分布；
- 数据年龄和 replay 占比。

#### 学习质量

- policy KL、value CE 分头曲线；
- WDL calibration / Brier score；
- value 按 phase、按距终局步数的误差；
- 梯度范数、更新范数、Adam 有效步长；
- search policy 相对 raw policy 的 KL：`KL(π_search || p_net)`。

最后一个指标尤其重要：若长期接近 0，搜索没有提供有效 policy improvement；若突然很大，搜索 teacher 与网络分歧剧烈，训练可能不稳定。

#### 棋力

- Random：仅作早期健康度；
- 固定弱/中/强 anchor；
- paired openings 的 candidate-vs-anchor；
- tournament Elo/BT；
- 每个结果带置信区间和 decisive 有效局数。

---

## 15. 分享时可能被问到的问题

### Q1：这到底算不算 AlphaZero？

训练闭环、双头网络、自博弈和访问次数蒸馏是 AlphaZero 风格；但默认搜索不是标准完整 MCTS。最准确的说法是“AlphaZero-style self-play with a GPU-batched Root-PUCT improvement operator”。

### Q2：为什么不用 Q-learning？

动作合法空间随阶段变化，终局奖励延迟且搜索可调用精确规则模型。policy-value + planning 能把规则模型直接用于决策，并用搜索策略作为密集 policy teacher；这比为 220 个动作直接学习离策略 Q 更自然。

### Q3：平局 value=0 有什么问题？

`0` 本身没有错。问题是它同时表示确定平局、胜负各半、未知局面，以及在当前输入缺失计数器时的互相冲突标签。需要 WDL 分布和完整状态，而不是简单把 0 改成负数。

### Q4：soft value 为什么能帮助？

它让大量 hard draw 也有子力差监督，早期更容易学出“吃子通常更好”的方向。但它只是代理目标；权重过高会让模型优化 margin 而非胜率，且改变 `Q` 尺度后还会间接改变 PUCT 探索强度。

### Q5：为什么 simulations 增大后反而可能变差？

当前 simulations 不产生新叶子，只反复分配固定 `q_a`。如果 value 排序错了，更大的预算会把错误变成更尖的 policy target。它减少的是计数噪声，不是 value bias。

### Q6：高平局是不是说明模型变强了？

可能是，也可能是保守塌缩。必须看对固定对手强度、draw reason、终局 margin、policy entropy 和模型间 Elo。高平局不是单调棋力指标。

### Q7：为什么不只保留胜负局？

胜负局 value 信号强，但只保留它们会偏向容易分胜负的状态和激进策略；平局的 search policy 仍有信息。更稳妥的是分桶采样、限制长局权重，并把 WDL 与优势辅助任务解耦。

### Q8：当前最可能的第一瓶颈是什么？

从算法正确性看是状态不完整和 root-only 搜索；从实验可解释性看，是若干配置声明与实际生效路径不一致。应先消除这两类问题，再判断网络容量和超参。

---

## 16. 代码阅读地图

| 想理解什么 | 入口 |
|---|---|
| 正式规则 | [docs/rules.md](docs/rules.md) |
| Python 状态与阶段 | [src/game_state.py](src/game_state.py) |
| 网络、桶价值转换 | [src/neural_network.py](src/neural_network.py) |
| 220 动作与策略损失 | [src/policy_batch.py](src/policy_batch.py)、[v0/python/move_encoder.py](v0/python/move_encoder.py) |
| V1 Root-PUCT | [v1/python/mcts_gpu.py](v1/python/mcts_gpu.py) |
| PUCT 固定叶值访问分配 | [v0/src/mcts/root_puct_fused.cu](v0/src/mcts/root_puct_fused.cu) |
| 自博弈主循环 | [v1/python/self_play_gpu_runner.py](v1/python/self_play_gpu_runner.py) |
| 轨迹回填与视角 | [v1/python/trajectory_buffer.py](v1/python/trajectory_buffer.py)、[v0/src/bindings/module.cpp](v0/src/bindings/module.cpp) |
| 训练损失 | [v1/python/train_bridge.py](v1/python/train_bridge.py) |
| staged 训练入口 | [v1/train.py](v1/train.py) |
| 当前大训练参数与 gate | [scripts/big_train_v1.sh](scripts/big_train_v1.sh) |
| V1 历史设计记录 | [v1/Design.md](v1/Design.md) |

---

## 17. 最后一页：分享总结

1. 主干是“网络 → 搜索改进 → 自博弈数据 → 拟合搜索和终局 → 新网络”的闭环。
2. 六洲棋的核心特殊性是七阶段原子动作和同一玩家连续决策，因此 value 不能机械逐步取负。
3. 220 维动作和三策略头将不同阶段映射到统一训练空间，但 movement 的加性分解限制了表达能力。
4. 当前默认 V1 是 Root-PUCT，不是完整 AlphaZero MCTS；simulations 主要重分配固定叶值，不增加深度。
5. 当前默认 value 学终局子力软值，而不是硬 WDL；这缓解平局稀疏，却引入代理目标偏差。
6. 训练不稳定来自自博弈非平稳、搜索纠偏弱、超参强耦合、状态不完整、长平局过采样和评估噪声。
7. 平局不能简单删除或统一惩罚：先区分博弈论平局、学习塌缩和平局上限，再用 WDL 主任务 + 优势辅助任务 + 分桶采样处理。
8. 下一步先做语义与配置审计，再逐项 A/B；不要用更多算力掩盖目标、状态或搜索定义的问题。

一句收尾：

> 这套系统最值得分享的不是“我们用了 AlphaZero”，而是我们在一个多阶段、高平局的小众棋类里，如何把规则、动作、价值视角和自博弈闭环接起来；而下一阶段的关键，是让搜索真的提供可靠改进，让网络看到完整状态，并让训练目标与最终想赢的游戏保持一致。
