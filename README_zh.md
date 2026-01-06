# 基于 MCTS 的强化学习六洲棋 AI 系统实现

本项目聚焦传统棋类“六洲棋”的智能对弈系统，涵盖规则建模、MCTS 强化学习训练与人机对战系统实现。

[六洲棋][1]是广泛分布于中国北方的一种传统棋类。该棋类规则各地不一，仅以山东中部地方规则作为实现标准。具体规则叙述见[规则文档]。(./rule_description.md) 

[1]: https://zh.wikipedia.org/wiki/%E6%96%B9%E6%A3%8B

## 📰 里程碑（News）

* 2025-10-29：完成 legacy 主体框架搭建并通过运行验证。

  * GeForce RTX 3060训练3小时，即可以81%-11%-8% (win-draw-lose，200 eval game) 战胜 Random Agent。

* 2025-12-27：完成 v0 主体框架搭建并通过运行验证。

  * 使用C++/CUDA重写，在训练效果不降低的前提下，将自博弈速度提升至原来的~16倍，速度瓶颈从规则引擎/合法动作生成转移到神经网络前向推理。

---

# 规则说明（Rules）

本文档记录了在重构后（将每个动作拆分为原子阶段）所采用的最新规则集。
以下内容取代了旧版本中将多个决策捆绑为单个动作的描述。

---

## 🎮 游戏概述（Game Overview）

* **棋盘**：6 × 6 网格（共 36 个交叉点）。
* **棋子**：两种颜色，`BLACK`（黑）与 `WHITE`（白），每方初始各有 18 枚棋子。
* **目标**：通过吃子移除对方所有棋子。
  如果在限定步数内双方都未能达成此目标，游戏以平局结束。

---

## ⚙️ 阶段结构（原子动作）

游戏在任意时刻都处于下列阶段之一。
每个阶段会持续到所有待处理的子任务完成后，才会交由下一位玩家或进入下一阶段。

---

### 1️⃣ `PLACEMENT`（落子阶段）

* 当前玩家在一个**空的交叉点**上放置一枚棋子，
  但不能放在目前被对方标记的位置上。
* 放置完成后，检查新棋子（若其位置之前未被标记）是否形成以下结构：

  * **正方形**：2×2 方块 → 生成 **1 个** 对方棋子标记任务；
  * **直线**：同行或同列 6 枚棋子（忽略标记棋） → 生成 **2 个** 对方棋子标记任务。
* 若存在待处理标记 → 进入 `MARK_SELECTION` 阶段；
  否则：

  * 若棋盘已满 → 进入 `REMOVAL` 阶段；
  * 若未满 → 轮到对手，继续 `PLACEMENT`。

---

### 2️⃣ `MARK_SELECTION`（标记选择阶段）

* 逐个处理待处理的标记任务。
* 当前玩家为每个待标记任务选择一个合法的对方棋子：

  * 合法目标为**未被标记的对方棋子**；
  * 当存在未构成方块/直线的对方棋时，**优先选择这些棋子**。
* 当所有标记任务完成后：

  * 若棋盘已满 → 进入 `REMOVAL`；
  * 否则交换玩家并返回 `PLACEMENT`。

---

### 3️⃣ `REMOVAL`（移除阶段）

* 若有被标记的棋子 → **同时移除双方所有被标记的棋子**，
  然后进入 `MOVEMENT` 阶段，由 `WHITE` 先行。
* 若没有标记（棋盘填满但无触发标记），则进入 **强制移除序列**（`FORCED_REMOVAL`）：

  1. `WHITE` 先移除一枚不属于方块/直线结构的 `BLACK` 棋子；
  2. 然后 `BLACK` 移除一枚符合同样规则的 `WHITE` 棋子；
  3. 随后进入 `MOVEMENT` 阶段，由 `WHITE` 先行。

---

### 4️⃣ `MOVEMENT`（移动阶段）

* 当前玩家选择并移动自己的一枚棋子，沿**正交方向**（上下左右）移动一步至相邻空位。
* 每次移动后检查是否形成新结构：

  * 方块 → 待捕获数 +1；
  * 直线 → 待捕获数 +2。
* 若存在待捕获任务 → 进入 `CAPTURE_SELECTION`；
  否则交换玩家并继续 `MOVEMENT`。
* **无可移动规则（No-move rule）**：
  若玩家无合法移动，则可以移除一枚不在方块/直线中的对方棋（若无“普通”棋，则任意移除一枚）。
  该操作通过特殊动作 `no_moves_remove` 实现，随后进入 `CAPTURE_SELECTION` 阶段由对方执行**反移除**。

---

### 5️⃣ `CAPTURE_SELECTION`（吃子选择阶段）

* 逐个执行待捕获任务。
* 合法目标与标记阶段一致：优先移除未构成方块/直线的对方棋。
* 每次吃子立即从棋盘移除。
* 若对方棋子全部被移除 → 当前玩家立即获胜。
  否则当所有捕获完成后 → 交换玩家并返回 `MOVEMENT`。

---

### 6️⃣ `COUNTER_REMOVAL`（反移除阶段）

* 仅在执行 `no_moves_remove` 动作后触发。
* 由对手（之前无法行动的一方）移除刚才行动玩家的一枚棋子，遵循相同合法性规则。
* 反移除完成后，若该玩家仍有棋子，则返回 `MOVEMENT` 阶段并交回回合。

---

## 🏁 游戏结束条件（End-of-Game Conditions）

* **吃子胜利**：若某次吃子（包括反移除）使对方棋子全部消失，则当前玩家立即获胜。
* **步数上限平局**：自对弈和训练脚本设定最大步数为 **200 步**。
  若达到上限仍无人获胜，则判定平局（双方得分为 0）。
  此限制用于防止无限循环，可根据需要调整。
* **无合法动作**：若 `generate_all_legal_moves` 返回空列表，则该玩家判负（通常意味着无子可下或早期阶段无法落子）。

---

## 📘 附加说明（Additional Notes）

* 被标记的棋子会在阶段 1（落子阶段）中保留在棋盘上，直到后续的 `REMOVAL` 阶段才被移除。
  被标记棋不可用于形成新的方块或直线。
* 当落子阶段结束且没有任何标记时，必须先执行**强制移除**，再进入移动阶段。
* “无可移动”与“反移除”机制可防止僵局：若某方完全阻挡对手，则给予对手一次移除机会，随后执行反移除，再恢复正常回合。
* 实现层面上，每个 `GameState` 都维护**待标记数**与**待捕获数**，
  确保在 `MARK_SELECTION` 与 `CAPTURE_SELECTION` 阶段中，每次仅处理一个原子任务。

---

该规则集与当前代码逻辑（`rule_engine.py`、`move_generator.py`、以及 Monte Carlo 搜索模块）一致。
在编写测试、调试游戏逻辑或将引擎移植到其他语言时，请以此文档为权威参考。


## ⚡ 性能对比（V0 C++ Core vs Legacy Python）

使用 `tools/benchmark_self_play.py` 进行性能基准测试，对比 v0（C++ 核心）与 legacy（纯 Python）实现的自博弈性能。

**测试环境**: Linux H20, CUDA, 2局游戏, 200次MCTS模拟/步

```bash
# 分别运行两个版本的性能分析
python -m cProfile -s cumtime -m tools.benchmark_self_play --num-games 2 --mcts-simulations 200 --device cuda --skip-legacy
python -m cProfile -s cumtime -m tools.benchmark_self_play --num-games 2 --mcts-simulations 200 --device cuda --skip-v0
```

### 总体对比

| 指标 | Legacy | V0 | 提升 |
|------|--------|-----|------|
| **总时间** | 74.65秒 | 16.43秒 | **4.5倍** |
| **cProfile时间** | 72.7秒 | 7.5秒 | **9.7倍** |
| **函数调用次数** | 8670万 | 378万 | **减少23倍** |
| **positions/sec** | 5.36 | 23.44 | **4.4倍** |

### Legacy 版本热点

| 函数 | 累计时间 | 调用次数 |
|------|---------|---------|
| `mcts.py:expand` | 26.5秒 | 78,910 |
| `move_generator.py:apply_move` | 22.7秒 | 686,095 |
| `neural_network.py:get_move_probabilities` | 16.6秒 | 78,910 |
| `rule_engine.py:check_squares` | 11.0秒 | 1,582,834 |

### V0 版本热点

| 函数 | 累计时间 | 调用次数 |
|------|---------|---------|
| `neural_network.py:forward` | 4.2秒 | 2,661 |
| `torch.conv2d` | 1.9秒 | 26,610 |
| 模块导入 | 2.5秒 | 160 |

### 关键发现

1. **V0 几乎没有规则引擎开销** — `rule_engine.py` 系列函数在 Legacy 占用 ~40秒，V0 完全用 C++ 实现
2. **V0 的 MCTS 逻辑不可见** — 核心搜索已移至 C++
3. **V0 主要瓶颈在神经网络推理** — 游戏逻辑已高度优化，NN 推理占 V0 总时间的 ~55%
4. **V0 的 NN 调用次数更少** — 从 5,136 次降到 2,661 次（批量推理优化）

---

## 📂 代码结构（Code Structure）

---

### 🧠 核心游戏逻辑（Core Game Logic）

* **`src/game_state.py`** —— 定义 `GameState` 容器、阶段枚举（phase enum）、玩家枚举（player enum）以及辅助方法（如复制、统计棋子数量、待标记/待捕获计数器等）。
  这是全局使用的**标准游戏状态快照**。

* **`src/rule_engine.py`** —— 实现每个原子阶段的状态转换逻辑：落子、标记选择、移除、移动、捕获选择、强制移除、反移除等。
  同时包含形状检测逻辑（`detect_shape_formed`、`is_piece_in_shape`）以及兼容旧代码的封装函数（`apply_move_phase1/3`）。

* **`src/move_generator.py`** —— 生成当前阶段的合法动作（并可直接执行）。
  调度对应的 rule engine 函数，保持对外接口 `generate_all_legal_moves` / `apply_move` 的稳定性，供 MCTS 与训练循环使用。

---

### 🧩 学习与搜索（Learning & Search）

* **`src/neural_network.py`** —— 实现 AlphaZero 风格的网络（包含三个策略头 + 一个价值头）：`pos1` 负责落子/移动终点选择，`pos2` 负责移动起点，`mark_capture` 统一负责各类提子/吃子/标记目标的定位。模块还提供张量转换工具与 `get_move_probabilities` 方法。
  是训练与 MCTS 搜索的核心模块。

* **`src/mcts.py`** —— Monte Carlo 树搜索实现。
  使用神经网络提供先验概率与价值估计，处理搜索模拟、日志记录，并将搜索结果转换为可执行的动作分布。

* **`src/train.py`** —— 自博弈与训练循环的调度器。
  负责生成对局、收集 `(state, policy, value)` 样本、训练网络、管理检查点（checkpoint）并执行评估对局。

* **`src/evaluate.py`** —— 工具模块，用于让模型与基线（随机代理或历史最优模型）对战进行离线评估。

---

### 数据流程图

```
训练迭代开始
    ↓
[自博弈阶段]
    ├─ self_play() 
    │   ├─ self_play_single_game() (每局)
    │   │   ├─ 初始化: GameState(), MCTS()
    │   │   └─ 游戏循环 (每步):
    │   │       ├─ mcts.search(state)
    │   │       │   ├─ 建立/复用根节点
    │   │       │   ├─ 批量模拟循环:
    │   │       │   │   ├─ Selection (PUCT选择)
    │   │       │   │   ├─ Expansion:
    │   │       │   │   │   ├─ generate_all_legal_moves()
    │   │       │   │   │   ├─ state_to_tensor()
    │   │       │   │   │   ├─ model(batch) → (log_p1, log_p2, log_pmc, value)
    │   │       │   │   │   ├─ get_move_probabilities()
    │   │       │   │   │   └─ node.expand()
    │   │       │   │   ├─ Backpropagation
    │   │       │   └─ 提取根策略 (基于访问次数)
    │   │       ├─ 保存 (state, policy)
    │   │       ├─ 采样动作
    │   │       ├─ apply_move()
    │   │       └─ mcts.advance_root()
    │   │   └─ 返回 (game_states, game_policies, result, soft_value)
    │   └─ 返回 training_data: List[Tuple[...]]
    ↓
[训练阶段]
    ├─ 数据转换: (state, policy, value, soft_value)
    ├─ train_network():
    │   ├─ 创建 DataLoader
    │   └─ 训练循环 (每个epoch):
    │       └─ 每个batch:
    │           ├─ model(states) → (log_p1, log_p2, log_pmc, value_pred)
    │           ├─ 计算策略损失 (KL散度)
    │           ├─ 计算价值损失 (MSE)
    │           └─ loss.backward() + optimizer.step()
    └─ 更新模型权重
    ↓
[评估阶段]
    ├─ 与RandomAgent对战
    ├─ 与BestModel对战
    └─ 决定是否更新best_model.pt
    ↓
保存检查点 → 下一迭代

```
