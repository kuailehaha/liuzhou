# 方法说明

AlphaZero 风格自博弈训练：规则引擎定义合法空间，神经网络评估局面与动作分布，MCTS 在网络引导下搜索，自博弈数据迭代训练模型。

## 1. 方法定位

项目围绕六洲棋的规则特性和工程可训练性做实现，核心覆盖：

- 可验证的规则引擎（Python / C++ / CUDA 三层对拍）
- 与阶段语义对齐的动作编码（220 维，三策略头）
- 可扩展的 MCTS 搜索
- 大规模 staged 训练流水线
- 人机对战前后端

## 2. 网络输出与状态表示

核心模型为 `ChessNet`（`src/neural_network.py`）。

### 输入

状态编码采用棋盘张量表示，覆盖：

- 当前行动方棋子
- 对手棋子
- 双方被标记棋
- 当前阶段 one-hot

默认输入通道数为 `11`。

### 输出

网络输出由三个策略头和一个价值头组成：

- `log_p1`：位置相关策略头
- `log_p2`：移动来源相关策略头
- `log_pmc`：标记 / 提吃 / 移除相关策略头
- `value_logits`：价值头输出，bucketed value logits

三头设计让动作评分结构与六洲棋的阶段语义对齐，而不是把所有动作压成单一同构分布。

## 3. MCTS 与模型协作

在主线实现中：

- 策略头给出动作先验
- 价值头估计局面价值
- MCTS 在合法动作空间内分配搜索预算
- 根节点访问计数转成训练用策略目标

训练样本遵循 `(s, π, z)` 风格，实现层面保留张量化批处理、staged 数据落盘与加载、与 v0 动作编码的一致性。

## 4. v1 训练方法

### 自博弈阶段

`v1/python/self_play_gpu_runner.py` 批量执行 GPU-first self-play，直接输出张量化训练数据。

### 训练阶段

`v1/python/train_bridge.py` 桥接张量样本到单卡、DataParallel 或 DDP 训练路径。训练链路包含批量化策略损失、bucketed value 目标、staged 数据加载、draw 控制参数。

### staged 调度

`v1/train.py` 把训练分成 `selfplay` → `train` → `eval` → `infer` 四个阶段，对应调度脚本 `scripts/big_train_v1.sh`。

## 5. 模型选择与评估

评估分为三类：

- `vs_random`：健康度探针
- `vs_previous`：gating 判定（候选模型 `wins > losses` 即通过）
- 锦标赛 / Elo / BT：相对强度排序

`vs_random` 不能单独代表真实强度，模型长期选择依赖锦标赛与排序结果。

## 6. 三代实现的关系

Legacy 建立了完整的规则与训练闭环；v0 将关键路径迁移到 C++/CUDA；v1 在 v0 能力之上重构了自博弈到训练的数据流，形成当前的 staged pipeline。

逐阶段设计细节与历史决策记录在 `v1/Design.md`。
