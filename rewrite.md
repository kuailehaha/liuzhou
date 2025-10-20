## Motivation
- 现有实现将“形成形状后的提子/标记”直接并入同一次动作，导致 `capture_positions` 出现大量组合，MCTS 先验被稀释，温度收缩后几乎从不采样到提子分支。
- 搜索树节点无法区分同一落子但不同提子目标的价值，既增加分支数也使策略学习困难，训练数据难以覆盖真正关键的提子场景。
- 代码层面难以维护：`move_generator`、`rule_engine`、`apply_move` 都要同时处理落子、标记、提子，修改规则/调试行为时牵一发动全身。

## Revised Game-State Flow
- 每个原子动作只完成一件事；由阶段 (Phase) 切换驱动完整流程，避免一次动作里塞多个决策。
- 新的阶段及含义：
  1. `PLACEMENT`：单纯落子；若形成方/洲则写入 pending 信息并转到 `MARK_SELECTION`。
  2. `MARK_SELECTION`：当前玩家逐个标记对方棋子，直到满足 pending 数量；完成后若棋盘满则进入 `REMOVAL`，否则换手回 `PLACEMENT`。
  3. `REMOVAL` / `FORCED_REMOVAL` / `COUNTER_REMOVAL`：沿用现有逻辑，移除被标记或强制移除、反制移除。
  4. `MOVEMENT`：只处理棋子移动；若移动后形成方/洲则记录 pending 并切到 `CAPTURE_SELECTION`。
  5. `CAPTURE_SELECTION`：逐个提掉 pending 数量的对方棋子，验证合法性；完成后判断胜负并回到 `MOVEMENT`（换手）。
  6. `NO_MOVES_REMOVE` 情形仍然调用现有流程，完成后进入 `COUNTER_REMOVAL`，保持原节奏。
- `GameState` 需要新增字段：pending 标记/提子数量、剩余次数，以及可选的候选坐标缓存，保证状态拷贝时能 faithfully 复制。

## Planned Structural Changes
- `game_state.py`：添加 pending 字段及拷贝/初始化逻辑，调整 `__str__` 输出协助调试。
- `rule_engine.py`：
  - 拆分 `apply_move_phase1`、`apply_move_phase3`，分别处理落子/移动与后续的 `apply_mark_selection`、`apply_capture_selection`。
  - 维护阶段切换与 pending 计数的递减、胜负判定。
- `move_generator.py`：
  - Phase1 仅生成落子动作；新增 `MARK_SELECTION`、`CAPTURE_SELECTION` 阶段的合法动作生成。
  - Phase3 只生成移动动作；捕获流程由新阶段负责。
  - `apply_move` 根据新的阶段分支调用对应处理函数。
- `mcts.py`：
  - 更新节点动作格式（不再含 `capture_positions`），保持搜索树节点与新阶段一致。
  - 确认 Dirichlet 噪声和打印逻辑仍然工作；必要时调整策略展示。
- `neural_network.py`：
  - `get_move_probabilities` 改为只读 `from_position` / `to_position` / 单点选择动作，与新的动作字典同步。
  - 若需要，可考虑新增针对标记/提子的策略头或复用现有 `mark_capture` 通道。
- 测试与脚本：更新自动化测试和调试脚本，使其覆盖新的阶段流程；重新生成示例日志/样例对局，验证提子概率已恢复。
