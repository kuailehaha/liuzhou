# 高难问题解答手册

本文件逐条回答 `docs/faq.md` 中的 216 个问题。

---

## 1. 规则与问题建模

**Q1** 六洲棋规则（7 Phase、方/洲触发、优先级约束）适合 AlphaZero 的核心原因：**没有公开棋谱数据库**。监督学习需要人类专家棋谱，而六洲棋是民间棋类，不存在这些资源。手写评估函数需要领域专家迭代——7 个阶段的组合复杂度使其极难手写。AlphaZero 闭环只需要规则引擎（已有 `rule_engine.py` / `rule_engine.cpp`）和可微模型（`ChessNet`），无需外部数据。

**Q2** 原子阶段拆分确实将复杂性转移到了 `ActionEncodingSpec`（220 维）。但这是有意的设计：每个原子阶段的合法动作集是确定的（`generate_all_legal_moves` 按 `Phase` 分发），动作编码与阶段一一对应。如果不拆分，就需要设计"一步到位"的复合动作空间（例如"落子+标记2个目标"），这种组合空间更大且结构更不规则。220 维 flat 空间虽然稀疏，但通过 `legal_mask` 精确控制有效区间，训练时不会在非法动作上浪费梯度。

**Q3** 合并 `MARK_SELECTION` 为一次性多选会将问题变成组合优化：从 N 个候选中选 K 个（K=`pending_marks_required`=1或2）。当 K=2 时，需要从 N 个候选中选 2 个的有序或无序对，动作空间变为 O(N²)。当前逐个选择是 2×O(N)，更扁平。但逐个选择引入了"先选哪个"的人为顺序性——第一次标记选择可能改变第二次的候选池（因为已标记的棋子被排除）。合并后 `pending_marks_remaining` 递减逻辑需要改为原子递减（一次减到 0），`apply_mark_selection` 中的循环需要重写为批量操作。训练是否更容易取决于 K：K=1 时合并无意义，K=2 时合并可能减少序列长度但增加动作空间复杂度。

**Q4** 不对称确实存在。`MARK_SELECTION` 合法池 = 对手未标记棋（可达 18），`CAPTURE_SELECTION` 合法池 = 对手非结构棋（通常 < 10），`FORCED_REMOVAL` 合法池 = 对手非结构棋（通常更少，因为棋盘满时结构多）。在 `batched_policy_loss` 中，`masked_log_softmax` 只在合法动作上归一化，所以动作池大小影响 softmax 的 entropy 基线：大池 entropy 高、小池 entropy 低。小池阶段的 policy loss 梯度更集中，学习信号更强但也更容易过拟合到少数局面。

**Q5** 不连续跳变确实会发生。`is_piece_in_shape` 检查 2×2 和长度 6 结构。如果对手剩余唯一一个"普通棋"被提吃或移除，下一次标记/提吃的合法集突然从"只能选普通棋"变为"所有棋子都可选"。这个跳变在 `generate_mark_targets` 第 98-101 行：`pool = opponent_normal_pieces if opponent_normal_pieces else opponent_pieces`。对策略网络来说，这需要模型理解"还剩几个普通棋"——但这个信息没有显式编码在 11 通道中，需要从 ch0-1（棋子位置）隐式推断。

**Q6** 10 层 `PreActResBlock` 每层 3×3 conv 的感受野递增 2。10 层后感受野 = 1 + 10×2 = 21 > 6，完全覆盖 6×6 棋盘。理论上网络可以"发现"方（2×2 局部模式）和洲（整行/列模式）。但问题是：这些结构的检测需要精确的二值匹配（4 个同色且非标记），而卷积学到的是软特征。显式注入结构通道（例如 `is_piece_in_shape` 的输出作为 ch11-12）会降低学习难度，但代价是增加工程耦合和通道数。当前选择是让网络自行发现，这更符合 AlphaZero"不注入人类知识"的哲学。

**Q7** 两条平局保护都是工程需要。`MAX_MOVE_COUNT=144`（`game_state.py:29`）防止无限循环，对应棋盘 36 格 × 每方 2 步的上限估计。`NO_CAPTURE_DRAW_LIMIT=36`（`game_state.py:31`）仿照中国象棋的无吃子判和。删除它们后，走子阶段可能出现双方来回移动的无限循环，训练中的自博弈会卡死。这不是"扭曲"真实博弈——六洲棋原始民间规则并无明确的平局判定，工程上必须补充。36 步的具体值可以调整，但删除不可行。

**Q8** `moves_since_capture` 归零逻辑在 `move_generator.py:124-137`。`PLACEMENT` 和 `MARK_SELECTION` 阶段直接置 0（第 125-127 行），其他阶段比较棋子总数变化。标记阶段置 0 意味着标记动作不计入"无吃子"计数，这是合理的——标记本身不移除棋子，它只是在 `REMOVAL` 阶段才生效。如果标记也计入（设为累加），那么整个落子阶段的计数器会始终为 0，`NO_CAPTURE_DRAW_LIMIT` 只在走子阶段生效。当前设计正确反映了规则语义：只有实际移除棋子的动作才算"吃子"。

**Q9** `LOSE_PIECE_THRESHOLD=4` 在 `game_state.py:174-177` 的 `get_winner` 中使用：`if black_pieces < 4: return WHITE`。这意味着**不是减到 0 才判输，而是减到 4 以下就判输**。这与 FAQ 第 1 条和 `rules.md` 中"使对方无子可用"的描述不完全一致。阈值 4 暗示了一种"实质性优势即胜"的替代判定方案——当一方只剩 3 子或更少时，认为其已无法有效对抗。这个常量不是"未启用"的，它是当前胜负判定的核心参数。

**Q10** 三个来源都有贡献，但权重不同。当前实验数据显示平局率一度超过 98%（`decisive_game_ratio` 1.23%），大量 `z=0` 使 value head 信号极弱——这是最直接的。7 阶段复杂度增加了序列长度（典型对局 50-200 步），但 `pending_marks/captures` 的跨阶段依赖主要影响 policy 而非 value。根因分析：平局主导 > 序列长度 > 阶段依赖。

**Q11** 是的，ch2-3 编码了双方标记位。落子时 `apply_placement_move` 第 39-40 行检查 `(r,c) in opponent_marked`，拒绝落在对手标记位上。网络能看到 ch3（对手标记位），理论上可以学到"对手标记了哪些位置 → 我不能落子 → 这些位置有战略价值"。但实际学到这种双重语义需要训练过程中遇到足够多的"标记位封锁落子空间"的情形。

**Q12** 固定顺序（White 先移除 Black 棋子）确实引入了先后手不对称。`apply_forced_removal` 第 348-368 行硬编码了这个顺序。White 先选择移除哪个 Black 棋子，拥有"先手优势"。如果改成随机先手，训练数据的 value target 需要对两种顺序取期望——这增加了训练复杂度。由棋子数决定先后是更符合规则精神的方案，但增加了 `GameState` 的条件分支。当前固定 White 先手的设计是为了简化实现，代价是系统性地让 White 在 `FORCED_REMOVAL` 中略有优势。

**Q13** `COUNTER_REMOVAL` 只在一方完全无法移动时触发（`_generate_moves_no_moves`）。在 6×6 棋盘走子阶段，空位稀少时才会出现无子可动。实际自博弈中触发频率很低——大多数对局在 `MOVEMENT` 阶段通过吃子分出胜负或触发平局保护。低频阶段的策略学习确实受样本不足影响，但由于 `selection` 区间（180-215）被多个阶段共享，`out_mark` 头仍然能从其他阶段获得梯度信号。

**Q14** `detect_shape_formed`（`rule_engine.py:465-479`）**先检查 line 再检查 square**，返回优先级 `line > square > none`。**不累加**：如果同时形成 line 和 square，只返回 `"line"`（2 个标记任务），忽略 square。这意味着同步形成方和洲时，只有洲的效果生效。`_set_pending_marks` 直接覆盖（不是加法），`pending_marks_required` 被设为 2（洲）而非 3（2+1）。

**Q15** 是的，新棋能立刻参与结构。`check_lines` / `check_squares` 的 `marked_set` 参数来自当前状态的 `marked_black/white`。标记棋被 `REMOVAL` 阶段移除后，标记集被清空（`process_phase2_removals` 第 186-187 行）。新落子不在任何标记集中，可以自由参与结构检测。这个边界条件理论上应该在 `check_rule_engine_cases.py` 中覆盖，但需要具体查看测试用例列表才能确认。

**Q16** "所有对手棋都在结构中"的回退条件在棋盘后期更容易触发——走子阶段棋子少，剩余棋子倾向于被保护在结构中。对手故意保持所有棋子在结构中（方或洲）是一种可行防守策略：它限制了攻击方的标记/提吃选择，迫使攻击方只能打破结构才能进攻。但维持全结构需要至少 4 个棋子组成一个方（或 6 个组成洲），且不能有任何游离棋子。这在实际对局中不容易维持。

**Q17** 是的，批量移除不需要考虑交互效应。`process_phase2_removals` 第 175-187 行直接遍历棋盘，移除所有标记棋（黑/白都移除）。移除顺序不影响结果——因为被标记的棋子在移除前就已经不参与新结构的形成（标记棋被 `marked_set` 排除）。移除后 `marked_black/white` 被清空，结构状态需要在下一次落子/走子时重新评估。

**Q18** 改 `BOARD_SIZE` 到 8 需要同步修改：`ActionEncodingSpec`（`placement_dim=64`, `movement_dim=256`, `selection_dim=64`）、`state_to_tensor` 的通道数不变但空间尺寸变为 8×8、`ChessNet` 的输入尺寸、`check_lines` 中洲的条件（行/列长度从 6 变为 8？或仍然是 6？）、`NO_CAPTURE_DRAW_LIMIT` 和 `MAX_MOVE_COUNT` 可能需要调整。这种耦合是设计选择——项目只服务于 6×6 六洲棋，不需要通用化。将其视为技术债只在需要支持多棋盘尺寸时才成立。

**Q19** 两者互补。`no_moves_remove` 解决的是**一方完全无法移动**的死锁（允许该方先移除对手一子，再被对手反移除一子），但这不能解决"双方都能移动但选择不吃子"的僵持。`NO_CAPTURE_DRAW_LIMIT=36` 解决后者——连续 36 步无吃子则判和。实际上 `no_moves_remove` 极少触发（需要一方所有棋子都被围住），`NO_CAPTURE_DRAW_LIMIT` 是更常见的平局出口。

**Q20** 三层同步成本确实很高。修改 Python `rule_engine.py` 后需要同步修改 `v0/src/rules/rule_engine.cpp`，重新编译 `v0_core`，再运行 `check_rule_engine_cases.py` 验证。这个周期（修改→编译→测试）可能需要 10-30 分钟。这确实阻碍了快速规则实验。缓解方案：先在 Python 层验证规则变更，确认后再移植到 C++。但当前没有"只用 Python 规则引擎做自博弈"的 v1 路径（v1 依赖 `v0_core.self_play_step_inplace`）。

## 2. 状态表示与动作表示

**Q1** `moves_since_capture`（无吃子步数）、`pending_marks_remaining`（剩余标记任务数）、`pending_captures_remaining`（剩余提吃任务数）确实没有通道编码。`pending_marks_remaining` 影响 `MARK_SELECTION` 阶段能执行多少次标记——但模型通过 ch4-10 的 one-hot 知道当前处于 `MARK_SELECTION`，且每次执行后 `pending_marks_remaining` 递减。模型无法直接知道"还剩几次"，但由于典型值只有 1 或 2，且形状结构在棋盘上可见，模型理论上可以推断。`moves_since_capture` 更难推断——它需要追踪历史信息。如果接近 `NO_CAPTURE_DRAW_LIMIT=36`，模型应该考虑主动吃子，但它看不到这个计数器。这是一个已知的信息缺失。

**Q2** 移除 phase one-hot 后，模型需要从棋子分布和标记位推断阶段。`PLACEMENT`（棋盘未满、无标记）vs `MOVEMENT`（棋盘已有空位、走子阶段标志）的区分需要模型理解"棋盘是否已经经历过 REMOVAL"。这在纯视觉特征上很难——两种阶段的棋盘都可能有相似的棋子分布。实验结论：移除 phase one-hot 几乎必然会让训练显著变慢，甚至可能无法收敛到有意义的策略，因为 `selection` 区间被 4 个不同阶段共享，模型必须知道当前阶段才能正确输出。

**Q3** `GlobalPool` 在 `PolicyHead` 中提取 mean/max/std（`neural_network.py:72-80`），得到 3×64=192 维全局向量，通过 `gpool_linear`（192→64）映射后 broadcast-add 回空间特征。这种设计让每个位置的策略分数能感知全局棋盘状态（如总棋子数、标记分布）。如果替换为简单全局平均池化（只有 mean，64 维），会丢失 max 和 std 的信息——max 反映了最活跃的通道（可能对应最重要的结构），std 反映了特征的空间不均匀性。对于六洲棋，结构检测（方/洲）依赖局部模式与全局上下文的交互，`GlobalPool` 的丰富性可能是策略质量的关键贡献者。

**Q4** 统一 220 维 flat head 的劣势：movement 动作是 `(from, to)` 对，flat head 需要独立学习 144 个 from-to 组合的概率，无法利用"from 和 to 的概率可分解"的结构先验。三头设计通过 `log_p2[from] + log_p1[to]` 强制分解，将 144 维降为 36+36=72 的有效参数空间。优势来自**样本效率**：在训练早期，模型只需学好 36 个"好的来源位置"和 36 个"好的目标位置"，而不是学好 144 个组合。但代价是牺牲了非分解表达能力。

**Q5** 加性分解 `log_p2[from] + log_p1[to]` 假设 from 和 to 的概率独立。当"从 A 出发好但到 B 不好，同时到 B 从 C 出发好"时，分解无法精确表达。在六洲棋中，这种非分解偏好出现在：走棋后形成方/洲需要特定的 from-to 组合（例如从 (2,3) 移动到 (2,2) 完成一个 2×2 方，但从 (2,3) 移动到 (2,4) 不形成结构）。形成结构的"关键一步"高度依赖具体的 from-to 对，而不是独立的 from 偏好和 to 偏好。

**Q6** `movement_dim=144` 中大量非法动作被 `legal_mask` 遮盖。`masked_log_softmax` 确保 softmax 只在合法动作上归一化。被 mask 的位置在 `build_combined_logits` 中可能有非 -inf 值（log_p2+log_p1 的组合），但在 loss 计算中不产生梯度。真正的问题不是"噪声概率"，而是 movement 区间的有效动作通常只有 10-20 个，softmax 在这些有效动作上可能过于集中（因为它们数量少）。这不是稀疏编码的问题，而是 softmax 在小动作集上自然尖锐的特性。

**Q7** 保留位 217-219 在 `build_combined_logits` 第 135 行填 0，在 `masked_log_softmax` 中被 mask 为 -inf。它们不影响 softmax 归一化——`masked_log_softmax` 的 `logsumexp` 只在 `mask=True` 的位置上计算（第 150-151 行）。mask 为 False 的位置输出 0，不参与 loss。这些保留位的存在只增加了少量内存（3 个 float），对训练没有实际影响。

**Q8** 四个阶段共享 `out_mark` 头是设计简化。`MARK_SELECTION`（选对手棋标记）、`CAPTURE_SELECTION`（选对手棋提吃）、`FORCED_REMOVAL`（选对手棋移除）、`COUNTER_REMOVAL`（选行动方棋移除）的目标语义相似度：前三个都是"选择对手的一个棋子"，语义接近；`COUNTER_REMOVAL` 是"选择自己（被困方）的一个棋子被移除"，语义不同。但由于 phase one-hot 告诉网络当前阶段，`out_mark` 可以条件性地输出不同分布。问题在于共享权重是否限制了表达——一个 1×1 conv 只有 64 个参数（policy_channels → 1），是否足够区分 4 种语义。

**Q9** 是的，当合法动作只有 1 个时，policy 是 deterministic（概率 = 1.0），target entropy = 0，KL = CE - H = -log(1.0) - 0 = 0。梯度为零。当合法动作只有 2 个时，梯度很小但非零。这在 `FORCED_REMOVAL` 后期和 `REMOVAL`（只有一个 `process_removal` 动作）阶段会出现。这不是 bug——单一合法动作的局面确实不需要学习策略。但这些样本仍然占用训练 batch 的空间。

**Q10** 是的。考虑 `FORCED_REMOVAL` 只有 3 个合法动作 vs `PLACEMENT` 有 30 个合法动作。softmax 在 3 个动作上归一化后，每个动作最低概率 ~1/3。在 30 个动作上，每个动作最低概率 ~1/30。这意味着小动作集的均匀基线概率更高，模型更容易"满足"。但在训练中，target policy `π` 来自 MCTS 访问计数，通常不是均匀的——它会集中在 value 最高的动作上。所以"过度自信"的风险更多出现在 target 本身就不够锐利的情况下。

**Q11** `PreActResBlock`（`neural_network.py:82-95`）的前向路径是 `BN→ReLU→Conv→BN→ReLU→Conv + skip`。标准 ResBlock 是 `Conv→BN→ReLU→Conv→BN + skip→ReLU`。关键差异：PreAct 的 skip connection 传递的是**未经 ReLU 的特征**，梯度可以无损地通过 skip path 回传。在 10 block 深度下，梯度消失不是主要问题（6×6 棋盘的任务相对简单），但 PreAct 的 BN 位置让特征分布在进入 conv 前被规范化，有助于训练稳定性。

**Q12** 101 bin 均匀分布在 [-1,1]，每个 bin 宽度约 0.02。中心 bin（index 50）对应 value=0。在平局附近，0.02 的分辨力意味着模型可以区分"完全平局"（bin 50）和"极微弱优势"（bin 51，对应 value ≈ 0.02）。相比直接 `tanh` 输出标量，bucket 表示在 value ≈ 0 附近的梯度更平滑——`tanh` 的梯度在 0 附近最大（=1），但单一输出的梯度噪声也最大。bucket 表示通过 two-hot 编码分散梯度到两个相邻 bin，更稳定。

**Q13** 两个 head 对 `GlobalPool` 的使用确实不同。`PolicyHead` 将 pool 结果 broadcast-add 到空间特征，保留空间维度——每个位置的策略分数 = 局部特征 + 全局上下文。`ValueHead` 直接用 pool 结果做线性映射——只需要全局信息，不需要空间分辨。`PolicyHead` 需要更精细的空间-全局交互：例如"全局看来对方有结构优势（全局），所以应该在这个位置（局部）吃子"。`ValueHead` 只需要总结整体局面好坏，空间细节不重要。

**Q14** 模型参数量主要集中在 trunk（128ch × 10 blocks ≈ 590k 参数）。减半到 64ch 会将 trunk 参数降到 ~150k。`PolicyHead` 参数从 ~15k 降到 ~5k，`ValueHead` 从 ~25k 降到 ~10k。哪个头更敏感：value head 通过 `GlobalPool`（3×channels）→`Linear` 映射，通道减半使全局信息维度从 192 降到 96。policy head 同样受影响但它有 broadcast-add 机制补偿。推测 value head 对容量更敏感，因为它需要从全局信息中准确预测标量值。

**Q15** 耦合确实很紧。`DIRECTIONS` 在 `policy_batch.py:24` 硬编码为 4 方向，`build_combined_logits` 的 movement 拼接逻辑（第 123-132 行）完全依赖 4 方向。扩展到 8 方向需要修改 `DIRECTIONS`、`movement_dim`、`build_combined_logits`、`action_to_index`、C++ 端的 `fast_legal_mask` 和 `project_policy_logits_fast`。这种耦合是六洲棋规则只支持 4 方向移动的反映，不是设计缺陷。

**Q16** `state_to_tensor` 第 28-29 行根据 `player_to_act` 确定 `self_val` 和 `opp_val`，ch0/ch1 始终是"我的棋子/对手棋子"。这种对称化编码意味着 Black 和 White 看到的输入结构相同（只是棋子颜色映射翻转）。网络**无法区分自己是 Black 还是 White**——它只知道"我"和"对手"。如果先后手存在系统性不对称（如 `FORCED_REMOVAL` 中 White 先手），模型无法学到"作为 White 我应该更激进"的策略。这是有意的对称化设计，符合 AlphaZero 传统。先后手不对称需要通过训练数据中的 value signal 自然习得。

**Q17** `encode_actions_fast` 在 CUDA 不可用时回退到 CPU C++ 路径。两条路径都在 `v0/src/game/fast_legal_mask.cpp` 中实现，使用相同的规则逻辑——legal mask 是 bool 张量，不涉及浮点运算。结果应严格一致。自博弈数据的不可复现性主要来自 Dirichlet noise 和温度采样的随机性，不是 legal mask。

**Q18** 正确。加性分解无法表达联合概率中的交互项。具体例子：假设 (3,2)→(3,3) 能形成方，但 (3,2)→(2,2) 不能。模型需要给前者高概率、后者低概率。在分解表示下，`log_p2[(3,2)]` 对两个目标动作的贡献相同——差异只能通过 `log_p1[(3,3)]` vs `log_p1[(2,2)]` 来表达。但 `log_p1` 同时服务于 placement 阶段，其含义是"这个位置适合落子"，而不是"从某个来源到达这个位置好"。这种语义混用限制了 movement 阶段的策略精度。

**Q19** `policy_channels=64` 和 `value_channels=64` 的对称是默认选择。`ChessNet.__init__` 中这两个参数独立可调。当前没有实验数据证明哪个头需要更多容量。由于 policy head 有 3 个输出头（各一个 1×1 conv），而 value head 只有 1 个线性输出，policy head 的参数效率可能更高（3×64 参数 vs 101 个参数）。调整建议需要消融实验。

**Q20** 101 bin 使得 bin 50 的中心恰好是 0。如果用 100 bin，中心值为 [-1, -0.98, ..., -0.02, 0.02, ..., 0.98, 1.0]，0 不在任何 bin 中心。`scalar_to_bucket_twohot` 会将 value=0 分散到 bin 49（center=-0.02）和 bin 50（center=0.02）的 two-hot 编码。在高平局率下，大量 `z=0` 样本的 target 集中在这两个 bin 上，而不是单一 bin——这可能降低 calibration 精度但增加数值鲁棒性。当前选择 101（奇数）使得 0 对应精确的 one-hot，对高平局率任务更友好。

## 3. MCTS 设计

**Q1** depth-1 搜索 vs 完整树搜索：`V1RootMCTS` 只展开一层（root → children），然后用 `root_puct_allocate_visits` 分配虚拟访问。完整 `MCTSCore` 会逐次展开、回传、重新选择。depth-1 的损失：无法评估"两步之后的局面"。在六洲棋中，一些关键决策（如走棋形成方/洲再提吃）需要 2-3 步才能看到效果。`num_simulations=1536` 在 depth-1 中只是在 1 层子节点上反复分配，所有模拟都在比较同一层的 value 估计——超过合法动作数后，额外模拟只在微调概率分布，不增加搜索深度。

**Q2** `root_puct_allocate_visits` 是一次性批量分配，而非逐次模拟。传统 PUCT 每次模拟：选择 → 展开 → 评估 → 回传。批量分配等价于假设所有子节点的 leaf value 在搜索过程中不变（因为没有更深展开）。在 depth-1 搜索中，子节点 value 确实不变（只做一次评估），所以批量分配与逐次模拟在结果上**等价**。差异只在多层搜索中出现——逐次模拟会在后续模拟中使用更新后的统计，而批量分配不会。

**Q3** `_evaluate_values_only`（`mcts_gpu.py:811-815`）只调用 `_forward_model` 然后提取 value。子节点没有 prior 是正确的——在 root-only 搜索中，子节点不需要 prior（prior 只在选择下一层子节点时使用）。如果扩展到 depth-2，需要为第一层子节点也计算 policy（使用 `_evaluate_batch` 而非 `_evaluate_values_only`），以便为第二层展开提供 prior。这需要将 `child_eval_mode` 改为 `"full"` 并增加第二层的 `root_pack_sparse_actions` + `batch_apply_moves` 逻辑。

**Q4** `_soft_tan_from_board_black`（`mcts_gpu.py:677-691`）使用 `tan(clamp(k * delta/18))` 而非 `tanh`。材料差比例 vs value head 输出的差异：value head 可能对某些局面有系统性偏差（如高估先手、低估平局趋势），而材料差是客观指标。对终局状态（一方棋子 < 4），材料差是确定性信号，value head 则可能不确定。使用材料差避免了 value head 对终局的潜在误判。两者严重分歧的情况：一方棋子很多但位置极差（被完全包围），material_delta > 0 但实际劣势。

**Q5** `exploration_weight=1.0` 对应 AlphaZero 标准值。六洲棋分支因子在 PLACEMENT 阶段最高（~35），MOVEMENT 阶段中等（~10-20），SELECTION 阶段最低（~3-10）。AlphaZero 对围棋用 cpuct 约 2.5（分支因子 ~250），对国际象棋用 ~1.0（分支因子 ~35）。六洲棋分支因子接近国际象棋，所以 1.0 合理。在 PLACEMENT 阶段可能偏向利用（分支因子更大，探索项被稀释），在 SELECTION 阶段偏向探索（分支因子小，探索项占比大）。

**Q6** `_soft_tan_from_board_black` 对终局用 soft value 而非硬编码 ±1。当一方棋子 < `LOSE_PIECE_THRESHOLD=4` 时为胜利终局，材料差确定（如 3:15 → delta = -12/18 → soft ≈ -0.93）。这种 soft encoding **确实低估必胜**：tan(2×12/18) ≈ tan(1.33) ≈ 0.97 而非 1.0。但代价很小——0.97 vs 1.0 的差异在 MCTS value 回传中几乎不影响排序。Soft 的好处：提供了关于**多有利**的渐进信号，而不是二值的胜/负。

**Q7** `dirichlet_alpha=0.3` 适合平均分支因子 ~10-30 的游戏。AlphaZero 使用 `alpha = 10/average_branching_factor`：围棋 10/250 ≈ 0.03，国际象棋 10/35 ≈ 0.3。六洲棋平均分支因子约 15-20，理论最优 alpha 约 0.5-0.7。当前 0.3 偏低，噪声更集中在少数动作上（更"尖锐"的噪声），探索的广度不如 0.5-0.7。但 0.3 是保守选择，避免过多随机性。

**Q8** `temperature_threshold=10` 对应前 10 步（ply，不是回合）。在六洲棋中，前 10 ply 全部在 `PLACEMENT` 阶段（双方各落 5 子，棋盘从空到有 10 枚棋子）。假设是：开局多样性主要体现在落子位置的选择上，一旦棋盘格局形成（10+ 子），策略应该更确定性。10 步是比较保守的值——AlphaZero 对围棋用 30 步。六洲棋总步数更短（典型 50-200 ply），10 步占比 5-20%，与围棋的 30/300 ≈ 10% 相当。

**Q9** 低温度（0.1）下 `_stable_legal_policy_from_visits`（`mcts_gpu.py:858-903`）在 log 空间操作：`logits = log(visits) / 0.1`，差异被放大 10 倍。如果两个动作访问 100 次和 90 次，log(100)/0.1 ≈ 46.1，log(90)/0.1 ≈ 45.0，softmax 后差异巨大。策略目标确实非常尖锐。这让 policy loss 的梯度集中在最高概率动作上——对于高确信度的局面这是正确的，但可能让模型对 value 估计的微小误差过于敏感。

**Q10** `force_uniform_random_mask` 在 `opening_random_moves` 步内生效。这些步的 policy target 来自 `search_batch` 的 `policy_dense`——但由于 `force_uniform_mask` 只影响 `chosen_action`（采样哪个动作），**不影响 `policy_dense`**（MCTS 原始策略输出）。实际上 `policy_dense` 仍然是 MCTS 搜索结果，不是均匀分布。但 `chosen_action` 是均匀随机的，所以产生的**对局轨迹**更多样。这些步骤的 policy target 仍然是 MCTS 质量的，只是执行的动作是随机的。

**Q11** CUDAGraph 缓存 key = `(device_idx, num_roots, max_actions, batch_size, sample_moves)`。`max_actions` 取决于当前 batch 中最大合法动作数。在自博弈中，不同阶段有不同的合法动作数上限（PLACEMENT: 36, MOVEMENT: ~50, SELECTION: ~18, FORCED_REMOVAL: ~15）。当 `concurrent_games=8192` 时，8192 个对局可能分布在不同阶段，`max_actions` 是所有对局中的最大值。如果 batch 中有 PLACEMENT 阶段的对局（max=36），graph 就需要按 36 维度创建。阶段分布变化时可能创建新 graph，但典型的 max_actions 值域有限（5-36），graph 数量不会太多。

**Q12** `_child_values_to_parent_perspective`（`mcts_gpu.py:694-713`）基于 `same_to_move = children.eq(parents)`。在 `MARK_SELECTION` 连续标记时，side-to-move 不变（`apply_mark_selection` 第 149-151 行：`pending_marks_remaining > 0` 时不切换玩家），所以 `same_to_move = True`，value 不翻转。这是正确的：同一玩家连续标记，子节点的 value 已经是当前玩家视角，不需要翻转。

**Q13** 当只有 1 个合法动作时，`root_pack_sparse_actions` 仍会创建矩阵，但 PUCT 分配只有一个选择。`num_simulations` 次模拟全部分配给唯一的子节点。这确实是浪费——但由于 `max_actions > 1` 是 Dirichlet noise 的前提条件（`mcts_gpu.py:972`），单一合法动作的搜索会跳过噪声注入，直接选择唯一动作。计算浪费来自 model forward（评估唯一子节点的 value），但这只有一次。

**Q14** `V1RootMCTS` 没有子树复用。`MCTSCore` 的 `AdvanceRoot(action_index)` 用 `CompactTree` 保留子树——根节点的选定子节点变成新根，其子树的统计信息被保留。v1 每步重建的设计简化了实现（无需管理 tree 状态），但确实丢弃了可复用信息。在 `num_simulations=1536` 下，每步搜索的主要成本是 model forward（所有子节点 value 评估），子树复用可以节省部分子节点评估，但由于 v1 是 root-only（没有子树），复用收益有限。

**Q15** `EvalBatcher`（C++ 端）通过 `timeout_ms=2` 积攒请求。v1 的 `_forward_model` 直接做同步 batch forward。引入 `EvalBatcher` 的收益：当多个自博弈对局需要独立推理时，batcher 可以收集它们的请求并合并成一个大 batch。但 v1 已经通过 `concurrent_games=8192` 实现了批量化——所有活跃对局在同一步中批量搜索。`EvalBatcher` 的异步设计在 v1 的同步批处理模式下没有额外优势。

**Q16** `V1_FINALIZE_GRAPH` 在 nsys 或 `CUDA_LAUNCH_BLOCKING` 下自动禁用。性能分析时看到的是 non-CUDAGraph 路径的时间，而实际运行可能用 CUDAGraph。正确的基准测试：分别测量 CUDAGraph on/off 的吞吐，在相同条件下（无 nsys，无 launch blocking）。可以通过环境变量 `V1_FINALIZE_GRAPH=off` 强制禁用来做 A/B 对比。

**Q17** 固定 `exploration_weight=1.0` 在整个对局中意味着探索偏好不随阶段变化。在 MOVEMENT 后期（棋子少、分支因子 5-10），1.0 的探索权重使得探索项 `sqrt(N)/（1+n）` 在少量模拟后就能主导决策——模型可能过度探索低 value 的子节点。降低到 0.5-0.8 会更偏向利用 value head 的估计。但动态调整 exploration_weight 需要额外的调度逻辑，违反"不增加开关"的设计原则。

**Q18** `batch_apply_moves_compat` 对所有合法子动作做批量 apply。当 30+ 合法动作 × 8192 并发对局时，子节点总数可达 30×8192 ≈ 246k。每个子节点状态是 `GpuStateBatch` 的切片（board: 6×6 int8, marks: 2×6×6 bool, 等），约 200 bytes/子节点，246k × 200B ≈ 50MB。加上 model forward（246k 个状态的 batch），显存需求约 246k × 11×6×6 × 4B ≈ 400MB（float32 输入）。在 H20（96GB 显存）上不是瓶颈，但在 RTX 3060（12GB）上可能需要降低 `concurrent_games`。

**Q19** 终局由 `NO_CAPTURE_DRAW_LIMIT` 触发时，`_terminal_mask_from_next_state` 检测到 `moves_since_capture >= 36`，标记为终局。`_soft_tan_from_board_black` 基于材料差输出 soft value，但实际 `z=0`（平局）。这两个信号**确实矛盾**：soft value 可能是 +0.3（黑方多 3 子）但结局是平局。在 MCTS 搜索中，soft value 用于子节点估计，不直接用于训练 target。训练 target 的 value 来自 `finalize_trajectory_inplace` 中的 `result_from_black`（=0），soft value 通过 `soft_label_alpha` 混合。如果 `soft_label_alpha > 0`，训练 target 会受 soft value 影响——这可能让 value head 对这类"材料领先但平局"的局面产生偏差。

**Q20** deterministic eval（argmax）测的是"最强单步"，sampled eval（温度采样）测的是"策略分布质量"。一个模型可能 argmax 很强但分布不好（次优动作概率太高），或者分布好但 argmax 选择不够稳定。在锦标赛（`temperature=1.0, sample_moves=True`）中，胜率反映的是策略分布的整体质量而非峰值表现。两种模式测量的**不是同一种棋力**——前者是"确定性下限"，后者是"平均水平"。

**Q21-30 概括回答：**

**Q21** stable(1536) vs aggressive(131072) 的策略目标确实不可比。131072 次模拟的访问计数分布远比 1536 次更收敛、更接近"真实最优"。两个 profile 产出的模型不应直接比较——它们有效地在不同的"MCTS 质量级别"上训练。

**Q22** `sqrt(1536) ≈ 39.2` vs `sqrt(128) ≈ 11.3`，探索项增长 3.5 倍。但 `1+N` 的分母也在增长——高访问次数的子节点探索项衰减。总体上，高模拟次数让每个子节点都被充分探索，不会过度探索。

**Q23** root-only 搜索中所有子节点 value 来自同一个 value head forward——如果 value head 有系统性偏差（如对某种棋盘格局总是输出 +0.1），所有子节点会共享这个偏差，PUCT 的相对排序不受影响（偏差被消掉）。但如果偏差与子节点特征相关（如对先手棋局高估），排序可能被扭曲。完整树搜索通过多层回传平均化了单次 value head 的偏差。

**Q24** `v0_core.project_policy_logits_fast` 在投影前不使用 legal_mask——它纯粹是将三头 logits 组合成 220 维。legal_mask 在之后应用（`root_pack_sparse_actions` 用它来确定哪些子节点是有效的）。所以 prior 分布确实可能给非法动作分配概率，但这些非法动作会被 mask 排除在 PUCT 搜索之外。

**Q25** 子节点处于 `CAPTURE_SELECTION` 阶段。root-only 搜索看到的是提吃选择前的局面——子节点的 value 来自 value head 对"提吃选择阶段"的估计。模型需要从这个状态预测最终结果。如果模型对 CAPTURE_SELECTION 阶段的 value 估计不准（例如不理解提吃后对手棋子会减少），搜索质量会下降。

**Q26** transposition table 在自博弈中理论上有用——对称开局（如黑先 (2,2) 白先 (3,3) vs 黑先 (3,3) 白先 (2,2)）可能产生相同局面。但 v1 的 GPU batch 搜索模式下，transposition 检测需要跨 batch 的状态比较，计算开销可能抵消收益。

**Q27** `dirichlet_epsilon=0.25` 在 2-3 个合法动作时确实过高——25% 噪声可能完全扭曲 prior。代码中 `counts.gt(1)` 检查（只有多于 1 个合法动作才加噪声）保护了单一动作的情况，但没有根据动作数调整 epsilon。改进：`epsilon = min(0.25, 0.5 / max_actions)` 可以在少动作时减少噪声。

**Q28** `REMOVAL` 后 `current_player = Player.WHITE`（`rule_engine.py:191`）。如果 Black 在 `PLACEMENT` 阶段形成结构并完成标记，进入 `REMOVAL`，`REMOVAL` 后变为 White to move in `MOVEMENT`。`_child_values_to_parent_perspective` 基于 `current_player` 比较——`REMOVAL` 是一个特殊阶段（只有一个 `process_removal` 动作），其子节点的 `current_player` 已经被设为 White。如果 parent 是 Black（在 MARK_SELECTION 后进入 REMOVAL 的仍是 Black），子节点变为 White，`same_to_move = False`，value 翻转。这是正确的。

**Q29** 随训练轮次增加，value head 更准确→root-only 搜索的子节点估计更准→depth-1 搜索质量接近更深搜索。但同时模型更强→对手也更强→需要更深搜索才能找到最优应对。两个趋势部分抵消。实际上差距可能先收窄（value head 改善的收益大于搜索深度需求的增长），后扩大（模型足够强后，depth-1 成为瓶颈）。

**Q30** 动态 `num_simulations` 是合理的优化方向。在只有 2 个合法动作的阶段，2 次模拟就足够。省下的模拟可以分配给 30+ 动作的阶段。但实现上，`root_puct_allocate_visits` 是统一 batch 调用，需要为每个根节点设置不同的模拟次数——当前 API 不支持。

## 4. 强化学习目标

**Q1** 区分方法：检查 value head 的 bucket 分布。如果"局面无差别"，bucket 分布应该在中心 bin 集中但两侧有合理的尾巴（反映真实的胜率不确定性）。如果"信号不足导致塌缩"，bucket 分布会退化为固定的窄峰，不随局面变化而变化。可以通过检查 `bucket_logits_to_scalar` 在不同局面上的输出方差来诊断——方差过低意味着 value head 没有学到有区分度的评估。

**Q2** 101 bin、bin 宽 ≈ 0.02。对于"微弱优势" vs "平局"的区分：1 子优势 → soft value ≈ 0.11 → 占据 bin 55-56 附近（center ≈ 0.10-0.12）。与 bin 50（value=0）有 5 个 bin 的距离。precision 足够。但问题在于**训练数据的 target**：对局以平局结束时 `z=0`（bin 50），即使一方长期领先。模型从 `z=0` 学不到"领先但平局"的信息——除非通过 `soft_value_targets` 混入。

**Q3** 是的，`mixed_values = (1-α)*z + α*soft` 的语义随 α 变化而变化。α=0 时是纯终局结果，α=1 时是纯材料差。curriculum 变化意味着模型在训练初期学"材料差估计"，后期学"终局预测"。这种目标漂移可能让优化器的动量方向失效。但这也可能是有益的——初期用 soft signal 加速 value head 冷启动，后期切换到更准确的终局信号。

**Q4** `tanh(2.0 × delta/18)` 的映射：delta=1 → 0.11, delta=2 → 0.22, delta=5 → 0.51, delta=9 → 0.76, delta=18 → 0.96。这种映射假设边际棋子价值递减（第 1 子差比第 18 子差更重要）。在实际六洲棋中，棋子数量差与胜率之间的关系取决于棋子位置和结构——3 个棋子如果都在方/洲中可能比 6 个分散棋子更有防守价值。`tanh` 映射忽略了这种位置依赖性。

**Q5** `_WDL_AUX_LOSS_WEIGHT = 0.0`（`train_bridge.py:24`）。打开它（设为 0.1-0.5）会增加一个交叉熵损失，要求 bucket 分布的 WDL 概率匹配从 raw value 推导的 WDL 分布。这可以改善 calibration（让 bucket 分布的形状更合理），但也可能与 bucket loss 冲突——bucket loss 优化的是精确的 bin 分布，WDL loss 优化的是粗粒度的三分类。两个目标的梯度方向可能不一致。

**Q6** `anti_draw_penalty=-0.1` 告诉 value head "平局 ≈ -0.1"。这会让 value head 对平局局面输出负值，MCTS 搜索会倾向避免平局——可能导致更激进的策略。风险：如果真正的最优策略确实是和棋（双方完美博弈），anti-draw penalty 会让模型追求不存在的胜利，导致实际棋力下降（冒险失败）。当前 `-0.00` 设置说明这个参数被谨慎对待。

**Q7** `policy_draw_weight=1.0` 意味着平局样本的策略信号完整保留。在 80%+ 平局的训练集中，policy loss 的 ~80% 梯度来自平局样本。降低到 0.5 会让胜负样本的权重翻倍（相对于平局），但也会减少总训练信号量。关键问题：平局样本的策略信号是否有用？如果两个差不多强的模型打出平局，其间的策略选择仍然包含"好的走法"信息。完全降权可能浪费这些信号。

**Q8** 正确。低温度下 MCTS target 非常尖锐。target entropy ≈ 0 时，KL ≈ CE = -log(pred[argmax(target)])。梯度集中在 argmax 动作上——模型只被要求"在这个局面选这一步"。这在高确信度局面是合理的，但可能让模型对 value head 误差过度敏感（如果 value head 在某步估计错误，MCTS 可能选错 argmax，模型被训练去强化这个错误）。

**Q9** `TensorTrajectoryBuffer` 初始化 `_value_targets` 为 NaN（`trajectory_buffer.py:71`）。`finalize_games_inplace` 通过 `v0_core.finalize_trajectory_inplace` 填入实际值。如果某些对局的 finalize 失败（例如 C++ 异常），NaN 会留在 buffer 中。`train_bridge.py:213-218` 的非有限值过滤会丢弃这些行。这是**安全网**——但也是**隐藏的数据丢失**。如果丢弃比例显著（>1%），应该在日志中报告 `filtered_non_finite_samples`（第 220 行的变量已经记录了这个值）。

**Q10** `grad_clip_norm=1.0` 对全模型统一裁剪。如果 value loss 梯度（101 维交叉熵）通常比 policy loss 梯度（220 维 KL）更大，裁剪后 policy head 的更新方向确实被 value head 主导。可以通过检查 `torch.nn.utils.clip_grad_norm_` 的返回值（裁剪前的原始 norm）来诊断：如果总是被裁剪，且 value head 梯度占 80%+，policy head 实际在被压制。分别裁剪需要把 optimizer 分成两组。

**Q11-32 要点总结：**

**Q11** 旧样本的 `π` 确实过时——它们来自更弱模型的搜索。但 `z` 仍然有效（终局结果是客观的）。可以只用旧样本训练 value head，降低其在 policy loss 中的权重。

**Q12** `epochs=3` 可能让模型在一轮训练中变化很大。off-policy 程度取决于 `lr` 和数据量——大 lr + 少数据 = 大变化 = 大分布不匹配。

**Q13** `warmup_steps=100` 在 `batch_size=8192` 下约 820k 样本。对于 300k 样本的 epoch，warmup 占满整个 epoch，第一个 epoch 基本在 warmup。这确实太慢——可以降低到 20-50。

**Q14** `loss = policy_loss + value_loss` 隐含假设两者量级相当且等重要。如果不是，应该加权重。

**Q15** `soft_label_alpha > 0` 确实改变了 value head 的学习目标——从"终局预测"转向"当前材料估计"。这是有意的：在训练初期 z 信号太弱（多数为 0），soft signal 提供了有用的替代目标。

**Q16** `finalize_trajectory_inplace` 用统一的终局结果回填所有步骤的 value target。过程中的优势变化确实被忽略。这是 AlphaZero 标准做法——(s, π, z) 中的 z 始终是终局结果，不是中间奖励。

**Q17** `policy_soft_only=True` 的设计意图：当 `soft_label_alpha > 0` 时，value target 已经混入了 soft signal（包含平局信息），此时再通过 `policy_draw_weight` 降权平局样本可能导致双重惩罚。`policy_soft_only=True` 时所有样本等权，避免重复处理。

**Q18** `LR_COSINE_FINAL_SCALE=0.5` 意味着最后一轮 lr 为初始值的 50%。在 60 轮中，后半段 lr 持续下降。如果棋力在第 30 轮已饱和，后续的 lr 下降只是在"精调"一个已经不再进步的模型——可能无害但也无益。动态 curriculum 需要一个"棋力是否仍在增长"的在线信号。

**Q19** DDP 的 rank 分片是 stride 式的（`rank::world_size`）。如果 replay window 的数据按迭代顺序拼接，stride 分片会让每个 rank 均匀采样到新旧数据。不会出现"某个 rank 只看旧数据"的问题。

**Q20** 消融方法：在 `opening_random_moves=0` 下重新训练，观察 `decisive_game_ratio` 变化。如果大幅下降，说明 `opening_random_moves` 是主要贡献者。

**Q21-32** 剩余问题的核心观点：bucket value 的 two-hot 编码在 z=0 时退化为 one-hot（Q21）；draw 类型应区分对待（Q32）；entropy bonus 缺失可能加速和棋收敛（Q26）；bucket loss 梯度可能消除 value head 的有用双峰信号（Q27）。这些都指向同一个核心挑战：**高平局率下的 value 信号稀疏性**。

## 5. 自博弈数据分布

**Q1** `concurrent_games=8192` 的数据独立性靠 Dirichlet noise（`alpha=0.3, epsilon=0.25`）和温度采样（`temperature=1.0` 前 10 步）维持。8192 局的策略**相同**（同一模型、同一参数），但每局的噪声实例不同。噪声在 prior 上按 25% 比例混合——对 15 个合法动作的局面，每个动作的 prior 偏移约 ±0.02。这种差异在前几步产生不同的开局，后续步骤因蝴蝶效应逐渐分化。8192 局之间的策略差异在初期较小但随步数增加而扩大。

**Q2** `step_index_matrix` 的交错不影响独立性——训练时 batch 从 buffer 中随机采样（`_train_permutation`），不按时间顺序读取。同一局的连续步骤在 buffer 中相邻，但训练 batch 通过 permutation 打散。

**Q3** `opening_random_moves` 创造的开局与真实强对局差距取决于随机步数。随机 5 步 → 棋盘有 5 个随机棋子，分布与强模型选择差异大。随机 0 步 → 完全由模型决定，可能集中在少数"最优"开局。curriculum 衰减速度应该根据模型强度调整——模型越强，越早停止随机。

**Q4** `max_game_plies=512` 是极端上限。典型对局在 `MAX_MOVE_COUNT=144` 或 `NO_CAPTURE_DRAW_LIMIT=36` 时结束。实际步数（含原子阶段）通常在 50-200 之间。512 的 buffer 预分配浪费了约 60-75% 的空间。但 GPU 内存通常不是瓶颈（预分配的是索引矩阵 `step_index_matrix`，每个元素 int64=8 bytes，8192×512 ≈ 33MB），可以接受。

**Q5** seed 为 `iteration_seed * 10007 + (worker_idx+1) * 9973`。不同 worker 使用不同 seed，保证噪声独立。跨迭代的 `iteration_seed` 不同保证了多样性。seed 的乘法常数（10007, 9973）是大素数，避免 seed 碰撞。

**Q6** `self_play_chunk_target_bytes=8GiB` 控制 chunk 分割。多 chunk 通过 manifest 管理。`train_bridge.py` 的 `train_network_streaming` 使用 DataLoader 可以流式处理多 chunk——每次只加载一个 batch 到 GPU。`train_network_from_tensors` 则需要将所有 chunk 拼接后一次加载。

**Q7** `decisive_game_ratio` 从 1.23% 到 81.72% 意味着有效梯度来源大幅增加。但如果 `policy_draw_weight=1.0`，所有样本（含平局）都贡献梯度——数据分布变化主要影响 value loss（更多 z≠0 的样本）。配套调整：如果 decisive 样本增多但棋力不升，可能需要增大 batch size 或调低 lr 以适应新的梯度方差。

**Q8** `replay_window=4` 混合 4 轮数据可以缓解单轮偏好问题。但如果连续 4 轮的模型都有相似偏好（例如都偏好某个开局），混合后偏好仍然存在。解决方案：增大 replay window 或引入对手池（不同 checkpoint 对战）。

**Q9** 高熵早期样本（`temperature=1.0`）的 policy target 分布更均匀。在训练时，这些样本的 policy loss 梯度分散在多个动作上，单个动作的梯度较小。低熵后期样本的梯度集中在少数动作上。如果 batch 中两类样本混合，低熵样本的梯度可能主导更新方向，高熵样本的信号被稀释。

**Q10** `anti_draw_penalty` 通过将平局 value target 设为负值来打破正反馈环——即使自博弈产生大量平局，模型被训练为"平局是不好的"，从而在下一轮自博弈中尝试避免平局。`policy_draw_weight < 1` 降低平局样本的策略权重，让模型更关注胜负样本中的策略差异。两者都是打破"平局正反馈环"的手段。

**Q11-20** 关键观点：NaN 样本只丢弃 value target，policy target 可以利用（Q11）；buffer 利用率约 25-40%（Q12）；不同 GPU 的对局长度差异被 merge 平均（Q13）；阶段混合的计算不均衡存在（Q14）；C++ 规则 bug 可能表现为 value loss 不降（Q15）；`num_simulations=1536` 已有边际收益递减迹象（Q16）；soft value 对平局的误导在 alpha 衰减时缓解（Q17）；process backend 的模型一致性靠共享 checkpoint path 保证（Q18）；对手池是值得探索的方向（Q19）；不同迭代样本的重要性加权是合理改进（Q20）。

## 6. 优化与训练稳定性

**Q1** 分离方法：（a）固定数据质量（使用同一批 selfplay 数据），只调 optimizer → 测试优化器效果；（b）固定优化器，只调 selfplay 参数（simulations, temperature）→ 测试数据质量效果；（c）固定两者，只调 value target 设计（bucket vs scalar, soft alpha）→ 测试目标设计效果。当前最可能的瓶颈：**数据质量**（由 decisive_game_ratio 反映）。

**Q2** loss 下降但 vs_previous 不升的最可能原因：**过拟合到旧数据**。`replay_window=4` 意味着 3/4 的数据来自更弱模型。loss 下降可能只是更好地拟合了旧分布，而新模型在面对更强对手时没有改善。次要可能：评估口径失真（评估条件与训练条件不匹配）。

**Q3** DDP 在 4 GPU、50k 样本下每卡约 12k 样本，batch_size=2048/卡，每 epoch 约 6 步。DDP 的 allreduce 开销约 2-5ms/步，6 步 = 12-30ms。总训练时间约 1-2s。通信占比 < 5%，不是瓶颈。`data_parallel` 在这种小规模下可能更简单但性能类似。

**Q4** value head 崩溃传播路径：value logits → softmax → bucket 分布 → 如果 logits 有 Inf → softmax 输出 NaN → `bucket_value_loss` = NaN → `loss = NaN` → `check_train_metrics_finite` 捕获。日志最早征兆：`value_loss` 突然跳变（从 ~0.1 变到 1000+），然后下一 batch 出现 NaN。policy loss 不受直接影响（独立计算），但 `mixed_values` 中如果 `value_targets` 本身包含 NaN（来自 finalize 失败），会通过 `scalar_to_bucket_twohot` 传播。

**Q5** PreAct 的 skip connection 传递未被 ReLU 截断的梯度——10 层后梯度仍可直接流回输入。增加到 20 层后 BN 统计可能不稳定（小 batch 的 BN 估计噪声大），但梯度流应该仍然健康。训练稳定性更可能受 BN 的 running stats 影响而非梯度消失。

**Q6** AMP 下 bucket loss 的 FP16 精度问题：softmax 输出接近 0 的 bin（远离 target）的 `log(prob)` 可能溢出。但 `GradScaler` 管理 loss scaling，避免了小梯度下溢。如果 value logits 的 range 很大（>100），FP16 softmax 可能产生 NaN。当前 `grad_clip_norm=1.0` 间接限制了 logits range 的增长。

**Q7** 8192×11×6×6 ≈ 12MB（float32），trunk 中间特征 8192×128×6×6 ≈ 150MB。H20 显存 96GB，完全不是瓶颈。带宽方面：H20 的 HBM 带宽约 4TB/s，150MB 的读写在 microsecond 级别。实际瓶颈更可能在计算（FLOPS）而非内存。

**Q8** `self_play_step_inplace` 产生非法状态（如棋子数为负）：`state_to_tensor` 不检查合法性，会生成包含负值的张量。`train_bridge.py` 的非有限值过滤检查 `torch.isfinite`——棋子数为负不会产生非有限值（int8 到 float32 的转换是精确的）。这类错误**不会被捕获**，会静默进入训练数据。

**Q9-20** 核心观点总结：统一 grad clip 可能压制 policy head（Q9）；旧数据稀释新信号（Q10）；gating 失败率是训练稳定性的指标（Q11）；optimizer state 恢复减少 warmup 浪费（Q12）；streaming 加载延迟可能成为瓶颈（Q13）；value head 参数更多但任务更难（Q14）；CUDAGraph on/off 的性能差异需要 A/B 测试（Q15）；3 epochs 在 300k 样本下可能不足（Q16）；curriculum 应根据实际强度动态调整（Q17）；非有限值→梯度爆炸→value collapse 是因果链而非独立症状（Q18）；108 次梯度更新可能太少（Q19）；分 optimizer 是值得探索的方向（Q20）。

## 7. 评估与选模

**Q1** vs_random 99.80% 时，0.20% 非胜局：查看 `EvaluationStats` 中的 `draws` vs `losses`。如果全是平局（`MAX_MOVE_COUNT` 或 `NO_CAPTURE_DRAW_LIMIT` 触发），说明模型偶尔无法在限时内结束对局。如果有败局（极少），可能是 Random 偶然走出了某种"陷阱"序列。需要具体日志才能分析。

**Q2** `wins > losses` 的 gating（`big_train_v1.sh:192+`）确实非常宽松。1胜0负99平就通过。这是因为在高平局率环境下，要求更多胜场会导致大量迭代被浪费。但"1胜0负"可能只是随机波动——需要更多对局才能确认统计显著性。改进方案：要求 `wins - losses >= 3` 或 `win_rate > 0.52`。

**Q3** 六洲棋中 White 先行（`FORCED_REMOVAL` 和 `MOVEMENT` 阶段）。50/50 color 分配确保每个模型扮演两种角色各半。如果先后手优势显著（如 White 胜率 55%），50/50 分配的总胜率 = 0.5×55% + 0.5×45% = 50%，颜色效应被对称化了。但如果要**测量**颜色效应，需要分别报告 Black 和 White 的胜率。

**Q4** `EvaluationStats` 只有 win/loss/draw，没有每局细节。如果需要区分"强力获胜"（10步内吃光对手）和"勉强获胜"（139步才分出胜负），需要扩展评估框架记录每局步数和终局棋子差。

**Q5** 锦标赛冠军的稳定性取决于分组随机性。`seed=20260226` 确定了分组。换 seed 可能改变冠军——尤其是当多个模型实力接近时。改进：多 seed 锦标赛 + Elo 排名更稳健。

**Q6** `MATCH_POINTS_WIN=3, DRAW=1` 确实偏好激进策略——一场胜利（3分）> 三场平局（3分）。"从不输棋但也不赢"的模型可能得分低于"偶尔赢也偶尔输"的模型。这是设计选择——鼓励争取胜利而非保守和棋。

**Q7-24** 核心要点：vs_random 和 vs_previous 测不同能力（Q7）；高温度锦标赛测平均水平（Q8）；Elo 在高平局率下不稳定（Q10）；gating 没有 vs_random 回归测试（Q12）；按阶段拆分胜率更有诊断价值（Q18）；跨迭代绝对强度衡量缺失（Q20）；opening book 可降低评估方差（Q22）；vs_self 可量化先后手优势（Q23）。

## 8. 系统设计与工程取舍

**Q1** v1 使用 `v0_core` 子集是经过深思的。`MCTSCore` 的完整树搜索在 GPU batch 模式下效率低（需要逐次模拟的串行循环），而 `root_pack_sparse_actions` + `root_puct_allocate_visits` 是专为 batch root-only 搜索设计的 GPU-native 操作。v1 跳过 `MCTSCore` 是性能驱动的选择，不是路径依赖。

**Q2** v1 没有复用 v0 MCTS wrapper 因为 root-only 语义无法适配 `MCTSCore` 接口——`MCTSCore.run_simulations` 是完整的选择-展开-评估-回传循环，而 v1 只需要一次展开+PUCT分配。接口不兼容是根因。

**Q3** `big_train_v1.sh` 保留 shell 的理由：进程编排（启动 torchrun、管理多个 Python 进程）在 shell 中更自然；错误处理和 exit code 管理在 shell 中更直接；curriculum 计算用 bash 数学足够。下沉到 Python 的收益：可单元测试 curriculum 函数、更好的类型检查。代价：需要在 Python 中实现进程编排。

**Q4** `train_entry.py` 的 v0/v1 不对称是历史遗留。v0 路径通过 subprocess 调用是因为 v0 训练循环（`train_loop.py`）有自己的完整主循环，不适合被 import。统一入口的价值主要在于简化 CLI——用户只需记住一个命令。

**Q5** `v0_core` 编译时间取决于机器和代码量。在 H20 上大约 2-5 分钟（包含 CUDA 编译）。`_soft_tan_from_board_black` 已经在 Python 端（`mcts_gpu.py:677-691`），不需要 C++ 编译。频繁修改的逻辑（如温度调度、curriculum）正确地保留在 Python/shell 端。

**Q6-20** 关键观点：文件 I/O 在 NVMe 上很快但网络存储可能慢（Q6）；buffer 有动态扩容机制 `_grow`（Q7）；`Design.md` 应在 docs 中引用（Q8）；测试用例需要验证覆盖率（Q9）；人机对战可能使用不同推理路径（Q10）；CUDA/CPU 双路径维护是风险点（Q12）；"不增加开关"原则已被 4 个 draw 参数挑战（Q20）。

## 9. 面向研究的问题

**Q1** `ChessNet` 约 1.2M 参数。6×6 棋盘的状态空间约 3^36 ≈ 1.5×10^17（上界），但有效状态空间（符合规则的）远小于此。1.2M 参数是否"Goldilocks"：从围棋 AlphaZero（~40M 参数，19×19 棋盘）按比例缩放，6×6 棋盘对应约 40M×(36/361) ≈ 4M 参数。当前 1.2M 可能偏小。但过大的模型在有限训练数据下可能过拟合。

**Q2** 显式注入 `is_piece_in_shape` 作为额外通道可以加速结构感知。C++ 端的 `encode_actions_fast` 已经计算了结构信息（在确定合法目标时调用 `is_piece_in_shape`），但没有把结果传入网络输入。增加 2 个通道（己方结构、对手结构）会将输入从 11 增加到 13，计算量增加约 18%，但可能显著加速策略学习。

**Q3** 高平局率是模型/搜索不够强**和**规则固有特性的组合。证据：`decisive_game_ratio` 从 1.23% 升到 81.72%——这说明更强的模型能产生更多胜负结局，意味着规则本身允许分出胜负。但 ~20% 的平局可能是规则固有的（双方完美博弈确实是和棋）。

**Q4-16** 核心研究方向建议：监督预训练 value head（Q4）；Q值作为策略目标（Q5）；reanalyse 可复用 `state_tensors`（Q6）；4×4 到 6×6 迁移需要检查 `BOARD_SIZE` 耦合点（Q7）；去掉 MCTS 做消融实验（Q8）；decisive_game_ratio 的归因分析需要消融（Q9）；宏动作可减少序列长度（Q10）；裸强度评估去掉训练 artifact（Q11）；Transformer 在 36 位置上计算量可接受（Q14）；项目最终证明的是"端到端形式化+训练闭环"的方法论（Q16）。

## 10. 给未来自己的追问

**Q1** root-only MCTS 的选择主要来自性能考量（GPU batch 效率），而不是实验证明 depth-1 足够。`MCTSCore` 的完整树搜索没有在 v1 pipeline 中做过 A/B 测试——因为接口不兼容。这确实是路径依赖。

**Q2** 消融方法：设 `opening_random_moves=0` 重跑训练，观察 `decisive_game_ratio`。如果大幅下降，说明 opening random 是主要驱动力。

**Q3** v1 的 25x-28x 吞吐提升来自 GPU tensor pipeline，但搜索质量从 v0 的完整树降到 v1 的 root-only。净效果需要在**相同训练时间**下比较棋力：如果 v1 在相同墙钟时间内产出更强模型，净效果为正。

**Q4** vs_random 99.80% 确实不能代表"模型很强"。Random Agent 不会防守、不会保护结构、不会避免被吃子。99.80% 只证明模型比随机好很多，不证明它能赢有策略的对手。

**Q5** `soft_value_k=2.0`、`dirichlet_alpha=0.3`、`exploration_weight=1.0`、`temperature_threshold=10` 这些值多数沿用了 AlphaZero/KataGo 的经验值，没有针对六洲棋做系统搜索。`soft_value_k` 可能例外——它是项目特有参数。

**Q6** 删掉一半复杂度的建议：删掉 `soft_value_targets`（及 `soft_label_alpha`、`soft_value_k`）。理由：soft value 是对 value head 冷启动的 workaround，如果通过更好的 warmup 或预训练解决冷启动，soft value 不再需要。bucketed value 是核心 value 表示，不应回退到标量。

**Q7** `replay_window=1` + 更多 epochs 是否更好：需要实验。risk：只有当前迭代数据，如果自博弈产出有偏，训练容易过拟合。replay window 提供了多样性缓冲。

**Q8** 更严格 gating（如 `wins - losses >= 5`）会增加浪费的迭代但可能长期更稳。当前宽松 gating 允许"噪声进步"混入——模型可能因为随机波动而通过 gating，而非真正变强。

**Q9** 核心问题选择建议："打破平局主导分布"更根本——它直接限制了 value head 的信号质量，间接限制了 MCTS 搜索质量。搜索效率问题在 value 信号改善后自然缓解。

**Q10** `v0_core` 的 API 边界不算冻结——修改 C++ 代码可以重新编译。但编译周期（2-5min）确实阻碍了快速实验。`root_puct_allocate_visits` 中加入 Progressive Widening 需要修改 CUDA 内核，风险较高。

**Q11** `log_p2 + log_p1` 分解确实是三头架构的限制。替代方案：单一 220 维 head（完全独立的动作概率）、或 from-to 注意力机制（`attention(from, to)` 生成 movement 概率）。需要消融实验评估。

**Q12** bucketed value 的沉没成本：`scalar_to_bucket_twohot`、`bucket_logits_to_scalar`、`bucket_value_loss`、`_bucket_logits_to_wdl_probs` 等约 200 行代码。如果回滚到标量 value，需要修改 `ValueHead`（输出 1 维而非 101 维）、训练 bridge（loss 改为 MSE）、evaluation（解码方式改变）。工程成本中等，但关键问题是：有没有数据证明 bucketed 比 scalar 更好？如果没有，沉没成本确实可能影响判断。

**Q13** 最值得保留的部分：**规则引擎的完整性与正确性**。`rule_engine.py` + `check_rule_engine_cases.py` 是项目独一无二的资产——它把一个民间棋类完整形式化并通过 1000+ 用例验证。训练 pipeline 可以重写，但正确的规则引擎是一切的基础。

**Q14** 4 个 draw 相关参数（`opening_random_moves`、`soft_label_alpha`、`lr`、`policy_draw_weight` 的 curriculum）确实超出了"简洁"的范畴。但它们分别解决不同问题：`opening_random_moves` 解决开局多样性、`soft_label_alpha` 解决 value 冷启动、`policy_draw_weight` 解决平局样本权重。简化方向：如果 decisive_game_ratio 自然提升到 90%+，`soft_label_alpha` 和 `policy_draw_weight` 可以去掉。

---

*本手册基于当前代码实现撰写，答案可能随实验结果和代码迭代而需要修订。*
