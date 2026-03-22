# 高难问题清单

我就想问问，让我也了解了解我的项目哈哈！

## 1. 规则与问题建模

1. 六洲棋的规则（7 个 `Phase`、方/洲触发、优先级约束）天然产生多阶段复合决策。为什么这种结构适合 AlphaZero 风格的 `(s, π, z)` 闭环，而不是优先做大规模监督学习或手写评估函数？
2. `Phase` 枚举把一个回合拆成 `PLACEMENT → MARK_SELECTION → REMOVAL → MOVEMENT → CAPTURE_SELECTION` 等原子阶段。这种拆分在让规则引擎更可验证的同时，是否把环境建模复杂性转移到了动作空间设计（`ActionEncodingSpec` 220 维）里？
3. 如果把 `MARK_SELECTION`（逐个标记）和 `CAPTURE_SELECTION`（逐个提吃）合并成一次性多选动作，训练会更容易还是更难？合并后 `pending_marks_remaining` 递减逻辑要如何重构？
4. `MARK_SELECTION`、`CAPTURE_SELECTION`、`FORCED_REMOVAL` 三个阶段的合法动作池大小差异很大：标记最多选全部对手棋，强制移除只能选非结构棋。这种动作密度不对称在 policy loss 里是否引入了学习难度的不对称？
5. `is_piece_in_shape` 同时检查 2×2 方和长度 6 洲。"存在普通棋时不能选结构棋"这条约束会不会让策略在某些局面产生不连续跳变——差一个普通棋存在与否，合法集完全不同？
6. 对模型来说，方（2×2）和洲（整行/列 6 连）是应当由 `ChessNet` 的 10 层 `PreActResBlock`（感受野覆盖 6×6）自己发现的特征，还是应当通过额外的结构通道显式注入？
7. `NO_CAPTURE_DRAW_LIMIT=36`（即走子阶段连续 36 步无吃子判和）和 `MAX_MOVE_COUNT=144` 这两条平局保护是工程需要还是任务定义的一部分？如果删掉它们，训练会收敛到真实博弈分布，还是陷入无限循环？
8. `moves_since_capture` 在 `apply_move` 中的归零逻辑是：走子/提吃阶段如果 opponent 棋子数减少就归零，否则累加。这意味着标记阶段的标记动作不影响计数器。这种设计是否低估了标记动作的"进攻价值"？
9. `LOSE_PIECE_THRESHOLD=4` 在哪里被使用？如果它只是一个未启用的常量，那它暗示了什么样的终局判定替代方案？
10. 六洲棋的长期信用分配难度，主要来自 7 阶段规则复杂度、`pending_marks/captures` 的跨阶段依赖，还是来自大量平局把 `z` 压成 0？
11. `PLACEMENT` 阶段不能落在被对手标记的位置上（`opponent_marked`）。这条规则意味着标记不仅是进攻手段，还能封锁对手落子空间。模型是否有足够的通道信息（ch2-3 编码双方标记位）来学到这种双重作用？
12. `FORCED_REMOVAL` 规定 White 先移除 Black 棋子、再 Black 移除 White 棋子。这个固定顺序是否引入了先后手不对称？如果改成随机先手或由棋子数决定先后，对训练目标有什么影响？
13. `COUNTER_REMOVAL`（`no_moves_remove` 后的反移除）在实际自博弈中触发频率有多高？如果它极少出现，模型在这个阶段的策略头输出是否有足够样本来学习？
14. `check_lines` 和 `check_squares` 在 `rule_engine.py` 中每次落子/走子后都被调用。当棋盘接近满员时，多个方/洲可能同时形成。当前实现如何处理同一步触发多个结构的情况？`pending_marks_required` 是累加还是取最大值？
15. 规则要求被标记棋不能参与形成新的方/洲。这条规则在 `check_lines` / `check_squares` 中通过排除 `marked_set` 实现。如果标记棋被移除后原位又被落子，新棋能否立刻参与结构？这个边界条件在测试用例 `check_rule_engine_cases.py` 中覆盖了吗？
16. `generate_mark_targets` 中的"优先普通棋"逻辑：当所有对手棋都在结构中时才允许选结构棋。这个回退条件在实际对局中何时触发？是否存在对手故意把所有棋子保持在结构中来限制标记选择的策略？
17. `REMOVAL` 阶段"同时移除双方所有被标记棋"。这种批量移除是否意味着双方标记棋的交互效应（例如一方标记棋被移除后另一方结构是否改变）不需要考虑？
18. 如果把 `BOARD_SIZE` 从 6 改为 8，哪些规则常量需要同步修改？`ActionEncodingSpec` 的 `placement_dim=36`、`movement_dim=144`、`selection_dim=36` 全部与 `6×6` 硬编码耦合，这种耦合是设计选择还是技术债？
19. `Phase.MOVEMENT` 阶段只允许上下左右移动一步到相邻空位。在棋盘后期空位稀疏时，大量棋子可能无法移动。`no_moves_remove` 机制是否足以应对这种僵局，还是说 `NO_CAPTURE_DRAW_LIMIT=36` 实际上在替 `no_moves_remove` 兜底？
20. 从 `docs/rules.md` 到 `src/rule_engine.py` 到 `v0/src/rules/rule_engine.cpp` 的三层实现必须保持语义一致。当前靠 `check_rule_engine_cases.py`（1000+ 用例）做对拍。如果规则发生微调（如改变方的触发条件），这三层的同步成本是否已经成为阻碍规则实验的主要障碍？

## 2. 状态表示与动作表示

1. `state_to_tensor` 的 11 通道中，ch0-3 编码己方棋子、对手棋子、己方标记、对手标记，ch4-10 是 7 个阶段的 one-hot。`moves_since_capture`、`pending_marks_remaining`、`pending_captures_remaining` 这些标量状态没有通道编码——模型是否有足够信息恢复或推断这些值？
2. 阶段 one-hot（ch4-10）直接告诉网络当前处于哪个 `Phase`。如果移除这 7 个通道，模型能否仅从棋子分布和标记位推断阶段？这样做会迫使网络学到更深层的规则理解，还是只会让训练更慢？
3. `PolicyHead` 通过 `GlobalPool`（对每个通道取 mean/max/std，拼成 3×64=192 维）生成全局上下文，再 broadcast-add 回空间特征。这种全局信息注入是策略质量的关键因素，还是可以用更简单的全局平均池化替代？
4. 三个策略头 `out_pos1`（目标位置）、`out_pos2`（来源位置）、`out_mark`（标记/提吃/移除位置）各输出 36 维 `log_softmax`。为什么不用一个统一的 220 维 flat head？三头设计的优势来自表达能力、优化稳定性，还是样本效率？
5. `build_combined_logits` 把三头输出拼成 220 维：placement[0..35] 取 `log_p1`，movement[36..179] 取 `log_p2[from] + log_p1[dest]`，selection[180..215] 取 `log_pmc`，auxiliary[216..219] 填 0。movement 的加性组合 `log_p2 + log_p1` 隐含假设来源概率与目标概率可分解——在什么棋局条件下这个假设最容易失败？
6. `ActionEncodingSpec` 的 `movement_dim=144`（36 位置 × 4 方向），但实际合法移动远少于 144（边界位置方向受限、目标位置需为空）。这种稀疏编码是否让 softmax 在 movement 区间产生大量"噪声概率"，从而挤压有效动作的学习信号？
7. `auxiliary_dim=4` 中只有 index 216 被 `process_removal` 使用，217-219 保留未用。这些保留位在 `build_combined_logits` 中填 0，在 `masked_log_softmax` 中被 mask 掉。它们的存在是否影响 softmax 的归一化行为？
8. `selection` 区间（index 180-215）同时服务于 `MARK_SELECTION`、`CAPTURE_SELECTION`、`FORCED_REMOVAL`、`COUNTER_REMOVAL` 四个阶段，靠 `Phase` 来消歧。这意味着模型在 selection 区间学到的是"位置偏好"的混合分布。不同阶段的目标语义是否足够接近来共享同一个 `out_mark` 头？
9. `masked_log_softmax` 用 `-inf` 遮盖非法位置，再对合法位置做 `logsumexp`，输出 clamp 到 -50。在合法动作只有 1-2 个时（如 `FORCED_REMOVAL` 后期），这种处理是否让 policy loss 退化为近似零梯度？
10. `placement`、`movement`、`selection`、`auxiliary` 四个动作区间的典型合法动作数差异巨大（placement 可达 30+，movement 可达 20+，selection 通常 < 10，auxiliary 最多 1）。`softmax` 归一化后，小区间的动作是否天然获得更高概率密度，从而让网络对某些阶段"过度自信"？
11. `ChessNet` 使用 `PreActResBlock`（BN→ReLU→Conv→BN→ReLU→Conv + skip）而非标准 ResBlock（Conv→BN→ReLU→Conv→BN + skip→ReLU）。在 6×6 棋盘上 10 个 block 的感受野已经完全覆盖全局，这种 pre-activation 选择的收益来自梯度流改善还是特征分布规范化？
12. 价值头从 trunk 出发经过 `Conv2d(128→64, 1×1) → BN → ReLU → GlobalPool → Linear(192→128) → ReLU → Linear(128→101)`，输出 101 个 bucket logits。`bucket_logits_to_scalar` 用 softmax 加权期望映射回 [-1,1]。这种 101-bin 离散化相比直接输出标量 `tanh`，在 value 接近 0（平局附近）时分辨力更强还是更弱？
13. `GlobalPool` 同时被 PolicyHead 和 ValueHead 使用，但 PolicyHead 将池化结果 broadcast-add 回空间维度，而 ValueHead 直接线性映射到标量。这两种用法对全局信息的需求一致吗？PolicyHead 是否需要比 ValueHead 更精细的空间-全局交互？
14. 当前 `trunk_channels=128` 对 6×6 棋盘是否过大？如果减半到 64 通道，policy head 和 value head 的质量会不成比例地下降吗？哪一个头对容量更敏感？
15. movement 编码使用 4 方向索引：`((-1,0), (1,0), (0,-1), (0,1))` → dir 0-3。如果棋盘扩展到支持对角移动，`movement_dim` 需要从 144 增长到 36×8=288，`build_combined_logits` 的拼接逻辑也要重写。这种编码方式与规则的耦合程度是否太紧？
16. `state_to_tensor` 中"己方"和"对手"是相对当前行动方的，每次调用时会根据 `current_player` 翻转视角。这意味着网络看到的永远是"我的棋子在 ch0"。这种对称化编码是否让模型无法学到先后手不对称的策略？
17. `encode_actions_fast` 在 CUDA 不可用时自动回退到 CPU 路径。回退后的 legal mask 结果与 CUDA 版本是否严格一致？如果存在浮点差异，是否会导致自博弈数据在不同硬件上不可复现？
18. `log_p2[from] + log_p1[dest]` 的加性分解意味着模型无法表达"从 A 到 B 好但从 A 到 C 差，同时从 D 到 C 好"的非分解偏好。在走子阶段形成方/洲时，来源-目标的联合概率是否需要这种非分解表达能力？
19. `policy_channels=64` 和 `value_channels=64` 相同。如果 policy head 比 value head 需要更多空间分辨率，是否应该给 policy head 更多通道？当前对称设计是刻意选择还是默认值？
20. `value_bucket_bins=101` 这个奇数使得 bin 中心恰好包含 0。如果用偶数（如 100），0 值将落在两个 bin 的边界上。在高平局率（大量 value ≈ 0）的六洲棋中，这个中心 bin 的存在对 calibration 有多重要？

## 3. MCTS 设计

1. `V1RootMCTS` 只展开根节点直接子节点（`root_pack_sparse_actions` → `batch_apply_moves_compat` → child eval），再用 `root_puct_allocate_visits` 分配虚拟访问。相比 `MCTSCore`（C++ 完整树搜索），这种 depth-1 搜索在 `num_simulations=1536` 时损失了多少搜索质量？
2. `root_puct_allocate_visits` 在一次调用中完成所有虚拟模拟的分配，而不是逐次模拟。这种批量分配与传统 PUCT 的逐次选择-展开-回传在结果上等价吗？在哪些分支因子条件下差异最大？
3. `child_eval_mode="value_only"` 时，`_evaluate_values_only` 只做 value 前向，跳过 policy projection。这意味着子节点没有 prior 信息。在 root-only 搜索中子节点 prior 不被使用，但如果未来扩展到 depth-2，这个路径需要如何修改？
4. `_soft_tan_from_board_black` 用 `tanh(soft_value_k * material_delta / 18)` 为终局子节点生成 soft value（`soft_value_k=2.0`）。为什么用棋子差比例而不是 value head 输出？在什么条件下两者会严重分歧？
5. PUCT 公式 `Q + exploration_weight * P * sqrt(N_parent) / (1 + N)` 中，`exploration_weight=1.0`。在六洲棋的分支因子（placement 阶段可达 30+，movement 阶段通常 10-20）下，这个值是偏向探索还是偏向利用？
6. `_terminal_mask_from_next_state` 检测子节点是否为终局。如果一步之内对手就输了，value 用 `_soft_tan_from_board_black` 而不是硬编码 ±1。这种 soft terminal value 是否会让 MCTS 低估必胜局面的价值？
7. Dirichlet noise 在根节点 prior 上按 `(1-0.25)*P + 0.25*noise` 混合，`alpha=0.3`。`alpha=0.3` 对于平均合法动作数 ~15 的六洲棋是否合适？AlphaZero 对围棋用 0.03、对国际象棋用 0.3——六洲棋的分支因子更接近哪个？
8. 温度调度在 `self_play_v1_gpu` 中实现：前 `temperature_threshold=10` 步用 `temperature_init=1.0`，之后用 `temperature_final=0.1`。10 步对应的游戏阶段通常在 `PLACEMENT` 中后期。这个阈值是基于什么假设——开局多样性在落子阶段最重要？
9. `_root_finalize_from_visits` 将访问计数转成策略目标。`_stable_legal_policy_from_visits` 在 log 空间操作以避免低温度下的溢出。当 `temperature=0.1` 时，访问计数的微小差异会被放大 10 倍——这是否让策略目标过于尖锐，导致 policy loss 梯度集中在少数动作上？
10. `force_uniform_random_mask` 在 `opening_random_moves` 步内强制均匀分布。这些步的 policy target 也是均匀分布。这些均匀样本进入训练后，是在增加开局多样性，还是在让 policy head 学习一个无用的均匀先验？
11. CUDAGraph 缓存的 key 是 `(device_idx, num_roots, max_actions, batch_size, sample_moves)`。当 `max_actions` 在不同 batch 之间变化时，会创建新的 graph。在自博弈过程中，`max_actions` 的变化频率有多高？频繁重建 graph 是否抵消了 CUDAGraph 的加速收益？
12. `V1RootMCTS` 中 `_child_values_to_parent_perspective` 在 side-to-move 变化时翻转 value 符号。在 `MARK_SELECTION` → `MARK_SELECTION`（同一玩家连续标记）时，side-to-move 不变，value 不翻转。这种处理是否正确——连续标记的子节点 value 应该从谁的视角计算？
13. `root_pack_sparse_actions` 输出 `legal_index_mat`、`priors_mat`、`action_code_mat`、`valid_mask` 等稀疏矩阵。当某个根节点只有 1 个合法动作时，MCTS 搜索是否退化为直接选择？这种情况下 `num_simulations` 次模拟是否全部浪费？
14. 在 `MCTSCore`（C++）中，`AdvanceRoot` 用 `CompactTree` 保留子树。`V1RootMCTS` 没有这个能力——每次 `search_batch` 都从头建树。在同一对局的连续步之间，是否存在大量可复用但被丢弃的搜索信息？
15. `EvalBatcher` 在 C++ 端通过 `timeout_ms=2` 积攒请求再批量推理。`V1RootMCTS` 不使用 `EvalBatcher`，而是直接在 Python 端做批量前向。如果将 `EvalBatcher` 的异步批处理引入 v1，是否能进一步降低推理延迟？
16. `V1_FINALIZE_GRAPH` 环境变量控制 `_root_finalize_from_visits` 是否使用 CUDAGraph。当 `CUDA_LAUNCH_BLOCKING=1` 或在 nsys 下自动禁用。这意味着性能分析时看到的时间特征与实际运行不同。如何设计基准测试来准确衡量 CUDAGraph 的真实收益？
17. `exploration_weight` 在整个对局中保持不变（1.0）。是否应该在对局后期（movement 阶段，棋子少、分支因子小）降低探索权重，以获得更精确的策略目标？
18. `batch_apply_moves_compat` 批量应用所有合法子动作来生成子状态。当根节点有 30+ 合法动作时，这意味着一次性前向 30+ 个子状态。这种"全展开"策略在显存方面的开销是否线性增长？在 `self_play_concurrent_games=8192` 时是否会成为瓶颈？
19. `V1RootMCTS` 对终局子节点使用 `_soft_tan_from_board_black`（材料差比例），而非直接用 ±1。当对局以 `NO_CAPTURE_DRAW_LIMIT` 触发平局结束时，soft value 基于材料差，但实际 `z=0`。这两个信号是否矛盾？
20. `sample_moves=True` 时，`_root_finalize_from_visits` 按温度缩放后的概率采样动作；`sample_moves=False` 时取 argmax。评估时默认 `sample_moves=False`（deterministic），但锦标赛用 `sample_moves=True`。这两种模式测量的是同一种"棋力"吗？
21. `num_simulations` 在 stable profile 中是 1536，在 aggressive profile 中是 131072。这近 100 倍的差距会不会让两个 profile 产生的策略目标在质量上完全不可比？
22. PUCT 中 `sqrt(N_parent)` 随模拟次数增长。当 `num_simulations` 从 128 增加到 1536，探索项增长约 3.5 倍。`exploration_weight=1.0` 是否在高模拟次数下导致过度探索？
23. 在 root-only 搜索中，所有子节点的 value 估计都来自 value head 的单次前向。如果 value head 对某类局面系统性偏差（如总是高估先手），MCTS 的所有子节点都会继承这个偏差。root-only 搜索是否比完整树搜索更容易被 value head 的系统性误差误导？
24. `_evaluate_batch` 中 policy projection 使用 `v0_core.project_policy_logits_fast`，将三头 logits 映射到 220 维。这个 projection 是否包含了 `legal_mask` 信息？如果 prior 在 mask 前就已经分配了概率给非法动作，PUCT 的先验质量如何保证？
25. 当同一步可以形成方和洲（同时触发两个结构），`pending_captures` 会累加。MCTS 在评估这步走棋的子节点时，子节点处于 `CAPTURE_SELECTION` 阶段。但 root-only 搜索只看一层——它看到的是提吃后的局面还是提吃选择前的局面？
26. `V1RootMCTS` 在 `search_batch` 中每次都重新构建整个搜索结构。如果同一个局面在不同对局中重复出现（例如对称开局），是否应该引入 transposition table 来复用先前搜索结果？
27. `dirichlet_epsilon=0.25` 意味着 25% 的 prior 来自噪声。在合法动作只有 2-3 个的局面中，噪声占比是否过高，导致搜索效率大幅下降？代码中有 `max_actions > 1` 的检查但没有根据动作数调整 epsilon。
28. `_child_values_to_parent_perspective` 的翻转逻辑依赖于 `side_to_move` 是否改变。但 `REMOVAL` 阶段后总是 White 先行——这意味着 Black 的 REMOVAL 子节点 value 需要翻转而 White 的不需要。这个边界条件是否被正确处理？
29. root-only MCTS 的策略目标质量与完整 MCTS 的差距，是否随训练轮次增加而收窄（因为 value head 更准确），还是随着模型变强差距反而扩大（因为更深搜索的边际收益更大）？
30. 如果将 `num_simulations` 动态调整为与合法动作数成正比（动作多时搜索更多），是否能在总计算量不变的情况下提高策略目标质量？当前实现每个根节点使用固定 `num_simulations`。

## 4. 强化学习目标

1. 训练样本 `(s, π, z)` 中 `π` 来自 MCTS 访问计数、`z` 来自对局终局结果。当绝大多数对局是平局（`z=0`）时，value head 学到的是"局面确实无差别"，还是"信号不足导致的输出塌缩"？如何从 `bucket_logits_to_scalar` 的输出分布区分这两种情况？
2. `scalar_to_bucket_twohot` 把 value 目标离散化到 101 个 bin 的 two-hot 编码。当 `z=0` 时，target 集中在中心 bin。101 bin 在 [-1,1] 均匀分布意味着每个 bin 宽度约 0.02。这个精度对于区分"微弱优势"和"平局"是否足够？
3. `mixed_values = (1-soft_label_alpha) * value_targets + soft_label_alpha * soft_value_targets` 将硬终局结果和 soft material value 混合。当 `soft_label_alpha` 通过 curriculum 从某个初始值变化时，训练目标的语义也在变——模型是否被迫不断适应变化的 target 分布？
4. `_soft_tan_from_board_black` 用 `tanh(2.0 * material_delta / 18)` 生成 soft value。这意味着 1 子优势 → soft value ≈ 0.11，9 子优势 → soft value ≈ 0.76。这种非线性映射是否准确反映了棋子数量差与胜率之间的关系？
5. `_WDL_AUX_LOSS_WEIGHT = 0.0`，即 WDL 辅助损失当前被关闭。`scalar_to_wdl` 和 `_bucket_logits_to_wdl_probs` 的代码已经存在但不参与训练。打开它会引入什么——更好的 value calibration，还是一个与 bucket loss 冲突的额外梯度？
6. `anti_draw_penalty` 在 `train_bridge.py` 中将平局样本的 value target 从 0 替换为指定值。当 `ANTI_DRAW_PENALTY=-0.00` 时实际无效。如果设为 -0.1，等于告诉模型"平局是微弱的失败"——这是在塑造更强策略，还是在制造虚假的 value 偏差？
7. `policy_draw_weight` 控制平局样本在 `batched_policy_loss` 中的权重。默认 1.0 意味着平局样本与胜负样本等权。在平局占 80%+ 的训练集中，这是否让 policy loss 被平局样本主导？降低到 0.5 会让模型更关注胜负样本的策略差异吗？
8. `batched_policy_loss` 计算 KL(target || pred) = CE − H(target)。当 MCTS target `π` 非常尖锐（低温度下单一动作占 90%+），target 熵 H(π) 接近 0，loss 退化为 cross-entropy。这种情况下 policy loss 的梯度是否过于集中在最高概率动作上？
9. `train_network_from_tensors` 中的非有限值过滤：如果 `value_targets`、`soft_targets`、`policy_targets` 或 states 中有任何 NaN/Inf，整行被丢弃。`TensorTrajectoryBuffer` 初始化 value targets 为 NaN，只在 `finalize_games_inplace` 后填入实际值。如果某些对局的 finalize 失败，这些 NaN 样本会被静默丢弃——这是安全网还是隐藏的数据丢失？
10. `grad_clip_norm=1.0`。在 policy loss 和 value loss 量级不同时，梯度裁剪是否对其中一个 loss 的梯度更具约束力？如果 value loss 梯度始终更大，policy head 是否实际上在被压制？
11. `replay_window=4` 意味着训练使用当前迭代加前 3 次迭代的自博弈数据。旧数据由更弱的模型生成。当模型快速进步时，这些旧样本的策略目标 `π` 是否已经过时？旧样本的 value target `z` 仍然有效（终局结果不变），但 `π` 是否应该被降权？
12. staged pipeline 中 selfplay 和 train 是分开的进程。selfplay 用 iteration N 的模型生成数据，train 用这些数据更新模型。如果一次 train 阶段的 `epochs=3` 让模型变化很大，模型与数据之间的分布不匹配有多严重？
13. `warmup_steps=100`。在 `batch_size=8192` 下，100 步 warmup 意味着模型在约 820k 样本后才达到目标学习率。对于 `self_play_games=12288`（stable profile，约 300k-500k 样本），warmup 几乎占满了整个 epoch。这是否太慢？
14. value loss 使用 bucket 分布的负对数似然，policy loss 使用 KL 散度。两者的量级是否自然匹配？当前实现直接相加 `loss = policy_loss + value_loss`，没有额外权重。这隐含了什么假设？
15. 当 `soft_label_alpha > 0` 时，value target 不再是纯粹的终局结果，而是混入了材料差信号。这是否改变了 value head 学习的目标语义——从"这个局面最终会赢/输/平"变成"这个局面目前的材料状况如何"？
16. `finalize_trajectory_inplace` 在 C++ 中实现，用 `player_signs`（Black=1, White=-1）做视角转换：`value_targets[i] = signs[i] * result_from_black`。如果 `result_from_black=0`（平局），所有步骤的 value target 都是 0，无论棋局过程中的优势变化。这是否浪费了过程中的信号？
17. `policy_soft_only=True` 时 policy loss 不使用 draw weighting，所有样本等权。这个选项的设计意图是什么——当 soft value 已经蕴含了平局信息时，再额外降权平局样本会导致双重惩罚？
18. `LR_COSINE_FINAL_SCALE=0.5` 意味着学习率在最后一个迭代降到初始值的一半。在 60 次迭代的 stable profile 中，后 30 次迭代的学习率都低于初始值的 75%。这种 cosine 退火是在帮助收敛，还是在过早限制模型的学习能力？
19. DDP 训练时，样本按 rank 分片。不同 rank 看到不同的数据子集。如果 replay window 中的旧数据和新数据在 rank 间分布不均匀，是否会导致不同 GPU 上的梯度方向不一致？
20. 在高平局率环境下，`decisive_game_ratio` 从 1.23% 提升到 81.72% 是项目里程碑。但这个提升有多少来自 `opening_random_moves` 创造的强制多样性，有多少来自模型真正学到了获胜策略？
21. `scalar_to_bucket_twohot` 产生的 two-hot 编码对于 value 恰好在 bin 中心的样本，退化为 one-hot。对于 `z=0` 的平局样本，target 始终是 bin 50 的 one-hot。这是否让 value head 对中心 bin 过拟合？
22. 如果引入 `value_target = z * (1 - draw_discount) + draw_discount * soft_value` 的混合目标，其中 `draw_discount` 只对平局样本生效，是否比当前的 `mixed_values` 全局混合更精确？
23. `anti_draw_penalty` 和 `policy_draw_weight` 都是处理平局的机制，但一个作用在 value target 上，一个作用在 policy loss 权重上。它们是否可能产生方向冲突——例如 `anti_draw_penalty` 告诉 value head 平局是负面的，但 `policy_draw_weight=1.0` 仍然让平局样本的策略信号完整传播？
24. 当前 `weight_decay=1e-4`（stable）或 `5e-5`（aggressive）。在 AlphaZero 风格的自博弈训练中，weight decay 的作用是防止过拟合还是维持探索能力？如果模型每轮都在新数据上训练，传统意义上的过拟合是否仍然是主要风险？
25. `batched_policy_loss` 中 draw 检测使用 `raw_batch_values.abs().lt(1e-8)`。这意味着 soft value 接近 0 但不完全为 0 的样本（例如材料差为 ±1 子的平局）不会被标记为 draw。这种阈值分类是否遗漏了大量"实质平局"样本？
26. 训练目标中没有 entropy bonus。AlphaGo Zero 和 KataGo 通过不同方式鼓励策略多样性。在六洲棋的高平局率环境下，缺少 entropy 正则化是否加速了策略向和棋偏好收敛？
27. `bucket_value_loss` 的梯度方向取决于 target bin 和预测分布的距离。当 target 是 `z=0`（bin 50）但预测分布双峰（倾向胜或负）时，梯度会把两个峰同时拉向中心。这是否有助于 calibration，还是在消除 value head 已经学到的有用区分能力？
28. 训练数据的 `state_tensors` 是 `(N, 11, 6, 6)` float32。每个样本约 1.6KB。`self_play_games=12288` 典型产出 300k-500k 样本，总量约 500MB-800MB。`REPLAY_WINDOW=4` 意味着训练集最大约 2-3GB。这个规模是否太小，以至于模型在 `epochs=3` 内就能记住所有样本？
29. `soft_value_targets` 通过 `soft_label_alpha` 混入训练。在自博弈早期（模型弱），soft value 基于材料差，可能比随机 `z` 更有信息。在自博弈后期（模型强），`z` 的信号质量提升，soft value 反而引入偏差。curriculum 是否应该让 `soft_label_alpha` 从高值衰减到 0？
30. 如果 policy loss 和 value loss 使用不同的学习率（例如 policy 更大、value 更小），是否能缓解 value head 学习滞后于 policy head 的问题？当前实现使用统一的 optimizer 和学习率。
31. `streaming_load=True` 使用 DataLoader 流式加载大规模样本。在 DDP 下，DistributedSampler 保证每个 rank 看到不同分片。如果流式加载的顺序与完整加载不同，是否会改变训练的收敛行为？
32. "和棋"在训练数据中是单一类别（`z=0`），但实际可能混合了三种完全不同的情形：`MAX_MOVE_COUNT` 触发（真正的胶着长局）、`NO_CAPTURE_DRAW_LIMIT` 触发（走子阶段僵持）、以及双方棋力相当的自然平衡。这三种平局是否应该有不同的 value target 或 policy weight？

## 5. 自博弈数据分布

1. `self_play_v1_gpu` 使用 wave-based 批处理：`wave_size = min(concurrent_games, num_games)` 个对局同时进行，共享同一个模型做前向推理。这些并行对局的策略相同（来自同一模型），数据独立性靠 Dirichlet noise 和温度采样维持。在 `concurrent_games=8192` 时，这 8192 局之间的策略差异足够吗？
2. `step_index_matrix[game_slot, step_position]` 把每个对局的每一步映射到 `TensorTrajectoryBuffer` 中的全局索引。当一局结束、另一局还在进行时，buffer 中的样本在时间上是交错的。这种交错是否影响训练时 batch 内样本的独立性？
3. `opening_random_moves` 通过 `force_uniform_random_mask` 强制前 N 步均匀随机。curriculum 将其从高值线性衰减到低值。这些随机开局创造的数据分布与真实强对局的分布差距有多大？衰减速度是否足够快，以避免后期训练被劣质开局污染？
4. `self_play_concurrent_games=8192` 意味着 8192 局在同一 GPU 上同时进行。每局最多 `max_game_plies=512` 步。单次 wave 的峰值样本数为 8192×512 ≈ 420 万。但实际对局长度远短于 512。典型对局长度是多少？这个 `max_game_plies` 上限是否在浪费 buffer 预分配？
5. process backend 下，`run_self_play_worker` 在独立进程中运行，seed 为 `iteration_seed * 10007 + (worker_idx + 1) * 9973`。这个 seed 决定了 Dirichlet noise 和温度采样的随机性。不同 worker 之间的数据是否足够不相关？seed 的选择是否保证了跨迭代的多样性？
6. `self_play_chunk_target_bytes=8589934592`（8 GiB）控制单个 chunk 文件的大小。chunk 分割发生在样本边界上。当一次 selfplay 产出远大于 8GiB 时，会产生多个 chunk 文件，由 manifest 管理。train 阶段加载时是否需要将所有 chunk 拼接？streaming 加载能否真正流式处理多 chunk？
7. `decisive_game_ratio` 衡量有胜负结果的对局比例。当这个值从 1.23% 升到 81.72%，policy loss 的有效梯度来源也从 1.23% 升到 81.72%（假设 `policy_draw_weight=1.0`）。这种数据分布的剧变是否需要配套调整学习率或 batch size？
8. 同一轮自博弈的所有对局使用同一个 checkpoint。如果这个 checkpoint 有某种系统性偏好（例如总是倾向某个开局模式），整轮数据都会继承这个偏好。`replay_window=4` 混合多轮数据能缓解这个问题吗？还是说 4 轮窗口仍然太小？
9. `temperature_init=1.0` 在前 10 步维持。在 placement 阶段，合法动作数从 35 逐步下降。温度 1.0 意味着策略目标保持 MCTS 原始访问分布，没有锐化。这些高熵的早期样本是否被后续训练有效利用，还是被低熵的后期样本淹没？
10. 自博弈闭环的核心风险：模型 A 生成数据 → 训练出模型 B → 模型 B 生成新数据。如果模型 A 偏好和棋，它生成的数据中 `z=0` 占主导，模型 B 从这些数据中学到的也是和棋偏好。`anti_draw_penalty` 和 `policy_draw_weight` 是否足以打破这个正反馈环？
11. `TensorSelfPlayBatch` 包含 `state_tensors`、`legal_masks`、`policy_targets`、`value_targets`、`soft_value_targets` 五个张量。batch 保存为 `.pt` 文件。如果 selfplay 阶段崩溃导致部分对局未 finalize，`value_targets` 中的 NaN 会被 train 阶段的非有限值过滤丢弃。但 `policy_targets` 是完好的——是否应该利用这些"只有策略目标没有价值目标"的样本？
12. wave 内所有对局完成后，`TensorTrajectoryBuffer.build()` 用 `clone()` 提取有效切片。如果某些对局异常长（接近 `max_game_plies=512`），buffer 的利用率如何？是否存在大量预分配但未使用的空间？
13. `_merge_self_play_stats` 在 process backend 下合并各 worker 的统计信息。如果不同 GPU 上的对局长度分布显著不同（例如 GPU 0 的对局平均更短），合并后的 `decisive_game_ratio` 是否能准确反映整体数据质量？
14. 在 `concurrent_games=8192` 下，每一步都需要对 8192 个状态做 MCTS 搜索（model forward + PUCT allocation）。如果其中 90% 的对局已经处于 MOVEMENT 阶段但 10% 仍在 PLACEMENT，这两种阶段的计算开销是否平衡？`active_idx = torch.where(~done)[0]` 只过滤已结束的对局，不区分阶段。
15. 自博弈过程中 `v0_core.self_play_step_inplace` 原地更新状态。如果这个 C++ 函数存在微妙的规则 bug（例如某个边界条件处理不同于 Python 参考实现），产出的数据会系统性包含错误状态。这种错误在训练中可能表现为什么症状——value loss 不降、策略震荡、还是完全隐蔽？
16. 在高算力条件下（4×H20），是增加 `self_play_games`（更多对局）还是增加 `mcts_simulations`（每局更深搜索）对模型提升更大？`num_simulations=1536` 已经远超典型 AlphaZero 的 800。继续翻倍到 3072 的边际收益是否递减？
17. `soft_value_from_black` 在 `finalize_games_inplace` 中被传入但仅当 `soft_label_alpha > 0` 时生效。如果一局以 `MAX_MOVE_COUNT` 平局结束，`soft_value_from_black` 基于最终材料差。但如果此前一方长期领先最终被追平，这个 soft value 是否误导性地暗示"几乎赢了"？
18. `_resolve_self_play_backend` 在多 CUDA + Linux 时选择 process backend。process backend 为每个 GPU 启动独立进程，避免 GIL 竞争。但进程间的模型权重通过文件共享（checkpoint path），不是内存共享。如果迭代间模型更新不及时，不同进程是否可能使用不同版本的模型？
19. 如果在自博弈中引入"对手池"（同时使用最近 K 个 checkpoint 作为对手），而不是纯粹的 self-play（自己打自己），数据分布会变得更丰富吗？当前实现没有对手池机制，所有对局都是模型 A vs 模型 A。
20. `big_train_v1.sh` 的 curriculum 中 `opening_random_moves` 从高到低线性衰减。但这意味着早期迭代的数据（开局更随机）与后期迭代的数据（开局更确定性）分布差异很大。`replay_window=4` 混合它们时，是否需要对不同迭代的样本做重要性加权？

## 6. 优化与训练稳定性

1. 从 `logs/` 中的训练日志看，当前真正限制棋力增长的瓶颈是自博弈数据质量（`decisive_game_ratio`）、value head calibration（bucket 分布是否有意义）、还是优化器动态（梯度方向是否一致）？如何设计实验来分离这三个因素？
2. `LR` 从 `2.5e-4`（stable）经过 cosine schedule 衰减到 `1.25e-4`。如果学习率降低但 loss 继续下降，而 vs_previous 胜率不上升，最可能的原因是什么——过拟合到旧数据、value-policy 目标冲突、还是评估口径失真？
3. DDP 训练在 4 GPU 上将 `batch_size=8192` 分为每卡 2048。如果训练数据只有 50k 样本（短迭代），每卡每 epoch 只有 ~24 个 step。这么少的更新步数是否让 DDP 的通信开销占比过高？`train_strategy="data_parallel"` 在这种情况下是否更高效？
4. `check_train_metrics_finite` 在 `big_train_v1.sh` 中检查训练输出是否包含 NaN/Inf。如果 value head 先崩（输出 logits 爆炸），再通过 `mixed_values` 传播到 policy loss 的 draw weighting，日志上最早会出现什么征兆——value loss 突变、policy loss 突变、还是 gradient norm 突变？
5. `PreActResBlock` 的梯度流特性（BN→ReLU 在 conv 前）使得 skip connection 传递的是未被 ReLU 截断的梯度。在 10 个 block 的深度下，这是否有效缓解了梯度消失？如果从 10 block 增加到 20 block，训练稳定性会如何变化？
6. AMP（自动混合精度）在 CUDA 训练时默认开启。`GradScaler` 管理 loss scaling。在 `bucket_value_loss`（101-bin 交叉熵）中，当某些 bin 的 softmax 输出接近 0 时，FP16 的精度是否足够？是否有静默的精度丢失导致 value head 训练不充分？
7. `self_play_concurrent_games=8192` 意味着单次 model forward 的 batch 维度为 8192。在 H20 上，8192×11×6×6 的输入张量约 12MB（float32），10 层 ResBlock 的中间特征约 8192×128×6×6 ≈ 150MB。这个规模是否接近 H20 的显存带宽瓶颈？
8. `v0_core.self_play_step_inplace` 是 C++ 实现的原地状态更新。如果这个函数在某些边界条件下产生非法状态（例如棋子数变为负数），下游的 `state_to_tensor` 会生成什么样的输入？`train_bridge.py` 的非有限值过滤能否捕获这类错误？
9. `grad_clip_norm=1.0` 对 policy head 和 value head 使用统一的裁剪阈值。如果 value head 的梯度范数通常是 policy head 的 5 倍，裁剪后 policy head 的更新方向是否被 value head 主导？是否应该分别裁剪？
10. `REPLAY_WINDOW=4` 下，训练数据包含 4 轮自博弈数据。旧数据的 policy target 来自更弱的模型。如果训练优化器对旧 target 过度适配，新数据带来的改进信号是否会被稀释？这是否是"棋力不升但 loss 下降"的一个可能原因？
11. `big_train_v1.sh` 在每轮迭代结束后调用 `gating_accept_candidate`。如果新模型未通过 gating（`wins <= losses`），训练回退到上一个 checkpoint。这意味着该轮的自博弈数据和训练计算全部浪费。在 60 次迭代中，典型的 gating 通过率是多少？频繁失败是否暗示训练不稳定？
12. `optimizer_state_path` 允许跨迭代恢复优化器状态（momentum、variance 等）。如果不恢复，每轮训练的 Adam 状态从零开始，前 `warmup_steps=100` 步需要重新积累动量。恢复与不恢复的训练稳定性差异有多大？
13. `streaming_load=True` 在 DDP 下使用 `DistributedSampler`。如果数据文件很大（多个 8GiB chunk），加载延迟是否成为训练循环的瓶颈？是否需要 prefetch 或 pipeline 加载策略？
14. value head 输出 101 个 logits，policy head 输出 3×36=108 个 logits。参数量上 value head（`Linear(192→128→101)`）约 25k 参数，policy head（3 个 `Conv2d(64→1)` + `Linear(192→64)`）约 15k 参数。value head 参数更多但学习任务可能更难（校准概率分布 vs 排序动作）。这种不对称是否合理？
15. `V1_FINALIZE_GRAPH=off` 在 process backend 下被强制设置以保证稳定性。这意味着 CUDAGraph 只在 thread backend 中启用。两种 backend 的性能差异有多大？稳定性代价是否值得？
16. 当 `epochs=3` 且样本量 300k 时，每个样本被看到 3 次。对于自博弈 RL 来说，3 次是否太少（信号未充分提取）还是太多（过拟合到当前分布）？如果改为 1 epoch + 更大 replay window，效果会更好吗？
17. `compute_curriculum_values` 通过 `progress = (it-1)/(ITERATIONS-1)` 线性或 cosine 调度参数。如果训练在第 30 轮已经棋力饱和，后 30 轮的 curriculum 变化是否仍有意义？是否应该根据实际强度指标动态调整 curriculum？
18. 非有限值、梯度爆炸、value collapse（所有局面输出同一 value）、policy entropy 异常下降——这四个症状中，哪些是根因、哪些是连锁反应？如果 value head 先崩（logits 爆炸），它会通过 `mixed_values` → `scalar_to_bucket_twohot` → `bucket_value_loss` 的路径如何传播到 policy head？
19. `batch_size=8192` 在 stable profile 中使用。在只有 300k 样本的情况下，一个 epoch 只有 ~36 步。这意味着整个训练（3 epochs）只有约 108 次梯度更新。这个更新次数是否太少以至于训练噪声主导了学习方向？
20. 如果将 policy loss 和 value loss 分别用不同的 optimizer 训练（例如 policy 用 SGD + 大学习率，value 用 Adam + 小学习率），是否能让两个 head 以各自最优的速度学习？当前 `Adam(lr=2.5e-4)` 是否对 policy head 太慢、对 value head 太快？

## 7. 评估与选模

1. `vs_random` 评估使用 `RandomAgent` 对手，默认 deterministic（`sample_moves=False`）。当模型达到 99.80% 胜率时，剩余 0.20% 的非胜局是平局还是败局？这些非胜局是否集中在特定的开局序列或阶段？
2. `vs_previous` 评估使用上一轮最佳 checkpoint 作为对手。gating 条件是 `wins > losses`——没有最小胜场要求。如果 100 局评估中 1 胜 0 负 99 平，新模型就会被接受。这种宽松条件是否足以确保模型在真正进步？
3. 评估中的 color balance：前半数对局 challenger 执 Black，后半数执 White。六洲棋 `FORCED_REMOVAL` 阶段 White 先移除，`MOVEMENT` 阶段也是 White 先行。这种先后手不对称是否意味着 50/50 的 color 分配不足以消除颜色偏差？
4. `EvaluationStats` 只记录 `wins`、`losses`、`draws`、`total_games`。没有记录每局的具体步数、终局阶段、或棋子差。如果需要区分"强力获胜"和"勉强获胜"，当前评估统计是否提供了足够的信息？
5. 锦标赛使用 5 阶段淘汰（`STAGE_PLAN`：80→32→16→8→4→1），每组循环赛制，`GAMES_PER_MATCH=1000`。冠军 `model_iter_032.pt` 可能在某些阶段因为分组运气而存活。如果换一个 `seed`（改变分组），冠军是否会改变？排名的稳定性有多高？
6. `_ranking_key` 的优先级是 `match_points → game_win_rate → wins_minus_losses → -seed_order`。`match_points` 用 `WIN=3, DRAW=1, LOSS=0`。一个 2 胜 1 负（6 分）的模型排在 1 胜 0 负 2 平（5 分）的模型前面。但第二个模型从未输过——它可能更"稳定"。当前排名系统是否偏好激进策略？
7. `vs_random` 使用 deterministic eval（argmax），`vs_previous` 保留温度与采样。这两种评估模式测量的是不同的能力：前者测"最优落子质量"，后者测"分布质量"。当两者指标不一致时（vs_random 上升但 vs_previous 持平），模型到底是进步还是退步？
8. 锦标赛中每局 1000 盘，`temperature=1.0`，`sample_moves=True`。高温度意味着策略接近 MCTS 原始访问分布，包含更多探索性动作。这种设定测的是模型的"平均水平"还是"最好水平"？与实际部署（可能用低温度）的表现差距有多大？
9. 如果一个模型在 `vs_random` 上 99.9% 但在锦标赛中排名中等，最可能的解释是什么——它学会了利用随机对手的弱点但对强对手无效？还是锦标赛的随机性太大？
10. Elo / Bradley-Terry 排名在高平局率环境下的行为：如果 90% 的对局是平局，Elo 差异几乎完全由 10% 的胜负决定。小样本下 Elo 估计的置信区间有多宽？`GAMES_PER_MATCH=1000` 是否足以让排名稳定？
11. `evaluate_against_agent_parallel_v1` 使用 `_V1EvalAgent`，其中 MCTS 配置与自博弈相同（可配置 `num_simulations`、`temperature`）。如果评估时的 `num_simulations` 与训练时不同，评估结果能代表模型在训练时配置下的强度吗？
12. `gating_accept_candidate` 只比较 `vs_previous`，不看 `vs_random`。如果新模型 vs_previous 胜率微增但 vs_random 大跌（例如从 99% 降到 95%），gating 仍然通过。这是否意味着 gating 标准存在盲区？
13. 评估时每次都从初始状态开始对局。如果两个模型在开局阶段相似但在残局阶段差异显著，短对局（快速分出胜负）和长对局（进入走子阶段才分出胜负）对评估结果的贡献是否应该不同？
14. `MATCH_POINTS_WIN=3, DRAW=1, LOSS=0` 的积分制与足球联赛类似。在六洲棋中，如果大多数比赛是平局（得 1 分），那么一场胜利（得 3 分）的权重被大幅放大。这是否过度奖励了"偶尔取胜"的模型，而低估了"从不输棋"的模型？
15. `tournament_v1_eval.py` 中分组使用 `seed=20260226` 做随机 shuffle。如果不同 seed 产生显著不同的排名结果，锦标赛的结论（"iter 32 是最强的"）可信度有多高？是否应该跑多个 seed 然后取 Elo 排名？
16. `_V1EvalAgent` 在评估时使用 `V1RootMCTS`，即 root-only 搜索。如果训练时也是 root-only MCTS，评估与训练条件一致。但如果有朝一日切换到完整树搜索训练，评估也需要同步切换。这种评估-训练耦合是否应该被显式管理？
17. `vs_random` 是"健康度探针"。当它达到 99.8% 后继续提升空间极小。这个指标在模型发展后期是否已经失去区分力？是否需要设计更有区分度的基线对手（例如使用固定弱模型替代 Random）？
18. 评估中没有"按阶段拆分"的胜率统计。如果模型在 `PLACEMENT` 阶段很强但在 `MOVEMENT` 阶段很弱，当前评估只能看到总体胜率。是否应该添加阶段级别的诊断信息？
19. `evaluate_against_agent_parallel` 使用 `_split_game_indices` 做 round-robin 分配。如果某些 worker 的对局系统性更长（例如特定开局序列导致长对局），worker 间的负载是否均衡？不均衡的评估是否会因为超时或资源争抢而产生偏差？
20. 如果模型 A 在 vs_previous 中胜率 55%，模型 B 在 vs_previous 中胜率 60%，但 A 的 previous 比 B 的 previous 更强，A 和 B 谁更强？当前评估体系是否缺少跨迭代的绝对强度衡量？
21. 锦标赛冠军模型是否一定适合作为下一轮训练基线？冠军模型可能是通过"特化某种策略"在锦标赛中取胜，但这种特化策略在自博弈中是否会导致数据多样性下降？
22. 如果评估采用 opening book（固定前 N 步）来标准化开局，是否能降低评估方差？当前评估完全依赖模型自己的开局选择，不同评估 run 之间的开局序列可能差异很大。
23. `vs_self` 评估（模型自己打自己）在 `big_train_v1.sh` 中也有入口。自己打自己的胜率理论上应接近 50%/50%（加上平局）。但如果黑白先后手不对称，`vs_self` 的胜率偏差可以量化先后手优势。这个信息是否被有效利用？
24. 评估报告以 JSON 格式保存。`gating_accept_candidate` 只读取 `name == "vs_previous"` 的行。如果 JSON 文件损坏或格式变化，gating 逻辑是否有 fallback？是否存在静默接受错误模型的风险？

## 8. 系统设计与工程取舍

1. v1 的核心价值定位是"在 `v0_core` 之上建 staged tensor pipeline"。但 `v0_core` 本身包含完整的 MCTS、规则引擎、张量状态批处理能力。v1 只使用了 `v0_core` 的子集（`root_pack_sparse_actions`、`batch_apply_moves`、`fast_legal_mask`、`finalize_trajectory_inplace` 等），跳过了 `MCTSCore` 的完整树搜索。这个子集是经过深思熟虑的 API 边界，还是 v1 早期开发的路径依赖？
2. `v0/python/mcts.py` 封装了三种 backend（`graph`/`ts`/`py`），`V1RootMCTS` 自己实现了不同的搜索路径。为什么 v1 没有复用 v0 的 MCTS wrapper，而是重新实现了 root-only 变体？这是因为 root-only 语义无法适配 `MCTSCore` 接口，还是因为跨代码共享的维护成本太高？
3. `scripts/big_train_v1.sh` 作为训练主入口，包含 shell 级别的 curriculum 计算（`compute_curriculum_values`）、gating 逻辑、错误处理和迭代调度。如果把这些逻辑下沉到 Python（`train_entry.py` 或 `v1/train.py`），可测试性和可调试性是否会显著提升？保留 shell 的理由是什么？
4. `train_entry.py` 通过 `--pipeline v0|v1` 统一两代训练入口。但 v0 路径是通过 subprocess 调用 `train_loop.py`，v1 路径是直接调用 `train_pipeline_v1`。这种不对称意味着 v0 的错误处理和日志管理与 v1 完全不同。统一入口的实际价值有多大？
5. `v0_core` 作为 PyBind11 模块暴露 C++ 能力。每次修改 C++ 源码（`v0/src/`）都需要重新编译。编译一次需要多长时间？这个编译成本是否阻碍了快速实验？如果把某些频繁修改的逻辑（如 `_soft_tan_from_board_black`）保留在 Python 端，是否更灵活？
6. `v1/python/self_play_worker.py` 的 process backend 通过文件系统共享数据（chunk 文件 + manifest）。worker 之间没有直接通信。如果改用共享内存或 NCCL AllGather，是否能减少 I/O 开销？当前的文件 I/O 在 NVMe 和网络存储上的表现差异有多大？
7. `TensorTrajectoryBuffer` 预分配固定大小的张量 arena（`max_steps_hint * concurrent_games_hint`）。如果实际步数超出预分配，是否会报错？是否有动态扩容机制？如果没有，预分配的保守程度如何影响显存利用率？
8. 文档结构中 `docs/` 聚焦 v1，`v1/Design.md` 保留为深设计归档。如果新开发者只读 `docs/`，是否会遗漏 `Design.md` 中记录的关键设计决策（如为什么选择 root-only MCTS、为什么用 bucketed value）？
9. `check_rule_engine_cases.py` 使用 1000+ 用例做规则对拍。这些用例是手工编写的还是自动生成的？如果是手工的，它们是否覆盖了所有 `Phase` 转换的边界条件？如果某些边界条件未被覆盖（如同时触发方和洲），错误可能在多少轮自博弈后才被发现？
10. `backend/main.py` 和 `web_ui/` 构成人机对战系统。这个系统使用的推理路径是否与自博弈/评估中使用的相同（`V1RootMCTS`）？如果不同，人类玩家体验到的 AI 强度与评估指标之间可能存在差距。
11. `scripts/tournament_v1_eval.py` 的 `STAGE_PLAN` 硬编码了 80→32→16→8→4→1 的淘汰路线。如果 checkpoint 数量不是 80（例如因为 gating 失败只有 45 个有效模型），这个脚本是否能自适应调整 stage 计划？
12. `v0/src/game/fast_legal_mask.cpp` 中的 CUDA 内核和 CPU 回退路径是两套独立实现。如果只修改了 CUDA 路径但忘记同步 CPU 路径，`encode_actions_fast` 的自动回退会静默产生不一致的 mask。这种双路径维护的风险如何管理？
13. `pytest.ini` 配置了测试运行方式。当前测试套件（`tests/`）的运行时间有多长？如果超过 5 分钟，是否会阻碍开发节奏？是否需要按标签拆分快速测试和慢速测试？
14. `filter_decisive_jsonl.py` 过滤 `value ≠ 0` 的样本。如果在训练之前先用这个脚本过滤掉所有平局样本，训练效果是否会更好？这与 `policy_draw_weight=0` 有什么区别？
15. 当前 `CMakeLists.txt` 要求 pybind11 + LibTorch + 可选 CUDA。如果在没有 CUDA 的机器上构建（例如 CI 环境），哪些功能会不可用？纯 CPU 路径是否覆盖了全部核心逻辑以支持测试？
16. `v1/train.py` 中 `_run_self_play_multi_device_thread` 和 `_run_self_play_multi_device_process_saved` 是两套不同的并行策略。它们在什么条件下产生不同的结果（不只是性能差异）？如果 thread backend 的 GIL 竞争导致某些 model forward 被延迟，对 MCTS 搜索质量有无影响？
17. `scripts/monitor_resources.sh` 和 `kill_top_io.sh` 是运维工具。在长期训练（60 轮迭代，每轮 selfplay+train+eval+infer）中，最常见的资源问题是什么——显存 OOM、CPU 内存泄漏、磁盘空间耗尽，还是 GPU 利用率低下？
18. `export_torchscript.py` 导出 TorchScript 模型。`InferenceEngine`（v0）使用 CUDA Graph 加速 TorchScript 推理。v1 的 `V1RootMCTS` 直接使用 PyTorch eager 模型做前向。如果 v1 也切换到 TorchScript + CUDA Graph 推理，自博弈速度能提升多少？
19. `v1/Design.md` 有 1800+ 行。它记录了设计里程碑与验收，但没有被 `docs/` 引用为必读材料。如果未来需要回溯某个设计决策的历史，`Design.md` 和 `git log` 哪个更可靠？
20. "保持设计简洁、不增加开关"这条 AGENTS.md 原则在强化学习系统里什么时候会变成障碍？例如，`anti_draw_penalty`、`policy_draw_weight`、`soft_label_alpha`、`soft_value_k` 已经是 4 个与平局相关的参数。这算"简洁"吗？

## 9. 面向研究的问题

1. 六洲棋的状态空间（6×6 棋盘、每格 3 态{空/黑/白}，加上标记位和阶段信息）远小于围棋但大于多数棋盘游戏。这意味着暴力搜索不可行但神经网络可能过拟合。当前 `ChessNet`（10 blocks、128 channels、约 1.2M 参数）的容量是否接近这个任务的 Goldilocks 区间？
2. 如果引入结构特征作为额外输入通道（例如每个位置是否属于方/洲、当前方/洲的数量），是否能加速 policy head 学到"保护结构"和"攻击结构"的策略？`is_piece_in_shape` 在 C++ 端已经计算了这些信息但没有传入网络。
3. 高平局率（早期 >98%）是这个项目最突出的挑战。在学术框架中，这类似于"稀疏奖励"问题。AlphaZero 原版对围棋/象棋没有这个问题。六洲棋的高平局率是规则固有的，还是模型/搜索不够强导致的？
4. 如果用棋子数量差（`soft_value_from_black`）训练一个纯监督的 value 模型，再用它初始化 AlphaZero 训练的 value head，是否能跳过 value head 从零学习的冷启动阶段？
5. 当前 MCTS 的策略目标 `π` 是访问计数的归一化分布。如果改用 MCTS 的 Q 值作为策略目标（Q 值高的动作获得更多概率），训练信号是否更直接？这等价于用 value 信号直接指导 policy，绕过了访问计数这个中间变量。
6. reanalyse（用更新的模型重新分析历史对局的状态，生成新的策略目标）在 MuZero 中被使用。如果在 v1 中引入 reanalyse，`TensorSelfPlayBatch` 的 `state_tensors` 可以直接复用，只需要重新运行 `V1RootMCTS.search_batch` 生成新的 `policy_targets`。这个改造的工程量有多大？效果是否值得？
7. 课程学习（curriculum learning）在当前系统中通过 `opening_random_moves` 衰减和 `soft_label_alpha` 调度部分实现。如果引入更激进的课程——例如先在 4×4 简化棋盘上训练，再迁移到 6×6——模型学到的特征能迁移吗？`BOARD_SIZE=6` 硬编码在多少个位置？
8. 如果把 MCTS 完全去掉，用 `ChessNet` 的 policy head 直接采样动作做自博弈（纯 policy iteration），训练是否会崩溃？或者说它会收敛到一个弱但稳定的策略？这个实验能量化 MCTS 在训练中的贡献。
9. `decisive_game_ratio` 从 1.23% 到 81.72% 的飞跃是目前最重要的里程碑。这个提升的归因分析：有多少来自 `opening_random_moves`、多少来自 `soft_label_alpha`、多少来自 `anti_draw_penalty`、多少来自模型容量增长？是否做过消融实验？
10. 六洲棋的动作阶段化使得一个"逻辑回合"可能包含多个 `Phase` 转换。如果把同一逻辑回合内的所有原子动作作为一个"宏动作"来训练（类似 options framework），是否能减少序列长度从而改善信用分配？
11. 当前模型学到的是"在自博弈训练制度下的最优策略"——它依赖于 Dirichlet noise、温度调度、`opening_random_moves` 等训练设定。如果把这些训练 artifact 全部去掉做 clean evaluation，模型的"裸强度"比训练时表现更强还是更弱？
12. `value_bucket_bins=101` 的离散 value 表示 vs. KataGo 风格的 `(value, score, ownership)` 多头输出。如果给 value head 增加一个 ownership 预测（每个位置最终属于谁），是否能提供更丰富的中间信号？六洲棋的棋子归属在走子阶段会发生变化，这与围棋的静态归属不同。
13. 如果引入对手建模（opponent modeling）——例如根据对手的历史动作推断其策略类型，再调整自己的策略——在自博弈（对手就是自己）的场景下，这是否退化为自我认知？是否有意义？
14. `PreActResBlock` + `GlobalPool` 的架构来自围棋 AI 的实践。6×6 棋盘上是否有更高效的架构选择？例如 Transformer（自注意力在 36 个位置上计算）、GNN（棋盘邻接图）、或者 MLP-Mixer？
15. 如果未来六洲棋的规则被正式标准化并加入竞赛，当前项目需要什么改动才能达到竞赛级别？规则实现的可审计性（`check_rule_engine_cases.py`）、搜索强度（`num_simulations`）、还是推理延迟（人类等待时间）是最关键的？
16. 项目最终想证明的到底是什么：能训出打败人类的 agent，能做高性能自博弈系统，还是能把一个无公开基准的民间棋类从规则形式化到训练闭环全部完成？答案如何影响后续资源分配——应该更多投入搜索改进、训练改进，还是规则验证与文档？

## 10. 给未来自己的追问

1. 你当前相信"root-only MCTS + value_only child eval 足够"的结论，来自代码里 `V1RootMCTS` 的实验结果，还是来自你不想重写 `MCTSCore` 接口的路径依赖？
2. `decisive_game_ratio` 从 1.23% 升到 81.72%，你确定这主要来自模型进步，而不是来自 `opening_random_moves` 创造了更多随机结局？如果把 `opening_random_moves` 设为 0 重新评估，`decisive_game_ratio` 会跌回多少？
3. 你有没有把"训练速度提升 25x-28x"误当成"方法改进"？v1 的吞吐提升来自 GPU tensor pipeline 和 wave batching，但搜索质量（root-only vs 完整树）实际上下降了。你确认净效果为正吗？
4. 你有没有把 `vs_random` 99.80% 胜率误当成"模型很强"？`RandomAgent` 的棋力下限极低，99.80% 可能只意味着模型比随机好，不意味着它能赢任何有策略的对手。
5. `soft_value_k=2.0`、`dirichlet_alpha=0.3`、`exploration_weight=1.0`、`temperature_threshold=10`——这些超参数是经过系统搜索的，还是沿用了其他项目的默认值？你有没有验证它们对六洲棋是最优的？
6. 如果今天必须删掉一半复杂度，你会删掉 `soft_value_targets`（以及 `soft_label_alpha`、`soft_value_k`）还是 `bucketed value`（回退到标量 value）？为什么？
7. 你有没有把"`replay_window=4` 能稳定训练"误当成"旧数据有用"？也许 `replay_window=1`（只用当前迭代数据）配合更多 epochs 效果更好——你试过吗？
8. 当前 `gating` 条件（`wins > losses`）非常宽松。你是否考虑过更严格的 gating 会导致更多迭代被浪费但长期棋力增长更稳？宽松 gating 是否在让"噪声进步"混入主线？
9. 如果今天必须保留一个最核心的研究问题继续做三个月，你会选"如何打破平局主导分布"（目标设计问题）还是"如何让搜索质量随计算量线性增长"（搜索效率问题）？
10. `v0_core` 作为 C++ 底层模块，它的 API 边界是否已经冻结？如果未来需要在 `root_puct_allocate_visits` 中加入新的探索策略（如 Progressive Widening），修改 C++ 代码并重新编译的周期是否可接受？
11. 你有没有把"当前实现方便"误当成"长期设计合理"？例如 `build_combined_logits` 中 movement 的 `log_p2 + log_p1` 分解是因为三头架构的限制，不是因为这种分解对六洲棋最优。你是否认真评估过替代方案？
12. 如果三个月后证明 bucketed value（101 bins）不如直接标量回归，你愿意回滚吗？`scalar_to_bucket_twohot`、`bucket_logits_to_scalar`、`bucket_value_loss` 这条路径的代码量不小——沉没成本是否影响了你的判断？
13. 如果未来证明当前很多设计都不是最优，你希望项目里哪一部分仍然是值得保留的——规则引擎的完整性与正确性、张量化自博弈 pipeline 的工程能力、还是从零开始形式化一个民间棋类的方法论经验？
14. 你在 `AGENTS.md` 中写了"保持简洁的设计准则，如无必要，不增加开关和参数传递"。但 `big_train_v1.sh` 中的 curriculum 调度已经包含了 `opening_random_moves`、`soft_label_alpha`、`lr`、`policy_draw_weight` 四个独立衰减曲线。你确定这还算简洁吗？

部分问题的解答已整理在 [高难问题解答手册](./faq_answers.md)。
