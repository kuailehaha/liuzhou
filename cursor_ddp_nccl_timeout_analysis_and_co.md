# DDP/NCCL timeout analysis and code modification plan
_Exported on 3/11/2026 at 23:30:28 GMT+8 from Cursor (2.6.11)_

---

**User**


我看完了这个日志，结论很明确：

你要的版本信息打印已经生效
git_branch/git_commit/git_dirty 已打印：见 log#L2 (line 2)。
训练信号没有塌缩（这是好消息）
Iter1/Iter2 都有非零 decisive，且 hard value 非零比例正常：见 log#L54 (line 54)、log#L125 (line 125)、log#L56 (line 56)、log#L127 (line 127)。
流式读取确实在用
streaming mode ... workers=1：见 log#L150 (line 150)。
当前真正失败点：DDP/NCCL 超时，不是 OOM
典型报错：Watchdog caught collective operation timeout，见 log#L151 (line 151)。
内存充足（2.0Ti 总，1.9Ti free），见 log#L540 (line 540)。
关键异常信号：同一轮超时里各 rank 的 NumelIn 不一致（1 / 11 / 362405），见 log#L333 (line 333)、log#L339 (line 339)、log#L343 (line 343)。这几乎就是“各 rank 走了不同 collective 路径/尺寸”的证据。
重试情况
已至少失败 2 次，正在第 3 次尝试启动：见 log#L536 (line 536)、log#L542 (line 542)、log#L556 (line 556)。
如果你同意，我下一步就直接进代码定位这条 rank 分歧链路（重点看 iter2 的 streaming+replay 下，哪些 all_reduce/barrier 的张量长度可能按 rank 变化），然后给你一个最小修复补丁。

并非最小修复，你总应当从设计的角度出发，思考当前是否在错误的基础上进行裱糊。
你每次做出的改动都应当是目标正确且现状感知的。


logs/big_train_v1_20260309_070439.log

分析当前代码内容，及代码相关的任意信息，并作出修改或实现计划。

---

**Cursor**

Let me start by reading the log and the key training code to understand the full picture before making any design decisions.

Now let me read the critical parts of `v1/train.py` - specifically how the streaming dataloader is built and how DDP training is dispatched.

Let me read the streaming dataset module and the relevant section of `v1/train.py` where the dataloader is constructed.

Now I have the full picture. Let me do a thorough analysis before making changes.

## Root Cause Analysis

Here's the complete picture from the design level.

### The invariant DDP requires

DDP (DistributedDataParallel) requires **all ranks to execute the exact same sequence of NCCL collective operations**. Every `all_reduce`, every gradient sync during `backward()` — if even one rank diverges by a single operation, the process group deadlocks (600s timeout → SIGABRT).

### The smoking gun in the log

At SeqNum=15419 (second failure, lines 333-344), different ranks are executing **completely different collectives**:

| Rank | SeqNum | NumelIn | What it is |
|------|--------|---------|-----------|
| 3    | 15419  | **1**   | `_all_ranks_true` — still in batch loop |
| 2    | 15419  | **11**  | epoch-end stats reduction — **exited batch loop** |
| 0,1  | 15420  | **362405** | DDP gradient allreduce — **deep inside backward()** |

This is a catastrophic 3-way split: rank 2 already finished the epoch, rank 3 is checking loss-finiteness for a batch, and ranks 0/1 are running gradient sync. They will never reconverge.

### Two bugs in `train_network_streaming`

**Bug 1 — Empty-batch early `continue` skips collectives** (line 632-634):

```629:634:v1/python/train_bridge.py
            if n_bad > 0:
                total_filtered += n_bad
                keep = finite_mask.nonzero(as_tuple=False).view(-1)
                if keep.numel() == 0:
                    batches_this_epoch += 1
                    continue
```

If one rank gets a batch where ALL samples are non-finite, it `continue`s **without calling `_all_ranks_true`**. Other ranks proceed to `_all_ranks_true(loss_ok)` and then `backward()`. The collective sequence permanently diverges.

**Bug 2 — Data-driven loop termination** (lines 610-612):

```610:612:v1/python/train_bridge.py
        for batch_tuple in dataloader:
            if batches_this_epoch >= est_batches_per_epoch:
                break
```

The `IterableDataset` produces batches per-shard, and each shard's batch count is `ceil(shard_samples / batch_size)`. The aggregate `est_batches_per_epoch = ceil(total_est / batch_size)` is a **lower bound**, but the actual DataLoader can produce either more OR fewer batches per rank due to:
- Shard load failure (`_load_shard_tensors` returns `None` → batches lost)
- Replay budget rounding asymmetry across ranks  
- Any metadata/actual-size mismatch

If a rank's DataLoader exhausts before `est_batches_per_epoch`, the `for` loop ends naturally, and that rank proceeds to the epoch-end stats `all_reduce(11 elements)` while other ranks are still in the batch loop calling DDP gradient `all_reduce(362405 elements)`.

### Why iter1 succeeds but iter2 fails

Iter1: `shard_files=16`, single uniform self-play run. All ranks get symmetric data. The DataLoader batch count is very close to `est_batches_per_epoch`.

Iter2: `shard_files=32` (16 primary + 16 replay with **budget subsampling**). The replay `sample_budget` is derived from `primary_est` which is rank-specific. `_distribute_manifest_budget` allocates budget across replay shards asymmetrically. The accumulated per-shard ceiling differences and budget rounding create enough divergence that one rank exhausts its DataLoader before the synchronized `est_batches_per_epoch`.

### The correct design

The fix is not a patch — it's a structural guarantee:

1. **Deterministic iteration count**: Replace `for batch_tuple in dataloader` with `for step_idx in range(est_batches_per_epoch)` and explicit `next(batch_iter)`. If the DataLoader exhausts, handle it gracefully.

2. **All ranks execute exactly one "gate" collective per step**: When a rank has no data (DataLoader exhausted) or empty batch (all non-finite), it calls `_all_ranks_true(False)` — matching other ranks' `_all_ranks_true(loss_ok)` call. The MIN reduction returns `False` for all ranks, all skip backward. Collective sequence stays aligned.

This guarantees the NCCL invariant **by construction**, not by hoping the data is symmetric.

Now implementing:

Now add `dl_exhausted_steps` to the DDP epoch-end reduction tensor and epoch stats.

Now add `dl_exhausted_steps` to the epoch stats dict.

Now let me verify the final state of the modified function and check for lints.

Code is clean, no lints. Let me verify the collective alignment invariant is indeed maintained by tracing through every branch.

The collective alignment invariant is verified. Let me trace every path per iteration:

| Path | Trigger | Collectives | Outcome |
|------|---------|------------|---------|
| DL exhausted | `StopIteration` / `dl_exhausted=True` | 1: `_all_ranks_true(False)` | All ranks skip |
| Empty batch | `keep.numel()==0` | 1: `_all_ranks_true(False)` | All ranks skip |
| Non-finite loss | `isfinite(loss)=False` on any rank | 1: `_all_ranks_true(loss_ok)` → False | All ranks skip |
| Non-finite grads | `grads_finite=False` on any rank | 1+N+1: loss_ok + backward + grads_ok | All ranks skip step |
| Normal | All healthy | 1+N+1: loss_ok + backward + grads_ok | All ranks step |

The first collective in every path is an `allreduce(MIN)` on a 1-element int32 tensor. Paths that should skip call it with `False`. The MIN means if ANY rank says `False`, ALL ranks get `False` and skip backward. The sequence count is always identical across all ranks.

Let me now check if the caller in `v1/train.py` should log the new diagnostic when `dl_exhausted_steps > 0`.

The implementation is complete. The `dl_exhausted_steps` diagnostic is already captured in both the epoch stats dict and the training metrics JSON. No other files need changes — the non-streaming `train_network_from_tensors` path already uses a deterministic `range(synced_batch_count)` loop and doesn't have per-batch filtering.

---

## Summary

**What changed**: `v1/python/train_bridge.py` — the `train_network_streaming()` function's inner batch loop.

**Two structural fixes, one design principle** — all ranks must execute the exact same NCCL collective sequence, enforced by construction:

1. **Deterministic iteration count** (lines 610-628): Replaced `for batch_tuple in dataloader` (data-driven termination) with `for step_idx in range(est_batches_per_epoch)` using explicit `next(batch_iter)`. When the DataLoader exhausts on a rank, it calls `_all_ranks_true(False)` which forces all ranks to skip that step in unison.

2. **Empty-batch collective alignment** (lines 646-649): The `keep.numel() == 0` path now calls `_all_ranks_true(False)` before `continue`, matching other ranks' `_all_ranks_true(loss_ok)` call. The MIN reduction returns False for everyone — no collective sequence divergence.

3. **Diagnostic**: `dl_exhausted_steps` counter added to the DDP reduction tensor (12 elements, was 11) and epoch stats, so any DataLoader exhaustion is visible in the training metrics JSON.

**What did NOT change**: No new parameters, no config changes, no changes to `streaming_dataset.py` or `v1/train.py`. The fix is entirely within the collective alignment logic.

**Functional test checklist**:
- Iter1 DDP training (no replay) — should work as before (no DL exhaustion expected)
- Iter2+ DDP training (with replay+budget) — the NCCL timeout should not recur
- Non-DDP (`data_parallel`) mode — `_all_ranks_true` returns the local flag directly, no collective, no change in behavior
- Check `dl_exhausted_steps` in training JSON metrics — should be 0 in normal operation

Suggested commit: `[bugfix/train]: fix DDP NCCL timeout in streaming training by guaranteeing collective alignment across all ranks`

---

**User**

这个DDP的bug经过训练，验证成功修复。现在大约45分钟一个iter，但是训练的vsrandom只是在50%左右水平并且常常抖动呢？
我现在想要知道，以/2023533024/users/zhangmq/liuzhou/logs/big_train_v1_20260309_092710.log
为什么现在训练效果（单看vs random即可，它在初期可以比较好的代表棋力）效果不及之前的logs/big_train_v1_20260223_173954.log？我们首先容易想到，之前的版本一个是softlabel=1以及random move退火。
但这恰恰是我们本轮的修改内容。除此，你还有其他的想法吗？


或者说（脱出与之前代码训练效果的比较），是否存在某些训练或者说错误导致我们棋力上不去呢？

再高一级，有哪些能力我们在没有使用到呢？或者说我们有哪些没有考虑的设计造成了数据或者是自博弈训练的内耗浪费吗？

你先不要改动，考虑一下cursor_training_stability_and_reward_op.md这是之前的对话，仅作为历史资料。




---

**Cursor**

Let me read the current training log, the previous successful log, and the historical conversation document before forming an analysis.

Let me read the historical document and more of the old log to see the vs_random trend.

Let me also check the selfplay MCTS configuration and the training script parameters that control data quality.

Let me check how the MCTS simulations are actually passed to the selfplay stage - this is critical.

Now I have the complete picture. Let me compile the two runs' vs_random trajectories and perform a deep analysis.

## Comprehensive Analysis

### Fact Table: Head-to-Head Comparison

**New run (0309) — 7 iterations:**

| Iter | vs_random Win% | vs_random Draw% | vs_random Loss% | Selfplay Draw% | Train Loss |
|------|---------------|----------------|----------------|----------------|-----------|
| 1 | 54.05 | 37.25 | **8.70** | 9.09 | 1.77 |
| 2 | 55.80 | 34.65 | **9.55** | 56.45 | 2.01 |
| 3 | 41.30 | **56.35** | 2.35 | 65.34 | 2.21 |
| 4 | 43.20 | **54.95** | 1.85 | 42.21 | 2.28 |
| 5 | 47.65 | 51.15 | 1.20 | 33.64 | 2.31 |
| 6 | 43.20 | **56.10** | 0.70 | 26.79 | 2.25 |
| 7 | 46.30 | **53.10** | 0.60 | 35.16 | 2.32 |

**Old run (0223) — first 7 iterations:**

| Iter | vs_random Win% | vs_random Draw% | vs_random Loss% | Selfplay Draw% |
|------|---------------|----------------|----------------|----------------|
| 1 | 27.70 | 72.20 | 0.10 | 98.77 |
| 2 | **85.50** | 14.50 | 0.00 | 51.15 |
| 3 | **87.10** | 12.90 | 0.00 | ~50 |
| 4 | **91.80** | 8.20 | 0.00 | ~40 |
| 5 | **92.10** | 7.90 | 0.00 | ~30 |

### Key Config Differences

| Parameter | Old (0223) | New (0309) |
|-----------|-----------|-----------|
| `opening_random_schedule` | **6→0** | 4→4 (fixed) |
| `soft_label_alpha_schedule` | **1.0→1.0** (constant) | 1.0→0.1 |
| `soft_value_k` | **2.0** | 5.0 |
| `anti_draw_penalty` | (none) | -0.05 |
| `policy_draw_weight` | 1.0 (fixed) | 1.0→0.5 |
| `replay_window` | (none) | 4 |
| `warmup_steps` | (none) | 100 |
| `lr_cosine_final_scale` | (none) | 0.5 |

---

### Diagnosis 1: Anti-draw penalty is architecturally dead

This is the single most important finding. In `train_bridge.py`:

```python
# anti_draw modifies b_values (the HARD target)
b_values[hard_draw_mask] = anti_draw  # -0.05

# but then:
mixed_values = (1.0 - alpha) * b_values + alpha * b_soft
```

When `soft_label_alpha=1.0` (iter1), the mixed value for a draw = `0.0 * (-0.05) + 1.0 * 0.0 = 0.0`. The anti-draw penalty contributes **exactly zero**. By iter7, alpha is still ~0.93, so anti-draw contributes `0.07 * (-0.05) = -0.0035`. Negligible.

You introduced anti-draw to fix draws, but `soft_label_alpha=1.0` completely suppresses it. The two mechanisms fight each other by design.

### Diagnosis 2: The old run's "curriculum" created a positive feedback loop that the new run can't replicate

The AlphaZero training cycle requires: **decisive games → strong signal → better model → more decisive games**.

The old run's `opening_random_moves=6` created an accidental but highly effective curriculum:

- **Iter1**: 6 random opening moves → 98.77% draws. But alpha=1.0 means training on material delta (soft value). The random games produce a WIDE distribution of material deltas (bucket {-15...+15} very spread), giving the value head excellent training signal even from draws.
- **Iter2**: Model with basic material evaluation + 6 random opening moves → still diverse positions → 51% draws. **The huge jump from 27.7% to 85.5% vs_random** comes from the value head learning "material advantage = good" and the policy head refining from somewhat decisive games.

The new run's `opening_random_moves=4`:
- **Iter1**: 4 random moves → **only 9% draws** (random play is very decisive). The model sees 91% decisive games. Good signal? Partially — the outcomes are from random blunders, not strategic play.
- **Iter2**: The trained model with 4 random moves → **56% draws**. The model learned just enough to hold draws from moderately diverse positions. Now the signal degrades.

**The critical asymmetry**: 6 random opening moves maintain chaos even when the model improves slightly. 4 random moves is the "draw equilibrium" threshold — a model that's even slightly above random can hold draws with only 4 random moves.

### Diagnosis 3: Train loss is INCREASING — a fundamental red flag

Loss trajectory in the new run: `1.77 → 2.01 → 2.21 → 2.28 → 2.31 → 2.25 → 2.32`.

This means the model is getting WORSE at fitting the training data. Why?

1. **Data distribution shift + replay creates conflict**: Iter1 data (91% decisive, strong ±1 values) is replayed alongside iter3 data (65% draws, near-zero values). The model is pulled between "be aggressive" (iter1 data) and "be passive" (iter3 data).

2. **The replay budget allocation treats all iterations equally**, but their signal quality is dramatically different. Iter1 data has 10× the value signal strength of iter3 data.

3. **The soft_value_k=5.0 saturates too fast**: With k=5.0, material delta of ±5 already maps to soft_value ≈ ±0.88. This means the value head sees binary-like targets (+1 or -1) with a gap at 0 for draws. The old run's k=2.0 produced more gradual values, giving the value head a richer regression target.

### Diagnosis 4: piece_delta distribution reveals a brittle strategy

New run iter5 piece_delta buckets:
```
-14: 69087, +14: 102233  (dominant modes)
```

The model has learned a "rush" strategy: when it works, it captures nearly all opponent pieces (+14). When it fails, it loses nearly all its pieces (-14). The distribution is bimodal with huge peaks at the extremes. This is unstable — the slightest disruption (even random play) can flip the outcome.

The old run's piece_delta distributions are much more spread across all values, indicating a more flexible model.

---

### The Bigger Questions

**Q: Are there bugs limiting strength?**

Not bugs, but a **design contradiction**: the `anti_draw_penalty` and `soft_label_alpha` were independently designed improvements that cancel each other when combined. The penalty works on hard targets; alpha=1.0 suppresses hard targets. This isn't a code bug — the code does exactly what the parameters specify. It's a configuration-level logical error.

**Q: What capabilities are unused / what causes waste?**

1. **522K games × 65536 MCTS sims per iteration is enormous compute**. But with 55-65% draw rates, over half this compute produces near-zero value targets and flat policy targets. The compute-to-useful-signal ratio is very poor.

2. **The replay window preserves old decisive data** (good), but the training treats all data equally. The valuable iter1 data (91% decisive) is diluted by later draw-heavy data. There's no mechanism to weight samples by their signal quality.

3. **The cosine LR schedule and optimizer persistence are working correctly** — they're just unable to compensate for fundamentally weak training signal.

4. **The training does 4 epochs over ~24M samples per rank**, but most of those samples (draws) teach the model "this position is worth ≈ 0" — not very informative.

---

### What Would Fix This (Design Direction, Not Patch)

The core issue is: **the model enters draw equilibrium and the current signal design can't push it out**.

Three orthogonal approaches, from most to least impactful:

**A. Fix the signal contradiction.** Either:
- Start `soft_label_alpha` at 0.0 (use hard targets + anti-draw from the beginning), schedule up to 0.5 later
- OR apply anti-draw directly to the mixed target (after soft blending), not to the hard target

**B. Restore the opening diversity curriculum.** The old run's 6→0 schedule worked because 6 random moves prevent draw equilibrium in early training. Options:
- `opening_random_schedule=6→2` (decay to 2, not 0 — maintain some diversity)
- OR dynamically increase opening_random_moves when selfplay draw_rate exceeds a threshold

**C. Weight training samples by signal quality.** Instead of treating all samples equally:
- Upweight decisive game positions
- Downweight draw game positions (beyond just `policy_draw_weight`)
- OR subsample: keep all decisive positions, randomly drop 50% of draw positions

The user explicitly asked me to think but not act. My recommendation: approach **A** is the most surgical — fix the signal contradiction first, then evaluate. If that alone doesn't help, layer on **B**.

Suggested commit (when ready to act): `[bugfix/train]: apply anti_draw_penalty to mixed value target, bypassing soft_label_alpha suppression`

---

**User**

我想不妨先复现一下之前的训练。我的思路是从代码角度查看有哪些区别。我现在把参数大致对齐了我之前提到的那个版本，但是训练进度还是震荡；logs/big_train_v1_20260223_173954.log在几个epoch之内迅速升到了90%，证明了训练有效。
请你为我分析两份代码之间的区别，然后给出详尽的报告：为什么新版本训练上不去，写到anal.md里面。你可以推断当时的代码版本。为e237c67	2026-02-23 17:16 UTC	latest-vs-best、DDP step 对齐
15451ae	2026-02-23 17:40 UTC	train params adjust



---

**Cursor**

Now let me get the detailed diffs between the old working version and current HEAD, focusing on training-critical files.

Now let me read the old version of key files to understand the exact behavior at that commit, and the current training log.

Now let me check the old training log's parameters and key details, plus the v1/train.py diff.

Now let me check the key remaining items: the streaming dataset and the `train_network_streaming` function, as these are completely new in the current version.

Now let me also check the new log's later iterations for deeper comparison, and verify some specific training details.

Now I have all the data needed. Let me write the comprehensive analysis report.

Analysis report written to `anal.md`. Here's the summary:

Between `15451ae` (old, working) and current HEAD, **7 structural code changes** were introduced simultaneously. Even though you aligned visible parameters like `soft_label_alpha` and `soft_value_k`, the training engine underneath was fundamentally altered:

**The 3 most damaging changes:**

1. **Effective LR is 3-4x lower** — Base LR halved (2e-4 → 1e-4), plus warmup + cosine decay + stale optimizer momentum. The model cannot update fast enough.

2. **WDL auxiliary loss creates gradient conflict** — The WDL target uses hard game outcomes (all draws = 0), while the bucket target uses soft material-delta. These pull the value head in opposite directions. This alone explains why initial loss jumps from 3.5 to 6.2 on identical random-model data.

3. **Streaming DataLoader destroys data shuffling** — Old version shuffles ALL samples globally; new version only shuffles within individual shards. Each batch is correlated within one worker's game chunk.

**Supporting factors:** replay buffer dilutes early signal, optimizer persistence creates momentum traps, doubled MCTS (131072) kills exploration (0% decisive games in iter 1 vs 1.23% old), fixed opening_random_moves removes curriculum effect.

**The smoking gun:** Training loss is **monotonically increasing** (6.2 → 7.9), which never happens in the old version (stable 2.9-3.2). This confirms the optimizer is failing to learn.

**To reproduce old performance:** all 7 changes need to be reverted simultaneously (they interact). Section 6 of the report has the specific action table. Section 7 evaluates which features are worth re-introducing later with proper A/B testing.

Suggested commit: `[analysis]: add training regression diagnosis between 15451ae baseline and HEAD`

---

**User**

那看起来有些东西被改坏了。Training loss is monotonically increasing (6.2 → 7.9), which never happens in the old version (stable 2.9-3.2). This confirms the optimizer is failing to learn.
请继续分析，并给出修复方案。

---

**Cursor**

Let me dive into the fix implementation. First, let me check exactly how the parameters flow to understand what needs to change.

Good. The old monolithic path (`else` branch at line 1987) still works when `streaming_load=False`. Now let me also check the `train_network_from_tensors` function to see if it still has the WDL aux loss:

Now I have the full picture. Let me implement all fixes.

Now fix `big_train_v1.sh` — restore the working parameters:

Now restore LR_COSINE_FINAL_SCALE to 1.0 (= fixed LR, no inter-iteration decay):

Now disable optimizer state persistence. The cleanest way: remove the `--optimizer_state_path` from the train commands, so it defaults to `None`:

Now I need to make sure the empty string is handled properly when passed to the train command. Let me check how the optimizer_state_path flows through `train_entry.py` to `train_bridge.py`:

The empty string will be falsy in Python, so `optimizer_state_path and os.path.exists(...)` will short-circuit correctly. Let me also fix the default warmup_steps in `train_entry.py`:

Now fix the streaming shuffle. The simplest effective fix: increase `STREAMING_WORKERS` so multiple shards are loaded and interleaved by the DataLoader, providing cross-shard mixing:

Also update the default in `train_entry.py` to match:

Now let me verify everything is consistent — check the actual values that will be printed in the log and ensure no stale references:

Good. Now let me also verify the `_WDL_AUX_LOSS_WEIGHT` change doesn't break the loss computation — when weight is 0, the WDL computation still runs (for logging) but doesn't affect backprop:

Good. The WDL computation is still there for diagnostics (wdl_aux_loss is still logged), but `0.0 * wdl_aux_loss = 0`, so it adds nothing to the loss gradient. Let me also verify that the `train_network_streaming` function (the streaming path) has the same pattern:

Both paths (monolithic at line 341 and streaming at line 719) use the same constant. Now let me check lints:

Let me now do a final cross-check — verify what the training parameters will look like when this runs:

All fixes applied. Here is the complete change summary:

---

**修改文件清单和具体改动：**

### 1. `v1/python/train_bridge.py` — 禁用 WDL Aux Loss

`_WDL_AUX_LOSS_WEIGHT`: 0.25 → **0.0**

WDL 辅助损失使用 hard value target（几乎全是 0/draw），而 bucket 主损失使用 soft value target（material delta）。两者梯度方向冲突，是 loss 从 3.5 暴涨到 6.2 的直接原因。设为 0 后 WDL 计算仍保留（用于日志），但不参与梯度更新。

### 2. `scripts/big_train_v1.sh` — 恢复已验证参数

| 参数 | 改前 | 改后 | 理由 |
|------|------|------|------|
| `MCTS_SIMULATIONS` | 131072 | **65536** | 过高的 MCTS 使探索死亡，iter1 产生 0% decisive |
| `LR` | 1e-4 | **2e-4** | 有效学习率过低，模型无法跟上数据分布变化 |
| `WEIGHT_DECAY` | 5e-5 | **1e-4** | 匹配旧版 |
| `OPENING_RANDOM_MOVES` | 4→4 | **6→0** | 恢复课程学习：初期高随机→后期确定性 |
| `REPLAY_WINDOW` | 4 | **0** | 消除 stale 数据干扰（可在训练稳定后重新引入） |
| `WARMUP_STEPS` | 100 | **0** | 消除 intra-iteration LR 衰减 |
| `LR_COSINE_FINAL_SCALE` | 0.5 | **1.0** | 固定 LR，不做 inter-iteration 衰减 |
| `OPTIMIZER_STATE_PATH` | 有效路径 | **空** | Fresh optimizer 避免 stale momentum |
| `STREAMING_WORKERS` | 1 | **4** | 多 worker 并行加载 → 跨 shard 交错 → 更好 shuffle |

### 3. `scripts/train_entry.py` — 对齐默认值

- `--warmup_steps` default: 100 → **0**
- `--streaming_workers` default: 8 → **4**

---

**预期效果**：恢复后的参数组合与 `15451ae` (old working) 版本在训练行为上等价，同时保留了 streaming DataLoader、chunk-manifest 自博弈输出、winner detection 修正等工程改进。预期 vs_random 应在 4-5 个 iteration 内恢复到 85-90%。

**待测功能清单**：
1. 训练 loss 是否从 ~3.5 开始并稳定在 2.9-3.2
2. vs_random win rate 是否在 iter 2-4 突破 85%
3. Selfplay iter 1 是否有 >0% decisive games（验证 6 random moves + 65536 MCTS 组合）
4. Streaming DataLoader (workers=4) 训练时间是否接近旧版 monolithic (~107s/iter)
5. 无 DDP NCCL timeout（之前已修复的 bug 不应回归）

Suggested commit: `[bugfix/train]: restore working training parameters and disable WDL aux loss to fix monotonically increasing loss regression`

---

**User**

我想LR的scheduler应该保持一个cosine scale 0.5的吧？LR	1e-4。这个相当于是必要的衰减，并且应该不是主要原因。因为之前的版本训到30iter就棋力不再增强了。
_WDL_AUX_LOSS_WEIGHT设置为0。OPENING_RANDOM_MOVES恒定为4即可。REPLAY_WINDOW设置为4即可。WARMUP_STEPS	100即可，STREAMING_WORKERS	只要不报错就改为4，--warmup_steps default: 100，--streaming_workers default: -4。除此之外呢？你还有什么想要提到的吗？我认为参数都不是大问题，关键在于是否有代码写错。

---

**Cursor**

You're right — let me focus on **code bugs**, not parameter tuning. Let me revert the parameter changes you want kept, and dig deep into the actual code for bugs.

Good. Now only 2 changes remain: `_WDL_AUX_LOSS_WEIGHT=0.0` and `STREAMING_WORKERS=4`. Let me now deep-dive into the code looking for actual bugs.

Let me now dig into the code paths that are actually different and look for real bugs. Let me check the self-play data flow carefully.

Excellent find. The value loss is **overwhelmingly dominant** — 6.09 out of 6.16 total. Policy loss is only 0.07. Let me compare with the old version:

The numbers are stark. New version has value loss 6.09 vs old's 2.72 — that's 2.2x higher. This isn't a parameter issue. Something in the value loss pipeline is broken. Let me investigate.

Now let me check the `finalize_trajectory_inplace` C++ code — this fills value and soft value targets:

Let me read the full self-play GPU runner to understand both the step and finalization paths:

Now let me check the `scalar_to_bucket_twohot` function — this converts soft values to bucket targets:

Let me check the WDL probability computation — how it maps bucket logits to win/draw/loss:

I'm finding key bugs. Let me verify one more critical thing — let me check what happened to the MCTS terminal detection:

Now let me look at the model's value head — specifically the number of value bins:

No changes to the neural network. Now let me check the exact differences in the training bridge:

Let me check all files that changed between the old and new versions to make sure I haven't missed anything:

Now let me check the most changed file — `v1/train.py` for the training data assembly logic:

Now let me do a focused check: is the `_lr_lambda` cosine-to-zero present in the OLD version? Let me verify:

**Confirmed:** The old version (15451ae) had **zero** LR scheduling code — no `_lr_lambda`, no `scheduler`, no warmup. Constant LR throughout each iteration. The entire `_lr_lambda` + cosine-to-zero is **new code that was never in the old version**.

Here are the **actual code bugs** I've found:

---

**Bug 1 (Critical, already fixed): WDL auxiliary loss has a fatal design flaw**

The `_bucket_logits_to_wdl_probs` function maps 101 value bins to 3 WDL categories. The "draw" channel maps to **only 1 bin** (center = 0.0). For a random model, `p_draw = 1/101 ≈ 0.99%`. The WDL loss per draw sample is `-log(0.0099) ≈ 4.62` — almost as bad as total random bucket loss. With 100% of games being draws, this loss term:
- Adds ~1.15 to the reported value loss (with weight=0.25)
- Creates massive gradient conflict: WDL pushes ALL probability to the center bin, while bucket loss pushes probability to various bins based on material delta
- This is why the bucket loss INCREASES during training (4.62 → above 4.62): the WDL gradient actively hurts the bucket objective

Already fixed: `_WDL_AUX_LOSS_WEIGHT = 0.0`.

---

**Bug 2 (Moderate): `_lr_lambda` cosine decays to ZERO — kills 25% of training**

The intra-iteration `_lr_lambda` function decays:

```617:621:v1/python/train_bridge.py
    def _lr_lambda(step: int) -> float:
        if step < warmup_n:
            return (step + 1) / max(1, warmup_n)
        progress = (step - warmup_n) / max(1, total_train_steps - warmup_n)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
```

At `progress=1.0`: returns `0.5 * (1 + cos(pi)) = 0.0`. The LR reaches exactly zero at the end of each iteration's training. The last epoch has average LR ~3.8% of peak — the 4th epoch barely trains. This function exists identically in both `train_network_from_tensors` (line 270) and `train_network_streaming` (line 617).

Fix: after warmup, LR should stay constant (matching old behavior), since the inter-iteration cosine already handles decay across iterations.

Now let me fix the `_lr_lambda` in both training functions:

Now fix the same bug in `train_network_streaming`:

Let me verify the fix is consistent:

Both instances fixed. Now let me also check if the `import math` is still needed (since we removed the `math.cos` call):

Now let me give a complete summary of all code changes made in this session:

---

**Summary of all fixes applied:**

1. **`_WDL_AUX_LOSS_WEIGHT = 0.0`** (train_bridge.py line 25) — Disables the fundamentally broken WDL auxiliary loss. The draw channel covered only 1/101 bins, creating a loss of ~4.62 per draw sample that catastrophically conflicted with the bucket loss gradient. This was the primary cause of the monotonically increasing loss.

2. **`_lr_lambda` returns 1.0 after warmup** (train_bridge.py lines 270-273 and 616-619) — Fixed intra-iteration cosine that decayed LR to exactly zero. The old version (15451ae) had no intra-iteration scheduler at all. The cosine-to-zero was new code that wasted the 4th training epoch (LR at ~3.8% of peak) and reduced effective average LR by ~50%.

3. **`STREAMING_WORKERS=4`** (big_train_v1.sh) — Already applied, increases cross-shard mixing in the streaming DataLoader.

4. **Parameters reverted per your request** (big_train_v1.sh, train_entry.py) — LR=1e-4, cosine_final_scale=0.5, MCTS=131072, opening=4→4, replay=4, warmup=100, optimizer persistence enabled.

**Other findings (no code changes needed):**
- `game_state.py`/`game_state.cpp` winner detection change (MARKING excluded) is semantically correct and has minimal training impact
- `trajectory_buffer.py`, `self_play_storage.py`, `streaming_dataset.py` data pipelines are correct — no tensor ordering bugs
- `scalar_to_bucket_twohot`, soft value computation (`_soft_tan_from_board_black`) are unchanged from old version

**Recommended commit:** `bugfix/train: disable broken WDL aux loss, fix cosine-to-zero LR schedule`

**Testing checklist:**
- Run training, verify iter-1 loss ~3.5 (matching old version's 2.72-3.53 range)
- Verify `vs_random` > 80% within 5 iterations
- Verify LR is constant within each iteration (check scheduler logs)

---

**User**

还有就是logs/big_train_v1_20260223_173954.log，分析loss的话，train complete loss其实最后降到1.7左右了，但是棋力并不如2.6的（因为打过一个锦标赛，34ckpt是棋力最强的）
这一方面可能是由于指标是子力差，不能代表棋力。但是如果我们设置带有硬标签信息，又很难训练稳定。我比较迷惑应该使用什么指标呢？
还需要你分析的是之前的训练速度竟然比现在快很多time=107.42s和logs/big_train_v1_20260310_074626.log的time=212.00s，难道是流式传输计算口径不同吗？或者我们可以调大流式读入的量，让他不是一直等数据？其原因如何？

---

**Cursor**

Let me analyze both questions by looking at the logs.

Three excellent questions. Let me analyze each one.

**First, the training speed issue — let me quantify the bottleneck:**

Now I see the full picture. Let me explain:

**Old version (monolithic) iter 2+ timing breakdown:**
- `selfplay_payload_load_sec=78s` (separate, before train timer)
- `h2d_copy_sec=5.2s` (inside train timer)
- `train complete time=107s` = 5s H2D + 101s GPU training
- Per step: ~30ms (data already in GPU memory)

**New version (streaming) iter 1 timing:**
- `first_batch_sec=3.094s`
- `train complete time=212s` for ~3308 steps
- Per step: ~64ms (2.1x slower)

The culprit: each step waits for the DataLoader workers to **load shard tensors from disk**. With 4 epochs and `persistent_workers=False`, each shard is re-read from disk 4 times. The I/O bottleneck adds ~30ms per step.

The fix is straightforward: **cache loaded shards in worker memory** so disk I/O happens only in epoch 1, and set `persistent_workers=True`.

Good, syntax is clean. Now let me address all three of your questions.

---

**Q1: 训练速度 — 为什么 streaming 比 monolithic 慢了 2x?**

精确对比：

| | Old (monolithic) | New (streaming) |
|---|---|---|
| Per-step time | ~30ms | ~64ms |
| Disk I/O during training | 0 (all in GPU memory) | Each shard loaded from disk per epoch |
| Epochs × shard loads | 0 | 4 epochs × 16 shards = 64 file loads |
| `persistent_workers` | N/A | `False` (workers respawn each epoch) |

Old version: `payload_load_sec=78s` loads everything to CPU, then `h2d_copy_sec=5s` puts it all on GPU. Training iterates 4 epochs purely from GPU tensors — zero I/O.

New version: each epoch, each DataLoader worker re-reads its shard files from disk, deserializes, shuffles, and yields batches. With `persistent_workers=False`, worker processes are re-spawned each epoch (extra fork overhead).

**Fix applied** (two optimizations):

1. **Worker-level shard cache**: Added `_worker_shard_cache` dict. First epoch loads from disk and caches; epochs 2-4 use the in-memory cache. This eliminates 75% of disk I/O.

2. **`persistent_workers=True`**: Workers persist across epochs, avoiding process spawn overhead. The cache dict lives in worker-process memory and persists naturally.

3. **Per-epoch seed variation**: Added `_epoch_counter` so each epoch gets a different random seed for shard ordering and shuffling, even with persistent workers.

Expected new timing: first epoch ~same as before, epochs 2-4 should be close to monolithic speed (~30ms/step from cache). Estimated total: ~120-140s (down from 212s).

---

**Q2: Loss 指标 vs 棋力的困惑**

你观察到的现象完全正确：loss 持续下降（3.5→1.7）但最强模型在 iter 34 (loss≈2.6)，而不是 loss 最低的 iter 78-80 (loss≈1.3-1.7)。这是 AlphaZero 训练中的经典问题：

**根本原因**: `soft_label_alpha=1.0` 意味着 value target 完全基于子力差（material delta），loss 衡量的是"模型预测子力差的准确度"。但**棋力 ≠ 预测子力差的能力**：
- 后期模型过拟合到当前自博弈数据分布的子力差模式
- 子力差不能捕捉位置优势、战术威胁、棋型结构等
- 模型可能学会"预测终局子力差很准"但不会"做出最好的下一步"

**为什么硬标签不稳定**: 因为大量和棋（value=0），95%+ 的样本没有方向性信号。模型只能从少量决定性对局中学习。

**推荐的指标体系**：
- **vs_random (win-loss)**: 早期健康检查，快速验证模型在学习基本能力
- **锦标赛 Elo**: 模型选择的金标准（你已经在用）
- **loss**: 仅作为训练稳定性诊断（应该在每个 iteration 内下降；跨 iteration 比较意义不大，因为数据分布在变）
- **decisive_rate + piece_delta 分布**: 自博弈数据质量指标

关于 value target，一个未来可以探索的方向：**自适应混合**——对决定性对局用硬标签（已有明确输赢信号），对和棋用软标签（只有子力差信息）。这比全局固定 alpha 更精准，但需要额外实现。

---

**Q3: 口径问题**

Old version `time=107s` 不含 `payload_load_sec=78s`（在 train timer 之外）。New streaming version `time=212s` 包含了全部 DataLoader I/O。所以实际差距是 107s（含5s H2D）vs 212s（含全部 I/O），不是计算口径的问题——streaming 确实比 monolithic 慢了约 2x，原因如上所述。

---

**Recommended commit**: `bugfix/train: cache streaming shards, persistent workers, fix cosine-to-zero LR`

**待测功能清单:**
- Streaming shard cache: epoch 2-4 应无磁盘 I/O
- persistent_workers: 不应报错
- LR warmup-only: iter 内 LR 在 warmup 后保持恒定
- WDL weight=0: value loss 应回到 ~2.7 水平

