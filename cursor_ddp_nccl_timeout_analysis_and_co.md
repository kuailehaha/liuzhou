# DDP/NCCL timeout analysis and code modification plan
_Exported on 3/9/2026 at 22:40:15 GMT+8 from Cursor (2.6.11)_

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

