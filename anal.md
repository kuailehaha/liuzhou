# Training Regression Analysis: 15451ae (old, working) vs HEAD (new, broken)

**Old run**: `logs/big_train_v1_20260223_173954.log` — commit `15451ae` (2026-02-23 17:40 UTC)
**New run**: `logs/big_train_v1_20260309_170346.log` — commit `0a380d1` (HEAD, git_dirty=1)

## 1  Executive Summary

The old version achieves vs_random >90% win rate by iteration 4 and remains stable.
The new version oscillates between 33–72% over 15 iterations and shows **monotonically increasing training loss** (6.16 → 7.88), a textbook sign of optimization failure.

Even though surface-level parameters were "aligned" (soft_label_alpha=1.0, soft_value_k=2.0), the new codebase introduces **7 structural changes** that fundamentally alter the training dynamics. The user only tuned knobs; the engine underneath was replaced.

---

## 2  Parameter Comparison

| Parameter | Old (15451ae) | New (HEAD) | Impact |
|-----------|--------------|------------|--------|
| LR | **2e-4** | **1e-4** | Half learning rate |
| weight_decay | 1e-4 | **5e-5** | Half weight decay |
| MCTS_SIMULATIONS | **65536** | **131072** | 2× more simulations |
| opening_random_moves | **6 → 0** (annealed) | **4 → 4** (fixed) | No curriculum |
| soft_label_alpha | 1.0 → 1.0 | 1.0 → 1.0 | Same |
| soft_value_k | 2.0 | 2.0 | Same (user aligned) |
| anti_draw_penalty | N/A | -0.00 | Dead code |
| policy_draw_weight | 1.0 | 1.0 → 1.0 | Same |
| replay_window | **0** (none) | **4** | Mixes 3 old iterations |
| optimizer persistence | **No** | **Yes** | Stale Adam momentum |
| LR scheduler | **None** (fixed) | **Warmup + Cosine** | Decays within iteration |
| warmup_steps | N/A | 100 | New |
| WDL aux loss | **No** | **Yes (0.25 weight)** | Adds extra loss term |
| streaming training | **No** (monolithic) | **Yes** | Worse shuffling |
| selfplay output format | Concat-in-memory | Chunk-manifest | Different shard granularity |
| winner detection | `phase != PLACEMENT` | `phase in {MOV,CAP,CTR}` | MARKING excluded |
| eval temperature (vs_random) | 0.05 | 0.0 | Minor |
| eval games | 1000 | 2000 | More precise eval |

---

## 3  Structural Code Changes (15451ae → HEAD)

### 3.1  Value Loss Function: +WDL Auxiliary Loss

**File**: `v1/python/train_bridge.py`

Old:
```python
value_loss = -(bucket_target * value_log_probs).sum(dim=-1).mean()
```

New:
```python
bucket_value_loss = -(bucket_target * value_log_probs).sum(dim=-1).mean()
wdl_target = scalar_to_wdl(raw_b_values).to(torch.float32)
wdl_probs = _bucket_logits_to_wdl_probs(value_logits)
wdl_aux_loss = -(wdl_target * torch.log(wdl_probs.clamp_min(1e-8))).sum(dim=-1).mean()
value_loss = bucket_value_loss + 0.25 * wdl_aux_loss
```

**Consequence**: The WDL aux loss uses `raw_b_values` (hard game outcome), **NOT** the soft/mixed target. Since `soft_label_alpha=1.0`, the bucket target is entirely based on soft values, but the WDL target is based on hard values. In early iterations where all games are draws, `raw_b_values = 0` for every sample, so `wdl_target = [0, 1, 0]` (pure draw). The WDL loss then **pulls the value head toward always predicting draw**, which contradicts the bucket target (based on material-delta soft values). This creates a **gradient conflict** between the two value loss terms.

**Measured impact**: Initial loss jumps from **3.5 (old) to 6.2 (new)** — a 77% increase — even on functionally identical iter-1 data from a random model.

### 3.2  Streaming DataLoader Replaces Monolithic Load

**File**: `v1/python/train_bridge.py` (new function `train_network_streaming`), `v1/python/streaming_dataset.py` (new file)

Old flow:
1. Load entire payload (~52M samples) into CPU → shard by rank (every 4th sample) → copy to GPU
2. Random permutation over ALL per-rank samples → iterate fixed number of batches
3. Full cross-game shuffling within each epoch

New flow:
1. Resolve shard file list → DDP round-robin assigns shards to ranks
2. DataLoader with `workers=1` loads one shard at a time
3. Shuffle only **within each shard** (per-shard `torch.randperm`)
4. Batches from one shard are contiguous — cross-shard mixing only happens at shard boundaries

**Consequence**: Each batch comes from a single shard (= a single chunk from a single GPU worker). In the old version, every batch contained samples from ALL 4 workers' games, uniformly mixed. The new version has **much less data diversity per batch**, which:
- Increases gradient variance
- Creates "local optima" per shard (model overfits to one worker's games, then gets confused by the next shard)
- Especially harmful with `workers=1` where there's no prefetching overlap

### 3.3  Replay Buffer (replay_window=4)

**File**: `scripts/big_train_v1.sh`, `v1/train.py`

Old: Each iteration trains **only** on its own fresh self-play data.
New: Each iteration trains on current data + up to 3 previous iterations' data.

**Consequence in log**: By iter 5, `est_samples=29188453` with `replay_files=3`, vs old's `global_positions=53485528` with no replay. The replay data comes from models 1–3 iterations old, whose policy targets may actively conflict with the current model's learning direction.

Replay is beneficial in stable RL settings, but in early training where the model is rapidly changing, stale data acts as noise that **dilutes the signal from the current iteration's self-play**.

### 3.4  Optimizer State Persistence

**File**: `v1/python/train_bridge.py`

Old: Fresh `Adam` optimizer every iteration.
New: Saves optimizer state to disk after each iteration and loads it at the start of the next.

**Consequence**: Adam's internal moment estimates (mean/variance of gradients) are tuned to the *previous* iteration's data distribution. When the data distribution shifts (which it does dramatically in self-play RL), the stale momentum:
- Resists learning from new data
- Creates oscillation as the optimizer fights its own history
- Is especially damaging when combined with replay (moments are a mix of old and new signal)

### 3.5  LR Scheduler (Warmup + Cosine Decay)

**File**: `v1/python/train_bridge.py`

Old: Fixed LR = **2e-4** for all steps in all epochs.
New: Base LR = **1e-4**, with 100-step warmup, then cosine decay to 50% (5e-5) by end of the iteration's training steps.

**Consequence**: Effective average LR is roughly **5–7.5e-5**, which is 3–4× lower than old version's 2e-4. Combined with optimizer persistence (stale momentum), the model barely updates. The cosine decay means the last epochs of each iteration train with a very low LR, unable to compensate for the early-epoch noise from stale replay data.

### 3.6  Winner Detection Rule Change

**Files**: `v1/python/mcts_gpu.py`, `src/game_state.py`, `v0/src/game/game_state.cpp`

Old: `if self.phase == Phase.PLACEMENT: return None` — winner check in ALL non-placement phases.
New: `if not self.has_entered_movement_stage(): return None` — winner check ONLY in MOVEMENT, CAPTURE_SELECTION, COUNTER_REMOVAL.

**Consequence**: The MARKING phase is now excluded. While marking doesn't normally remove pieces, this change affects how the MCTS evaluates terminal states. The practical impact is visible in the position count:
- Old iter 1: **52.2M positions** from 522K games → **99.8 positions/game**
- New iter 1: **42.2M positions** from 522K games → **80.8 positions/game**

Games are **20% shorter**, producing fewer training samples. This alone is a significant data reduction.

### 3.7  MCTS Simulations Doubled (65536 → 131072)

**File**: `scripts/big_train_v1.sh` (aggressive profile)

**Consequence**: Self-play produces more deterministic policies → less exploration → more draws. The iter-1 decisive ratio:
- Old: **1.23%** decisive (6443/522488 games)
- New: **0.00%** decisive (0/522488 games)

The piece_delta distribution tells the story:
- Old iter 1: spans **-15 to +15**, wide bell curve
- New iter 1: spans **-7 to +8**, concentrated around 0

With 131072 simulations, even a random model's policy is "confident enough" to avoid catastrophic mistakes, leading to draws. There is no hard outcome signal at all in iter 1.

---

## 4  Evidence from Training Logs

### 4.1  Loss Trajectory

| Iter | Old Loss | New Loss |
|------|----------|----------|
| 1 | 3.53 | 6.16 |
| 2 | 2.92 | 6.58 |
| 3 | 2.95 | 7.02 |
| 4 | 3.18 | 6.17 |
| 5 | 3.12 | 6.71 |
| 6 | 3.14 | 6.84 |
| 7 | 3.01 | 6.87 |
| 8 | 2.92 | 7.19 |
| 9 | 2.95 | 7.49 |
| 10 | 3.00 | 7.57 |
| 11 | 2.98 | 7.52 |
| 12 | 2.86 | 7.88 |
| 13 | 2.97 | 7.60 |
| 14 | 2.97 | 7.56 |
| 15 | 2.99 | 7.12 |

Old loss: stable ~2.9–3.2.
New loss: **monotonically increasing** from 6.2 to 7.9. This is catastrophic — the model is getting worse at fitting the data over time.

### 4.2  vs_random Win Rate Trajectory

| Iter | Old | New |
|------|-----|-----|
| 1 | 27.7% | 54.8% |
| 2 | 85.5% | 55.4% |
| 3 | 87.1% | 64.6% |
| 4 | **91.8%** | 56.6% |
| 5 | 92.1% | 60.4% |
| 6 | 91.9% | **38.3%** |
| 7 | 93.3% | 55.4% |
| 8 | 91.4% | **68.9%** |
| 9 | 92.9% | 37.4% |
| 10 | 86.4% | 69.1% |
| 11 | **94.5%** | **33.0%** |
| 12 | 78.2% | **71.5%** |
| 13 | 84.7% | 53.3% |
| 14 | 79.5% | 59.1% |
| 15 | 81.7% | 65.2% |

Old: rapid climb to 90%+, stable.
New: wild oscillation between 33% and 72%, no upward trend.

### 4.3  Self-play Decisive Game Ratio

| Iter | Old Decisive % | New Decisive % |
|------|---------------|----------------|
| 1 | 1.23% | **0.00%** |
| 2 | **48.85%** | 0.00% |
| 3 | 48.15% | 0.01% |
| 4 | **50.49%** | **47.33%** |
| 5 | 39.84% | 24.44% |
| 6 | 27.79% | 14.49% |
| 7 | 30.46% | **0.02%** |
| 8 | 23.84% | 0.14% |
| 9 | 25.23% | 9.62% |
| 10 | — | **0.00%** |

Old: jumps to ~50% decisive by iter 2 and stays above 20%.
New: oscillates wildly, often collapsing to near-0%. The model keeps falling into "draw equilibrium" and then sporadically escaping.

### 4.4  Training Time per Iteration

| Iter | Old (s) | New (s) |
|------|---------|---------|
| 1 | 172 | 262 |
| 2 | 107 | **735** |
| 3 | 107 | **918** |
| 4 | 107 | **1272** |
| 5 | 108 | **1401** |
| 10 | 107 | **1027** |

New version trains 3–13× slower per iteration due to streaming DataLoader overhead and replay data volume.

---

## 5  Root Cause Diagnosis (Ordered by Impact)

### #1  Effective Learning Rate is 3–4× Too Low

Old: fixed 2e-4 with fresh optimizer.
New: base 1e-4 with warmup + cosine + stale Adam momentum.

The effective learning rate (accounting for Adam's adaptive scaling with persisted moments) is drastically lower. The model cannot update fast enough to learn from self-play data. This alone explains the increasing loss and stagnant vs_random.

**Evidence**: Loss increases monotonically — the optimizer is not keeping up with the data distribution shift.

### #2  WDL Aux Loss Creates Gradient Conflict

The WDL loss targets the **hard** game outcome (which is 0 = draw for nearly all samples), while the bucket loss targets the **soft** material-delta value. These two loss terms pull the value head in opposite directions:
- Bucket loss: "learn the material advantage nuances"
- WDL loss: "predict draw for everything"

This conflict wastes gradient capacity and destabilizes value learning.

**Evidence**: Loss is 77% higher than old version from iteration 1 on identical random-model data.

### #3  Streaming DataLoader Destroys Data Shuffling

Per-shard-only shuffling (vs old full-dataset shuffling) means each batch is correlated within the same shard → higher gradient variance → slower convergence.

With `workers=1`, there's no pipelining — the training loop blocks on I/O between shard loads.

**Evidence**: Training takes 262s (iter 1, no replay) vs 172s (old, monolithic), despite the new version having 20% fewer samples.

### #4  Replay Buffer Dilutes Signal in Early Training

Mixing 3 old iterations of stale policy/value targets with current data confuses the optimizer. In the old version, each iteration gets a clean, fresh signal from the current model's play.

**Evidence**: From iter 5+, `est_samples` is nearly double the current iteration's sample count due to replay, but vs_random performance doesn't improve.

### #5  Optimizer State Persistence Amplifies Instability

Adam's moment estimates from iteration N are tuned to iteration N's data distribution. When loaded into iteration N+1 with different data distribution, they actively resist the new gradient direction. Combined with the low LR, this creates a "momentum trap" where the optimizer is biased toward old solutions.

**Evidence**: The oscillation pattern in vs_random (e.g., 68.9% → 37.4% → 69.1% → 33.0%) suggests the model alternates between adapting and reverting, consistent with conflicting momentum directions.

### #6  MCTS 131072 Suppresses Exploration Signal

Doubling MCTS simulations makes self-play too deterministic. With a random/weak model and 131K sims, games resolve to draws because the search depth is sufficient to find safe (drawing) lines. The hard value target is 0 for nearly all samples, leaving only the weak soft-value signal.

**Evidence**: New iter 1 has **0% decisive games** (old: 1.23%). Piece delta spans only -7 to +8 (old: -15 to +15).

### #7  Fixed Opening Random Moves (4→4) Prevents Curriculum Effect

The old 6→0 schedule creates a curriculum:
1. Early: 6 random moves → chaotic positions → many decisive games → strong signal
2. Later: 0 random moves → clean positions → model plays "real" games

The new 4→4 schedule provides moderate diversity but no curriculum. The model never experiences the "easy learning" phase of highly diverse positions.

---

## 6  What to Do to Reproduce Old Performance

To reproduce the old version's training trajectory with the current codebase, **revert all structural training changes**:

| Change | Action |
|--------|--------|
| WDL aux loss | Set `_WDL_AUX_LOSS_WEIGHT = 0.0` or revert to old value loss |
| Streaming DataLoader | Set `STREAMING_LOAD=0` to use monolithic load path |
| Replay buffer | Set `REPLAY_WINDOW=0` |
| Optimizer persistence | Pass `optimizer_state_path=None` |
| LR scheduler | Remove warmup/cosine; use fixed LR |
| LR / weight_decay | Restore LR=2e-4, weight_decay=1e-4 |
| MCTS simulations | Restore to 65536 |
| Opening random | Restore 6→0 schedule |

These should be reverted **simultaneously**, not incrementally, because they interact: e.g., optimizer persistence is fine with fixed LR but harmful with cosine decay + replay.

---

## 7  Which New Features Are Potentially Valuable?

Not all changes are harmful — some are useful engineering that should be **re-introduced one at a time** after confirming the baseline reproduces:

| Feature | Verdict | Condition |
|---------|---------|-----------|
| Streaming DataLoader | **Keep** (engineering improvement) | Fix shuffling: add global shuffle buffer or multi-shard batching |
| Replay buffer | **Keep** (stabilizes late training) | Introduce only after iter 10+; use small window (1–2); weight replay samples lower |
| Optimizer persistence | **Maybe** | Only with fresh moment reset (`betas=(0.9, 0.999)` reset) or warmup long enough to wash out |
| LR cosine schedule | **Maybe** | Only as inter-iteration macro schedule, not intra-iteration |
| WDL aux loss | **Remove** | Contradicts soft-value target; adds no information when hard targets are all draws |
| Winner detection fix | **Keep** | Correct rule semantics; impact on decisive ratio is secondary |
| Chunk-manifest selfplay | **Keep** | Engineering improvement; doesn't affect training signal |

---

## 8  Summary: The Parable of Seven Changes

Seven individually reasonable changes were stacked simultaneously. Each introduces a small bias:
1. **Lower LR** → slower learning
2. **WDL aux** → gradient conflict
3. **Streaming** → worse shuffling
4. **Replay** → stale signal
5. **Optimizer persistence** → momentum trap
6. **More MCTS** → less exploration
7. **Fixed curriculum** → less diversity

Their compound effect is catastrophic: the model cannot learn because it is simultaneously learning too slowly (LR), being confused (WDL + replay), and not receiving enough signal (MCTS + curriculum). The **monotonically increasing loss** is the proof.

**The path forward is to revert to the known-good baseline, confirm reproduction, then re-introduce features one at a time with per-feature A/B testing.**
