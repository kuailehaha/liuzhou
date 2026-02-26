# MEMORY

## Current Conclusions (2026-02-26)

### 1) Strength Milestone (Large-Scale V1 Run)
- Training log anchor: `logs/big_train_v1_20260223_173954.log`.
- `vs_random` reached a peak of `99.80%` (1000 games).
- Effective self-play signal improved substantially in the same run:
  - early: `decisive_games=6443/522488 (1.23%)`, `draw_rate=98.77%`
  - later: `decisive_games=426988/522488 (81.72%)`, `draw_rate=18.28%`

### 2) Tournament Ground Truth
- Tournament anchors:
  - `logs/v1_tournament_80models.log`
  - `logs/v1_tournament_80models.json`
- 80-model championship winner is `model_iter_032.pt`.
- Final stage (`final_4_to_1`) qualification confirms `model_iter_032.pt` as champion.

### 3) Correlation Check: Tournament vs Random-Agent Metric
- Correlation between tournament strength and `vs_random` **raw win%** is weak.
- Correlation between tournament strength and `vs_random` **win-loss** metric (`win% - loss%`) is also weak in the current 80-model sample.
- Practical implication:
  - keep `vs_random` as a health/probe metric;
  - treat tournament ranking (plus Elo/BT fit) as primary strength signal.

### 4) Engineering Status
- V1 acceleration is stable at high gain relative to v0 (multiple validations in `results/v1_validation_*.json` show ~25x-28x best-case speedup, with some configs near 30x).
- Current training script already includes:
  - opening-random annealing,
  - soft-label annealing,
  - non-finite filtering/guards,
  - wins-over-losses gating.

### 5) Next Priority
- Add LR scheduler in v1 training path to address the "more data but limited strength gain" regime.
- Continue strength-oriented tuning with fixed compute budget and tournament-based evaluation.

## Temporary Conclusions (2026-02-11)

### 1) Root Cause Fixed: Policy Index Misalignment
- Root cause identified: movement action index encoding mismatch between:
  - training side: `src/policy_batch.py`
  - v0 side: `v0/python/move_encoder.py` / `v0` fast legal mask
- Symptom before fix:
  - `action_type=move` index mismatch rate was effectively 100% in sampled checks.
  - Training appeared to optimize loss but failed to improve practical strength.
- Fix applied:
  - movement index is now cell-major: `placement + cell_idx * 4 + dir_idx`.

### 2) Validation Status
- Code-level validation passed.
- Random-state alignment spot checks after fix showed zero mismatch in sampled comparisons.
- Training startup/runtime issues resolved:
  - fixed indentation/runtime break in `src/policy_batch.py`
  - pinned `scripts/toy_train.sh` to project Python:
    `/2023533024/users/zhangmq/condaenvs/naivetorch/bin/python`

### 3) New Training Behavior (log: `logs/train_20260210_165133.log`)
- Relative to old collapse run (`logs/train_20260209_053206.log`), early-to-mid iterations improved clearly:
  - `win rate vs Random` no longer collapsed to near 0%.
  - observed range in current run (early/mid): ~14.5% to ~32%.
- Remaining issue:
  - self-play draw rate still climbs high (~90%+ in mid iterations), indicating strong "avoid-loss" tendency.

### 4) Interpretation of Rising Self-Play Draw Rate
- Not purely good or bad:
  - good: often means less blundering, stronger defensive stability.
  - bad: if too high for too long, it can stall policy improvement ("learn not to lose" > "learn to win").
- Must be judged jointly with:
  - `win rate vs previous iter`
  - `win rate vs Random`
  - draw trend itself

### 5) Scaling Rule (One-Line)
- To scale training effectively at current stage: increase self-play game count first, then simulations, and do retuning last via controlled A/B.

## Scaling Priority (for increasing training volume)

Current recommendation order:

1. Increase self-play game count first.
- Reason: at current stage, diversity/coverage is likely a larger bottleneck than per-position search quality.
- Practical: increase `self_play_games_per_worker` first, keep other knobs stable for A/B clarity.

2. Increase simulations (`mcts_simulations`) second.
- Reason: you already run a relatively high setting (`1024`); further increase costs a lot and may reduce throughput.
- Use this when policy quality plateaus after increasing game volume.

3. Parameter retuning third (controlled A/B only).
- Suggested order:
  - soft-label schedule floor (avoid annealing to pure hard too early)
  - exploration-related knobs
  - resign behavior
- Change one group at a time and compare against a fixed baseline window.

## Short A/B Plan Template
- Baseline: current `toy_train.sh` settings.
- A/B-1 (priority): increase self-play games only.
- A/B-2: keep A/B-1, then slightly increase simulations.
- Compare on the same horizon:
  - `vs previous` win rate
  - `vs Random` win rate
  - self-play draw rate
  - wall-clock efficiency (games/hour, positions/hour)
