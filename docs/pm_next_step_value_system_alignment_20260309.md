# PM Next Step: Value System Accuracy Alignment

Date: 2026-03-09
Owner: PM (project-wide)
Scope: v1 pipeline only

## 1) Goal and non-goals

Goal:
- Improve value-head usefulness for win-rate estimation without increasing self-play games or MCTS simulations.
- Keep current model architecture and training entrypoints stable for this iteration.

Non-goals:
- No rule changes.
- No action encoding changes.
- No large refactor of MCTS or model class in this iteration.

## 2) Facts from current code baseline

- Current network has one value head that outputs bucket logits (default 101 bins).
- Training value loss previously used only scalar-to-bucket two-hot supervision.
- Soft label mixing and anti-draw shaping can make outcome semantics harder to interpret as calibrated WDL.

## 3) Implemented in this iteration

File changed:
- `v1/python/train_bridge.py`

Changes:
- Added a bucket-to-WDL auxiliary loss term in both train paths:
  - `train_network_from_tensors`
  - `train_network_streaming`
- Kept architecture unchanged (still one bucket head).
- Added epoch metrics for decomposition:
  - `avg_value_bucket_loss`
  - `avg_wdl_aux_loss`
- Added run metadata:
  - `wdl_aux_loss_weight`

Design choice:
- No new CLI switches.
- Fixed internal weight: `WDL_AUX_LOSS_WEIGHT = 0.25`.

## 4) Team handoff and interface contract

Algorithm owner:
- Validate whether `avg_wdl_aux_loss` decreases together with stable `avg_value_bucket_loss`.
- Confirm no regression in `vs_previous` gating criterion (`wins > losses`).

Training owner:
- Ensure stage metrics JSON persists new fields from `train_bridge` outputs.
- Compare trend on same compute budget (no extra games/sims).

Evaluation owner:
- Extend offline analysis to track:
  - value-loss decomposition (`bucket` vs `wdl_aux`)
  - `vs_random` win-loss probe trend
  - self-play draw rate and decisive ratio

PM checkpoint:
- Decision point after 2-3 iterations on same budget:
  - Keep weight 0.25 if trends improve.
  - Move to next design step only if draw/internal metrics still plateau.

## 5) Invariants

- Keep zero-sum sign convention unchanged.
- Keep DDP collective alignment unchanged.
- Keep staged API behavior unchanged.

## 6) Risks and rollback

Risks:
- Auxiliary objective may over-regularize value bucket fitting.
- Metric decomposition may expose divergence under high draw distributions.

Rollback:
- Remove auxiliary term by reverting this single file change.
- No checkpoint format migration needed.

## 7) Verification checklist

- Static: `python -m py_compile v1/python/train_bridge.py`
- Runtime smoke:
  - single-device train path runs and returns new metric keys.
  - streaming path runs and returns new metric keys.
- DDP smoke:
  - no new collective mismatch introduced.

