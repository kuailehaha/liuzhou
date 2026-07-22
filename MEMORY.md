# MEMORY

## Current Conclusions (2026-07-22): Portable MPS One-Hour Smoke Acceptance

### 1) Scope and Verdict

- Evaluated commit `b5eed8ba2a2cff44b0716a10cb379a1e7de52353` (`feature&selfplay: add portable CPU/MPS full-MCTS reference pipeline`) on a MacBook Air with Apple M5, 16 GiB RAM, Python 3.10.20, PyTorch 2.9.1 and MPS. CUDA and `v0_core` were unavailable.
- Under the frozen portable smoke protocol below, the final one-hour checkpoint beat RandomAgent in `859/1000` games, with `0` losses and `141` draws. Raw win rate was `85.90%`, exceeding the strict `>80%` acceptance threshold by `5.90` percentage points.
- The Wilson 95% interval for wins over all games was `[83.61%, 87.92%]`; its lower bound also exceeded 80%.
- This establishes that the new portable pipeline can cross the requested `vs_random` smoke threshold under an appropriate one-hour MPS configuration. It remains a health/training-progress result, not tournament/Elo evidence of general model strength.

### 2) Pilot, Tuning and Frozen Configuration

- The near-untrained pilot checkpoint scored `7-16-77` against RandomAgent over 100 games. A 10-iteration calibration run improved to `63-0-37` over 100 games; iteration 5 had scored `55-0-45`.
- Early full-game self-play was draw-heavy, so the formal run retained mixed hard/soft value supervision with `soft_label_alpha=0.5` rather than relying on initially sparse hard W/D/L targets alone.
- The formal run started from scratch after calibration and froze the following configuration before launch; no parameter was changed during the hour and no intermediate checkpoint was selected after seeing the final result:
  - portable full-MCTS backend on MPS, single process;
  - 32 self-play games per iteration, concurrency 32 and 6 random opening moves;
  - 8 MCTS simulations, temperature `1.0 -> 0.1` at ply 10 and maximum 512 plies;
  - batch size 256, 3 epochs, learning rate `3e-4`, weight decay `1e-4`;
  - `soft_label_alpha=0.5`, `policy_draw_weight=1.0`;
  - model initialization seed `20260722`, self-play iteration seeds `1..117`;
  - explicit staged `selfplay` and `train` calls with checkpoint reload and persistent optimizer state.

### 3) Formal Training Evidence

- Wall-clock window: UTC `2026-07-22T02:49:22Z` to `2026-07-22T03:49:27Z` (Beijing time `10:49:22` to `11:49:27`), `3605` seconds total.
- Completed 117 full self-play/train iterations, producing 3,744 games and 373,365 positions.
- Aggregate self-play outcomes were 1,478 black wins, 856 white wins and 1,410 draws; decisive-game ratio was `62.34%`, with `63.13%` over the final 10 iterations.
- Aggregate measured self-play time was `2383.21s` at `156.66` positions/s; measured training time was `700.54s` over 4,548 optimizer steps.
- Average loss moved from `4.8333` on iteration 1 to `1.6666` on iteration 117; the final-10 mean was `1.9758`.
- Optimizer state loaded successfully on all 116 continuation iterations. Device/MCTS fallbacks, non-finite targets, non-finite checkpoint tensors and filtered non-finite samples were all zero.
- Final checkpoint: `tmp/v1_portable_goal_20260722/formal_1h/model_iter_117.pt`, SHA-256 `5b4cb05c12f4ccde36653873d124f85646160213130adf0e9a9ffacdde208c18`. Its 156 state tensors (3,040,446 elements) were finite.

### 4) Independent 1000-Game Acceptance

- Evaluated the final checkpoint, not a post-hoc best checkpoint, with portable MPS, 8 simulations, temperature 0, no opening randomization, concurrency 64 and one worker.
- The challenger played exactly 500 games as black and 500 as white. Aggregate result: `859-0-141` over 1,000 games (`85.90%` win, `0.00%` loss, `14.10%` draw).
- Evaluation took `240.06s` with zero device fallback. Draws counted as non-wins for the strict win-rate threshold.
- Evaluation report: `tmp/v1_portable_goal_20260722/formal_1h/eval_final_1000.json`, SHA-256 `7201512b7a256811b8a0452cd71957630cf824cc4dace338909f11ef4c553ad4`.

### 5) Verification and Artifact Anchors

- `tests/v1/test_portable_mcts.py`: 23 tests passed in 1.57 seconds on the same environment.
- The portable smoke checked CPU/MPS legal-mask and output parity, finite values, checkpoint save/reload and explicit fallback rejection before the formal run.
- Consolidated local report: `tmp/v1_portable_goal_20260722/formal_1h/formal_summary.json`.
- Exact staged run protocol: `tmp/v1_portable_goal_20260722/formal_1h/run_formal_1h.txt`.
- The retained ignored experiment directory is about 1.4 GiB and contains all 117 checkpoints plus paired self-play/train JSON metrics.

### 6) Known Limitations and Follow-Up

- `scripts/eval_checkpoint.py` currently derives its random seed from wall-clock time and does not persist the seed in its report. It balances challenger colors but reports only aggregate W/L/D, not per-color W/L/D.
- Repeated staged invocations write checkpoint-internal `iteration=1`; the true external iteration is currently recoverable from checkpoint filenames and paired JSON files. Future long staged loops should write explicit external-iteration metadata.
- Local collection of the V1 tensor-pipeline/V0 action tests was blocked by missing `v0_core`; no CPU/CUDA or portable/CUDA cross-layer parity claim follows from this Mac run.
- Eight simulations is a constrained portable-Mac smoke budget and is not comparable to the production H20/CUDA search configuration.
- Continue to use `vs_random` only for health and coarse progress. Any strength claim still requires fixed-opponent head-to-head, tournament/Elo/BT, or the agreed gating protocol under controlled compute.

## Current Conclusions (2026-07-21): Root-PUCT Audit and Portable CPU/MPS Direction

### 1) Confirmed Root-PUCT Design Facts
- Current V1 default `SPARSE_PLY=1` is a root-only policy-improvement operator, not full MCTS:
  - enumerate and evaluate every legal one-step child once;
  - convert each child value to the root player's perspective;
  - repeatedly allocate visits using fixed `P_a` and fixed `q_a`;
  - after the first visit of action `a`, `Q_a = q_a` remains unchanged.
- The fused CUDA implementation parallelizes action scoring within each root, while the simulation loop for that root advances sequentially inside one CUDA block. Its persistent `N/W` statistics are not a semaphore.
- The semaphore-like mechanism remembered from parallel MCTS is **virtual loss**: a temporary reservation applied while a selected leaf is awaiting evaluation, then reverted before the real value is backed up. Current fused Root-PUCT has no in-flight leaf batch within one root and therefore does not use virtual loss.
- Git history and `v1/Design.md` describe Root-PUCT as a GPU migration baseline chosen for batching/throughput. Full GPU tree selection, expansion and backup were explicitly left as a later milestone. Fixed one-step `q` is therefore a performance-driven algorithmic simplification, not an unavoidable GPU requirement.

### 2) Training-Dynamics Risk: Strong Hypothesis, Not Yet a Proven Root Cause
- Root-only fixed-`q` search cannot use deeper evidence to correct a wrong child-value ranking.
- Current `scripts/big_train_v1.sh` defaults to `PROFILE=aggressive`, `MCTS_SIMULATIONS=65536`, and late self-play temperature `0.1`. With fixed `q`, this combination can turn a small value-ranking difference into an almost one-hot policy target `π`.
- Likely amplification chain:
  - small value error changes the top-ranked child;
  - root-only search cannot correct it;
  - very large visit allocation and low temperature sharpen the selected action;
  - policy training copies the overconfident target;
  - the next self-play distribution becomes more biased.
- This can plausibly explain part of the observed sensitivity where small parameter changes produce very different learning trajectories. It does **not** by itself explain numerical failures such as NaN/Inf, and it has not yet been isolated by a fixed-model search A/B.
- Other established contributors remain separate: historical action-index misalignment, draw-dominated data, value-target/objective choice, learning rate/warmup, replay window, initialization and optimizer continuity.
- Deeper search cannot repair an objective mismatch. If value primarily represents a material proxy, deeper search may optimize that proxy more effectively without improving true W/D/L.

### 3) Existing `SPARSE_PLY>1` Is Experimental and Must Stay Out of Formal Training
- Static audit of `v1/python/mcts_gpu.py::_refine_via_topk_lookahead()` found that it is not a semantics-complete multi-ply search:
  - repeated refinements restart from root actions instead of recursively advancing the previous frontier;
  - unconditional max pooling does not model opponent choice or same-player consecutive phases correctly;
  - `max(shallow, deeper)` prevents deeper negative evidence from lowering an overestimated action;
  - inner top-K selection ignores inner policy priors/visit statistics;
  - partial terminal/no-legal rows may lose `valid_indices` alignment before reshape.
- Do not enable `SPARSE_PLY=2/3` for production training until perspective, recursion, terminal-row mapping and backup semantics have dedicated tests and fixed-state parity evidence.

### 4) Chosen Portable Reference Architecture
- The next correctness-first implementation should be a **portable V1 reference backend**, not a CUDA rewrite:
  - CPU owns rules, tree nodes, selection, expansion bookkeeping and backup;
  - PyTorch owns batched network inference and training;
  - device can be `cpu` or Apple Silicon `mps`;
  - portable path must not require `v0_core`, CUDA Graph, NVTX, CUDA events, NCCL, DataParallel or DDP.
- On Apple Silicon, the intended split is CPU search/control plus MPS model inference/training. First acceptance is single-process/single-device correctness; multiprocessing and performance optimization come later.
- The existing CUDA V1 path remains intact as the production throughput backend. Portable work is additive and provides a trustworthy search/semantics baseline for later GPU optimization.

### 5) Search Semantics That Must Be Preserved
- The network value is from the state `current_player` perspective.
- During backup, flip sign only when `current_player` changes across an edge; same-player mark/removal/capture phases must keep the sign.
- Terminal utility, trajectory target and search backup must use the same W/D/L and soft-value semantics.
- A deeper evaluation must be allowed to raise or lower a shallow estimate; never use optimistic-only `max(shallow, deeper)` as the general backup rule.
- Root policy target remains the temperature-scaled visit distribution in the shared 220-dimensional action encoding, with illegal actions exactly zero.

### 6) Evidence Required Before Claiming Training Improvement
- First run fixed-network, no-training comparisons: root-only simulation sweep, tactical phase cases, policy entropy/KL/top-1 stability and reference-search versus root-only head-to-head.
- Existing V0 `MCTSCore` cannot be treated as unquestioned ground truth until its per-edge sign backup is checked against consecutive same-player phases.
- Only after the portable search is stronger with the same checkpoint should short training A/B begin.
- Final comparison must report both equal-sample and equal-wall-clock/GPU-budget results. Tournament/Elo or fixed-opponent head-to-head is the strength criterion; `vs_random` and draw ratio remain health/data-distribution probes.

### 7) Portable Backend Status (implemented 2026-07-21)

- `portable` now provides the CPU rule/tree/full-MCTS reference with batched PyTorch CPU/MPS inference, persistent subtree reuse and current-player-aware backup. Existing `cuda_root` remains the default.
- Mac CPU and MPS toy workflows pass through self-play, float32 training, evaluation, checkpoint save and CPU/MPS reload with no fallback or non-finite output.
- The fixed-model search sweep confirms measurable fixed-q sensitivity at 65,536 allocations, but the available random/toy weights and two drawn head-to-head games do not establish stronger play. No training-improvement claim or short training A/B should follow until representative checkpoints or labeled tactical cases are available.
- During a draw-heavy, pre-saturation phase, fixed-condition `vs_random win-loss` plus draw rate can serve as the primary coarse progress/screening signal. Treat it as an interim proxy, not final strength ground truth; switch back to fixed-checkpoint/tournament evidence as it saturates.
- Reproducible commands and measured details are recorded in `v1/Design.md`; pending representative-checkpoint and training experiments remain in `TODO.md`.

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
