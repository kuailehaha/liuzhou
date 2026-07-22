# MEMORY

## Current Conclusions (2026-07-22): Apple M5 Portable 20-Hour Training Run

### 1) Goal, Baseline and Current Status

- Current target is an approximately 20-hour portable MPS run on the local MacBook Air, retaining the strongest checkpoint during training and testing whether it can reach at least `99%` raw wins against RandomAgent over 500 games. Draws count as non-wins. This is a `vs_random` health/progress target, not a tournament/Elo strength claim.
- The run starts from the accepted one-hour checkpoint `tmp/v1_portable_goal_20260722/formal_1h/model_iter_117.pt` and its matching `optimizer_state.pt`, rather than discarding the already verified 3,744 games / 373,365 positions. The optimizer-state SHA-256 is `845ab53cefc20e1a6456f2845c1d0a6e6011d7db184d6114f5a91946a265db76`.
- The implementation and smoke evidence are committed as `5f8e0d5e7360eace6d4da47d0aa01cfa0220c0f8`. The full run started at `2026-07-22 15:34:59 UTC+8` (`07:34:59Z`) and has a fixed deadline of `2026-07-23 11:34:59 UTC+8` (`03:34:59Z`). It is active under the one-shot LaunchAgent label `com.liuzhou.portable-mps-20h`, with runtime state in `tmp/v1_portable_long_20h/` and logs in `logs/portable_mps_20h.log` / `.err.log`.
- The seeded initial 500-game baseline completed in `114.64s`: `433-0-67` (`86.60%` raw wins), Wilson 95% `[83.33%, 89.31%]`; challenger black `217-0-33`, white `216-0-34`. MPS resolved without fallback. The run then entered iteration 1 self-play with 128 games/concurrency 128 and 8 simulations. The `>=495/500` target has not yet been reached.

### 2) Local Machine and Frozen M5 Parameters

- Host: MacBook Air `Mac17,3`, Apple M5, 10 CPU cores (4 performance + 6 efficiency), 8 GPU cores and 16 GiB unified memory. Runtime: macOS 26.5.2 arm64, Python 3.10.20, PyTorch 2.9.1, MPS available; CUDA and `v0_core` unavailable.
- Benchmarks were run on AC power with no thermal/performance warning and no swap. Self-play used the same accepted checkpoint, portable MPS and 8 simulations:
  - concurrency 32: `162.805` positions/s, about `0.96` GB maximum RSS;
  - concurrency 64: `186.119` positions/s, about `1.46` GB maximum RSS;
  - concurrency 128: `195.201` positions/s, about `2.29` GB maximum RSS and about `2.40` GB peak footprint;
  - concurrency 256: `197.455` positions/s, about `2.68` GB maximum RSS and about `4.22` GB peak footprint.
- Freeze self-play at 128 games / concurrency 128. It is about `19.9%` faster than concurrency 32; concurrency 256 adds only about `1.2%` throughput while raising peak footprint by about `76%`, which is poor headroom for a fanless closed-lid run.
- A fixed 25,457-position, three-epoch training comparison measured `33.40s` at batch 256, `28.26s` at batch 512 and `25.94s` at batch 1024. Freeze batch size 256 because larger batches reduce optimizer-step count and therefore change the learning experiment, rather than being a semantics-neutral throughput adjustment.
- A same-checkpoint/same-seed 200-game RandomAgent evaluation produced the identical `163-0-37` result at concurrency 64 and 128. Concurrency 64 took `55.83s` with about `1.27` GB peak footprint; concurrency 128 took `54.81s` with about `2.10` GB. Freeze evaluation concurrency at 64 because the `1.8%` speed gain at 128 does not justify roughly `65%` more peak memory.
- Frozen long-run defaults: portable MPS; 128 self-play games and concurrency 128 per iteration; 8 simulations; temperature `1.0 -> 0.1` at ply 10; opening random moves annealed `6 -> 0`; maximum 512 plies; replay window 4; batch 256; 3 epochs; cosine LR `3e-4 -> 5e-5`; weight decay `1e-4`; `soft_label_alpha=0.5`; `policy_draw_weight=1.0`; evaluation concurrency 64.

#### Simulation multiplier sweep

- A same-checkpoint, same-seed self-play sweep used 64 games, concurrency 64, six random opening moves and changed only MCTS simulations:
  - 8 sims: `35.05s`, `179.58 positions/s`, `35.94%` draws, about `1.36` GB peak footprint (`1.00x` cost);
  - 16 sims: `72.39s`, `92.30 positions/s`, `42.19%` draws, about `1.45` GB (`2.07x` cost);
  - 32 sims: `136.14s`, `48.30 positions/s`, `73.44%` draws, about `1.61` GB (`3.88x` cost);
  - 64 sims: `243.04s`, `24.31 positions/s`, `81.25%` draws, about `1.81` GB (`6.93x` cost).
- A fixed-seed 100-game RandomAgent screen was monotonic but too noisy for acceptance: 8/16/32/64 sims produced `87-0-13`, `89-0-11`, `93-0-7` and `98-0-2`, at `1.00x/1.86x/3.21x/6.60x` evaluation time. The apparent `98%` at 64 sims did not generalize to the required sample size.
- The authoritative same-seed 500-game comparison was:
  - 8 sims: `433-0-67`, raw win `86.60%`, Wilson 95% `[83.33%, 89.31%]`, `121.43s`; black `223-0-27`, white `210-0-40`;
  - 64 sims: `449-0-51`, raw win `89.80%`, Wilson 95% `[86.84%, 92.16%]`, `839.49s`; black `230-0-20`, white `219-0-31`.
- Increasing 8 -> 64 therefore bought 16 additional wins / `+3.2` percentage points for `6.91x` wall time and still missed the `495/500` target by 46 wins. Memory remained safe and no thermal warning, swap or fallback occurred; compute/sample efficiency and increasingly draw-heavy self-play are the limiting factors.
- Decision: retain 8 simulations for the 20-hour self-play and periodic 500-game gates. Do not select 32/64 from the optimistic 100-game screen. A higher-simulation training schedule would need an equal-wall-clock training A/B showing that improved targets outweigh the large loss of data volume; this sweep does not establish that.

### 3) Long-Run Orchestration and Evaluation Contract

- `scripts/long_train_portable_mps.py` implements a resumable, deadline-based outer loop with a lock file, paired model/optimizer commit hashes and rollback recovery, rolling replay, per-stage retries, finite/fallback audits, `state.json`, append-only `events.jsonl` and `final_summary.json`.
- `scripts/run_long_train_mps.sh` binds the run to the `torchenv` interpreter and keeps the process awake with `/usr/bin/caffeinate -ims`; display sleep is intentionally not disabled.
- Every 10 iterations, the candidate is evaluated for exactly 500 games against RandomAgent at temperature 0, with a recorded seed and an exact 250/250 challenger black/white split. `best_vs_random.pt` ranks by raw wins, then fewer losses.
- A separate sampled 500-game candidate-versus-incumbent evaluation retains `best_model.pt` only when challenger wins exceed losses. RandomAgent screening and incumbent gating are deliberately separate signals.
- A checkpoint that observes at least 495 wins in 500 RandomAgent games must repeat the result on a second, independent fixed-seed 500-game evaluation before `target_reached=true`. The run also performs a fresh final 500-game evaluation. This avoids treating a post-hoc maximum or a single random sample as confirmed acceptance.
- `scripts/eval_checkpoint.py` now accepts `--seed`, persists the requested/actual seed and reports aggregate plus per-color W/L/D. These additions remove the seed/color-report limitations recorded for the earlier one-hour run; legacy/v0 worker-level reproducibility remains backend-dependent.

### 4) Fresh Smoke and Test Evidence

- A real MPS smoke started from the one-hour checkpoint and optimizer, ran 4 self-play games per iteration, trained one epoch, evaluated every iteration and retained rolling best checkpoints. It first stopped cleanly at iteration 2, then resumed to iteration 3 from the same state. The resumed train loaded optimizer state, used iteration 2 replay, retained the incumbent after a `1-2-1` candidate result and finished at the wall-clock deadline. Self-play and evaluation reported zero MPS/device fallback. Artifacts: `tmp/v1_portable_goal_20260722/long_smoke/`.
- A second transaction-focused smoke ran iteration 1, stopped at `max_iterations`, then resumed to iteration 2. Its state persisted matching model/optimizer SHA-256 values, `optimizer_loaded=true`, the exact primary/replay input paths, zero fallback and zero filtered non-finite samples; no rollback file remained after commit. Artifacts: `tmp/v1_portable_goal_20260722/long_transaction_smoke/`.
- The smoke's 4-game evaluation results are only control-flow evidence. In particular, the final `3-0-1` and other 4-game samples provide no evidence that the 500-game `99%` target has been reached.
- Fresh focused validation:
  - `tests/test_eval_checkpoint_reproducibility.py`: 4 passed;
  - `tests/test_long_train_portable_mps.py`: 11 passed, including interrupted model/optimizer pair recovery, retry reset and replay-window-zero semantics;
  - `tests/v1/test_portable_mcts.py`: 23 passed;
  - combined focused suite: 38 passed in 1.43 seconds on the latest rerun;
  - Python compilation of both modified Python entry points passed;
  - `zsh -n scripts/run_long_train_mps.sh` passed;
  - `git diff --check` passed before this documentation update.
- A 20-game seeded real MPS evaluation produced `17-0-3`, with challenger black `8-0-2` and white `9-0-1`, and persisted seed `4242`. This verifies the new seed/color report path, not model acceptance.
- A 100-game sampled same-checkpoint incumbent-gate benchmark produced `39-32-29`, with exact 50/50 colors, in `61.79s` at concurrency 64 and about `1.26` GB peak footprint. A 500-game gate therefore projects to roughly 5.1 minutes; together with the measured RandomAgent gate, the every-10-iteration evaluation budget is roughly 7.5 minutes.
- Simulation-sweep artifacts are under `tmp/v1_portable_goal_20260722/sim_sweep/`; consolidated report: `summary.json`. The 500-game comparison uses independent seed `2026072400` and exact 250/250 challenger colors.

### 5) Open-Lid Launch and Persistence

- The user explicitly selected open-lid operation because no external display is connected. The Mac must remain open and on AC power; this run makes no closed-lid endurance claim. Apple silicon closed-lid operation still requires power, an external display and an external keyboard/mouse. `caffeinate` does not remove that hardware requirement.
- A direct `nohup` launched from the Codex command environment was immediately reaped, so the formal job uses a one-shot per-user LaunchAgent with `RunAtLoad=true` and `KeepAlive=false`. This survives Codex/network disconnection without relaunching after normal completion. The wrapper's `caffeinate -ims` assertions for idle system sleep, system sleep and disk idle were all observed active.
- Active run identity and monitoring commands:

```bash
launchctl print gui/$(id -u)/com.liuzhou.portable-mps-20h
tail -f logs/portable_mps_20h.log
python -m json.tool tmp/v1_portable_long_20h/state.json
```

### 6) Remaining Verification

- Resume continuity, target-confirmation de-duplication, paired model/optimizer recovery, retry cleanup, replay-window-zero semantics and the portable MCTS focused regressions are now covered by fresh tests and real MPS smoke.
- Re-run the final exact-path diff audit after documentation settles; local V0/CUDA cross-layer tests remain unavailable on this Mac because `v0_core`/CUDA are absent.
- The 20-hour task is active but incomplete. Sustained thermal behavior, final independent 500-game result and the `>=99%` outcome remain unverified. Closed-lid behavior is explicitly out of scope for this run.

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

### 6) Known Limitations at the Time and Follow-Up

- At the time of this one-hour acceptance, `scripts/eval_checkpoint.py` derived its random seed from wall-clock time and did not persist the seed or per-color W/L/D. The later 20-hour preparation work recorded above adds an explicit seed and color breakdown; this does not retroactively make the original 1,000-game sample fully reproducible.
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
