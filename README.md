# Rules

This document records the current rule set after the refactor that split every action into atomic phases.  The information below supersedes older descriptions that bundled multiple decisions into a single move.

## Game Overview

* Board: 6 × 6 grid (36 intersections).
* Pieces: two colours, `BLACK` and `WHITE`, each with 18 stones at the start of the game.
* Goal: remove all opponent stones through captures.  If neither side can achieve this within the move limit the game is declared a draw.

## Phase Structure (Atomic Actions)

Every turn the game state is in exactly one of the phases listed below.  Phases continue until no pending sub-task remains, then control passes to the next player or the next phase.

### 1. `PLACEMENT`
* The active player places one stone on an empty intersection that is not currently marked by the opponent.
* After the placement we check whether the newly placed stone (if it was not already marked) forms a **square** (2×2 block) or a **line** (six stones in a row or column ignoring marked stones).
  * Square → set a pending task to mark **1** opponent stone.
  * Line → set a pending task to mark **2** opponent stones.
* If pending marks exist, move to `MARK_SELECTION`; otherwise, if the board is full enter `REMOVAL`, and if not, hand the turn to the opponent and remain in `PLACEMENT`.

### 2. `MARK_SELECTION`
* Resolve the pending marks one stone at a time.
* The active player chooses a legal opponent stone for each outstanding mark.  Legal targets are opponent stones that are not already marked and, when normal (unmarked) targets exist, are not part of the opponent’s current square/line structures.
* Once all pending marks are resolved:  
  * If the board is full → `REMOVAL`.  
  * Otherwise switch player and return to `PLACEMENT`.

### 3. `REMOVAL`
* If at least one stone is marked, remove **all** marked stones from both colours and enter `MOVEMENT` with `WHITE` to play.
* If no stones were marked (board filled without triggering marks) we enter the **forced removal sequence** (`FORCED_REMOVAL`).  This sequence always starts with `WHITE` removing a `BLACK` stone that is not part of a square/line, followed by `BLACK` removing a `WHITE` stone under the same restriction, after which the game continues in `MOVEMENT` with `WHITE` to play.

### 4. `MOVEMENT`
* The active player moves one of their stones orthogonally by one step into an adjacent empty intersection.
* After each move we check for newly formed square/line structures exactly as in the placement phase:
  * Square → set pending captures = 1.
  * Line → set pending captures = 2.
* If captures are pending → move to `CAPTURE_SELECTION`; otherwise switch player and stay in `MOVEMENT`.
* **No-move rule**: if a player has no legal moves, they may remove one opponent stone that is not part of a square/line (or any opponent stone if none are “normal”).  This is handled by the special action `no_moves_remove`, which then passes play to `CAPTURE_SELECTION` for the opponent (counter removal).

### 5. `CAPTURE_SELECTION`
* Resolve pending captures one stone at a time.
* Legal targets follow the same priority as marking: prefer opponent stones outside of squares/lines when possible.  Each captured stone is removed immediately.
* If the opponent now has zero stones, the current player wins.  Otherwise, when all pending captures are completed, switch player and return to `MOVEMENT`.

### 6. `COUNTER_REMOVAL`
* Triggered only after a `no_moves_remove` action.
* The opponent (the player that was previously stuck) removes one stone belonging to the player who just gained a free capture, obeying the same legality checks as above.
* After the counter removal, if the active player still has at least one stone the game returns to `MOVEMENT`, handing the turn back.

## End-of-Game Conditions

* **Capture victory**: as soon as a player’s capture (including counter removal) removes the opponent’s final stone, that player wins immediately.
* **Move limit draw**: self-play and training scripts enforce a hard cap of 200 moves.  Reaching the cap without a capture victory results in a draw (zero value for both players).  This limit is intended to prevent endless loops in self-play; adjust it if different behaviour is desired.
* **No legal actions**: if `generate_all_legal_moves` returns an empty list for the side to move, that player loses (this case usually aligns with having no pieces left or no possible placement during early phases).

## Additional Notes

* Marked stones remain on the board throughout phase 1 and are only removed in the subsequent `REMOVAL` phase.  They cannot be used to form new squares/lines while marked.
* Forced removal always happens before entering the movement phase when the placement phase ends with no stones marked.
* The “no-move” and “counter removal” mechanism prevents stalemates: a player that blocks their opponent entirely grants them a removal opportunity, followed by an immediate counter removal, after which normal play resumes.
* From an implementation standpoint, each `GameState` holds counters for pending marks and pending captures, ensuring atomic actions in `MARK_SELECTION` and `CAPTURE_SELECTION` consume exactly one unit of the pending work.

This rule set matches the current code paths in `rule_engine.py`, `move_generator.py`, and the Monte Carlo search.  Use it as the authoritative reference when adding tests, debugging game logic, or porting the engine to other languages.

## Code Structure

### Core Game Logic
- **`src/game_state.py`** – Defines the `GameState` container, phase enum, player enum, and helper methods (copy, counting stones, pending mark/capture counters, etc.).  This is the canonical snapshot used everywhere else.
- **`src/rule_engine.py`** – Implements the state transitions for every atomic phase: placement, mark selection, removals, movement, capture selection, forced removal, counter removal.  Also contains shape detection (`detect_shape_formed`, `is_piece_in_shape`) and compatibility wrappers (`apply_move_phase1/3`) used by legacy code.
- **`src/move_generator.py`** – Produces legal actions for the current phase (and applies them).  Dispatches to the relevant rule-engine functions and keeps the public API `generate_all_legal_moves` / `apply_move` stable for MCTS and training loops.

### Learning & Search
- **`src/neural_network.py`** – AlphaZero-style network (three policy heads + value head), tensor conversion helpers, and `get_move_probabilities`.  Central to both training and MCTS exploration.
- **`src/mcts.py`** – Monte Carlo Tree Search implementation.  Uses the neural network for prior probabilities and value estimates, handles simulation bookkeeping, logging, and converts the search result into a move distribution.
- **`src/train.py`** – Self-play and training loop orchestrator.  Generates games, collects `(state, policy, value)` targets, trains the network, manages checkpoints, and runs evaluation games.
- **`src/evaluate.py`** – Utilities to pit models against baselines (random agent or best-so-far) for offline assessment.

### Agents, Tests, Utilities
- **`src/random_agent.py`** – Baseline agent that samples uniformly from legal moves (used in evaluation and debugging).
- **`src/test_random_agent_debug.py` / `src/test_random_agent_enhanced.py`** – Lightweight smoke tests that ensure random games can be executed end-to-end without exceptions after the rule refactor.
- **`tests/test_mcts.py`** – Unit tests for MCTS expansion/backprop mechanics using dummy models/tensors.
- **`run.py` / `run_tests.py`** – Entry points for running experiments or batched tests outside the training loop.
- **`README_TESTS.md`** – Notes on the available automated tests and how to extend them.
- **`AGENT.md.bak`, `dev`, `rewrite.md`** – Historical notes and scratchpad files kept for reference during the refactor.

### Project Layout (Top Level)
- **`checkpoints_eval_run/`** – Default directory where training checkpoints (best/latest model weights) are stored.
- **`3000simu_1.txt` / `test_300simu_1.txt`** – Example self-play logs (useful when inspecting MCTS behaviour or debugging training).
- **`README_zh.md`** – Chinese version of the rules (in progress).  Keep both README files in sync when updating documentation.

## TODO

- [ ] **Validate rule engine correctness** – add targeted unit/behaviour tests for the new atomic phases (mark selection edge cases, forced removal sequences, counter-removal loops).
- [ ] **Refine MCTS logging** – separate true terminal outcomes from network value estimates; optionally add mean/variance of predicted values to avoid confusion when no simulation reaches a terminal state.
- [ ] **Training telemetry** – log/visualise policy loss, value loss, and total loss per iteration (TensorBoard or Matplotlib) and add progress indicators for self-play/training loops.
- [ ] **Performance profiling** – measure CPU/GPU utilisation, optimise state copying/move generation, and explore batched or multi-process self-play to keep the GPU busy.
- [ ] **Model improvements** – experiment with deeper/wider networks, residual blocks, or mixed-precision; tune hyperparameters (learning rate schedules, batch size, Dirichlet noise) to stabilise training.
- [ ] **Value regularisation** – investigate adding step-based penalties/rewards (e.g., encourage faster wins or discourage slow losses) once baseline training is stable.
- [ ] **Potential C++/Rust rewrite** – evaluate portability/performance needs; if Python becomes the bottleneck, sketch an interop-friendly core for move generation and search.



## Q & A


### ❓Q: If the first self-play iteration ends in a draw, how can the model still learn anything?

> There seems to be no value signal (`z = 0`), so doesn’t that mean the network receives zero gradient and no useful update?

---

### 💡A: Even when the game result is a draw, the network *still learns* from the policy head.

In AlphaZero-style training, the neural network is optimized with two losses:

[
L = (v - z)^2 - \pi^T \log p
]

where

* **`v`** = predicted value of the current position,
* **`z`** = final game result (+1 = win, 0 = draw, -1 = loss),
* **`π`** = improved move distribution from MCTS (visit count–based),
* **`p`** = policy head output distribution.

---

#### 1️⃣ When the result is a draw (`z = 0`)

The **value loss** term `(v - z)^2` produces little or no gradient.
However, the **policy loss** term `-πᵀ log p` is still active — it forces the policy head to *imitate* the search distribution produced by MCTS.

MCTS acts as a *teacher*:
even though the final outcome is neutral, the search procedure prefers some actions over others based on simulated rollouts and the value network’s internal evaluations.
Thus, the generated π distribution implicitly carries “which moves look promising” information.

---

#### 2️⃣ What actually happens

* The policy head is penalized for assigning low probability to moves favored by MCTS.
* The value head is repeatedly used during search, indirectly shaping which states are explored next.
* As a result, even with `z = 0`, the policy head gradually shifts from random to meaningful move patterns.

In short:

> The model still learns because MCTS itself encodes value-guided preferences into the visit-count distribution π, and the policy head is trained to match π.
> This indirectly penalizes suboptimal actions — effectively acting as a “soft advantage” signal, even without explicit wins or losses.

---

#### ✅ Summary

| Component       | Source of learning                                           | Works even if draw?             |
| --------------- | ------------------------------------------------------------ | ------------------------------- |
| Value head      | Supervised by final result *(z)*                             | ❌ No                            |
| Policy head     | Cross-entropy imitation of MCTS π                            | ✅ Yes                           |
| Combined effect | Policy improves → better search → better data → better value | 🌀 Iterative self-bootstrapping |

