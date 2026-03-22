# Liuzhou Chess AI

An AlphaZero-style reinforcement learning system for **Liuzhou Chess** (六洲棋), a traditional Chinese board game — built end-to-end from rule formalization to self-play training to human-vs-AI gameplay.

## The Game

Liuzhou Chess is played on a 6×6 grid. Two players each have 18 pieces (black and white). The game proceeds in two main stages:

1. **Placement** — Players alternate placing pieces on the board. Forming a *Fang* (方, a 2×2 square of same-color pieces) lets you mark 1 opponent piece; forming a *Zhou* (洲, 6 same-color pieces filling an entire row or column) lets you mark 2. Marked pieces are removed together once the board fills up.

2. **Movement** — Players take turns moving one piece to an adjacent empty cell. Forming a Fang or Zhou now *captures* opponent pieces directly. The game ends when one side loses all pieces, or draw conditions are met.

The multi-phase structure (placement → marking → removal → movement → capture) makes Liuzhou Chess an interesting RL target: each "logical turn" can span multiple atomic decisions, and the reward signal is extremely sparse — most games end in draws.

Full rules: [`docs/rules.md`](./docs/rules.md)

## What This Project Does

This is a solo project that implements the full stack for training a Liuzhou Chess AI:

- **Rule engine** with atomic phase semantics, verified across Python and C++/CUDA implementations
- **MCTS search** guided by a neural network (policy + bucketed value heads)
- **AlphaZero-style self-play** training pipeline
- **Checkpoint evaluation** (vs-random, vs-previous, round-robin tournament)
- **Web-based human-vs-AI** gameplay (FastAPI backend + browser frontend)

### The Engineering Story

The Python prototype worked — but was far too slow for meaningful RL iteration. Profiling revealed bottlenecks in MCTS object copying, fragmented legal-mask computation, and inference overhead. This motivated two major rewrites:

| Generation | Focus | Key Change |
|:---:|---|---|
| **Legacy** (`src/`) | Get the full loop working | Pure Python rule engine + MCTS + training |
| **v0** (`v0/`) | Move bottlenecks to native code | C++/CUDA rule engine, move generator, MCTS core via PyBind11 |
| **v1** (`v1/`) | Scale self-play-to-training | Staged GPU pipeline with wave-batched simulation, fused operators, tensor-native data flow |

The current mainline is **v1**, built on top of `v0_core`'s C++/CUDA capabilities.

## Results

- **~1,157× throughput** vs. the fastest legacy single-GPU baseline (positions/s, same checkpoint)
- **99.8% win rate** vs. random agent (2,000 games, low-simulation setting)
- Trained on **4× NVIDIA H20** GPUs with staged self-play → train → eval → infer pipeline
- Self-play decisive-game ratio improved from **1.23% → 81.72%** over training, addressing the sparse-reward challenge through bucketed value targets and data-distribution strategies

## Quick Start

### Build

```bash
bash scripts/instruct.sh \
  --python-bin <your-python> \
  --cuda-home /usr/local/cuda \
  --build-v0 1
```

### Train

```bash
bash scripts/big_train_v1.sh
```

This runs the full staged loop: self-play → train → eval → infer, with curriculum scheduling.

### Play Against the AI

```bash
uvicorn backend.main:app --reload
```

Then open `http://localhost:8000/ui/index.html` in your browser.

## Project Structure

```
src/          Legacy Python implementation (rule reference + functional verification)
v0/           C++/CUDA core (rule engine, move generator, MCTS, PyBind11 bindings)
v1/           Training pipeline (staged self-play, train bridge, trajectory buffer)
backend/      FastAPI server for human-vs-AI gameplay
web_ui/       Browser frontend (board rendering + interaction)
scripts/      Training scripts, evaluation, model export
tests/        Unit tests, rule engine regression (1000+ cases), CUDA kernel tests
docs/         Project documentation
```

## Documentation

| Document | What it covers |
|---|---|
| [Quick Start](./docs/quickstart.md) | Build, train, evaluate, play — step by step |
| [Architecture](./docs/architecture.md) | System decomposition and the three-generation evolution |
| [Method](./docs/method.md) | Network design, MCTS integration, training objectives |
| [Rules](./docs/rules.md) | Complete Liuzhou Chess rules (authoritative source) |
| [Results](./docs/results.md) | Milestones and key metrics |
| [Human vs AI](./docs/gameplay_system.md) | Web gameplay system overview |
| [Hard Questions](./docs/faq.md) | Deep technical questions worth revisiting |

## Contact

Questions, suggestions, or want to play? Reach me at [kuailepapa@gmail.com](mailto:kuailepapa@gmail.com).
