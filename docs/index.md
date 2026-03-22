# Documentation

This is the documentation hub for the Liuzhou Chess AI project. Start wherever fits your goal.

## Guides

- **[Quick Start](./quickstart.md)** — Build, train, evaluate, and launch human-vs-AI play
- **[Architecture](./architecture.md)** — How the system is organized and how the three generations relate
- **[Method](./method.md)** — Network architecture, MCTS, training objectives, and the v1 pipeline
- **[Rules](./rules.md)** — Complete game rules (authoritative source for all implementations)
- **[Results](./results.md)** — Milestones, metrics, and what the project has achieved so far
- **[Human vs AI](./gameplay_system.md)** — Web-based gameplay system
- **[Hard Questions](./faq.md)** — Deep technical questions that don't have easy answers

## Key Entry Points

| Task | Entry |
|---|---|
| Build / compile | `scripts/instruct.sh` |
| Train (v1 staged pipeline) | `scripts/big_train_v1.sh` |
| Single checkpoint evaluation | `scripts/eval_checkpoint.py` |
| Tournament evaluation | `scripts/tournament_v1_eval.py` |
| Human-vs-AI backend | `backend/main.py` |
| Web UI | `web_ui/` |

## Notes

- The current mainline is `v1/`. Legacy (`src/`) and v0 (`v0/`) remain as reference implementations and underlying capabilities.
- `v1/Design.md` contains detailed design history and stage-by-stage records — useful for archaeology, not as a starting point.
- `docs/rules.md` is the authoritative rules source. The root-level `rule_description.md` is a legacy compatibility entry.
