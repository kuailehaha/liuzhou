# Liuzhou Web UI

A lightweight front-end plus FastAPI service for human-versus-model play.

## Prerequisites

Install the additional Python dependencies (back-end only needs FastAPI and Uvicorn on top of the existing project stack):

```bash
pip install fastapi uvicorn
```

The service reuses the project rule engine and V1 portable MCTS, so make sure the environment already satisfies the PyTorch and project requirements. Build the CPU-only portable extension before selecting `portable_cpp`:

```bash
python scripts/build_portable_cpp.py --force
```

## Running the API

```bash
uvicorn backend.main:app --reload
```

Optional environment variables:

- `MODEL_PATH`: path to the retained `best_model.pt`. If omitted, the local portable run's retained incumbent path is used.
- `MODEL_DEVICE`: inference device (`cpu` by default; use `mps` on Apple Silicon).
- `GAMEPLAY_SEARCH_BACKEND`: `portable_cpp` (default), `portable_python`, or `legacy`.
- `PORTABLE_CPP_THREADS`: C++ tree-search threads (default 1).
- `MCTS_SIMULATIONS` and `MCTS_TEMPERATURE`: defaults are 1024 and 0 unless overridden in the “New Game” form.

A missing/bad checkpoint, unavailable portable extension, or unsupported device is an explicit HTTP error. The service never silently switches to Legacy search or a random model/agent.

The API surface:

- `POST /api/new-game`: create a fresh session. When the human chooses WHITE the AI opens automatically.
- `GET /api/game/{gameId}`: retrieve the serialised game state plus the current legal move list.
- `POST /api/game/{gameId}/human-move`: submit one human move and get the immediate updated state.
- `POST /api/game/{gameId}/ai-move`: ask the backend to resolve the AI turn (including chained follow-up phases owned by AI).
- `DELETE /api/game/{gameId}`: remove a session.

The portable gameplay agent uses the same 220-dimensional action mapping and player-change-aware value backup as portable training. It disables root noise, preserves the subtree across consecutive atomic phases owned by the same player, and synchronises only after an external human move. Response metadata includes the resolved checkpoint SHA, device, backend, simulations, temperature, threads, elapsed search time and root top-10 `P/N/Q`. The additive `gameRecord` field accumulates every human/AI atomic action, its before/after state and the complete AI root audit, so a high-budget game can be replayed without relying on only the final board.

## Front-end

Static assets live in `web_ui/`. The FastAPI app mounts them at `/ui/`, so after the server starts you can simply open:

```
http://localhost:8000/ui/index.html
```

The UI mirrors the phase-based rule set:

- Placement / marking / captures: click the target intersection.
- Movement: click a source stone, then its destination (clicking a different stone changes the selection).
- Forced-removal phases: click the highlighted opponent stones.
- The special `process_removal` action is auto-applied for the human side when required.

Each response carries the legal moves computed by the core rule engine, so the front-end double-checks validity server-side before accepting a move.

Use `best_model.pt` for human play. `current.pt` is an unpromoted training candidate and may be weaker than the incumbent. The optional Legacy backend remains diagnostic only; its value backup has been corrected to flip only when the side to move actually changes, but it is not the portable C++ production path.
