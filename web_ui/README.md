# Liuzhou Web UI

A lightweight front-end plus FastAPI service for human-versus-model play.

## Prerequisites

Install the additional Python dependencies (back-end only needs FastAPI and Uvicorn on top of the existing project stack):

```bash
pip install fastapi uvicorn
```

The service reuses the existing `src/` code, so make sure your environment already satisfies the requirements for PyTorch, NumPy, etc.

## Running the API

```bash
uvicorn backend.main:app --reload
```

Optional environment variables:

- `MODEL_PATH`: path to a `.pt` checkpoint. If omitted, the backend falls back to a randomly initialised model (and the UI banner will mention the fallback).
- `MODEL_DEVICE`: inference device (`cpu` by default).
- `MCTS_SIMULATIONS` and `MCTS_TEMPERATURE`: default search parameters used unless overridden in the “New Game�? form.

The API surface:

- `POST /api/new-game`: create a fresh session. When the human chooses WHITE the AI opens automatically.
- `GET /api/game/{gameId}`: retrieve the serialised game state plus the current legal move list.
- `POST /api/game/{gameId}/human-move`: submit a move. The backend validates it, applies the change, resolves any AI follow-up moves, and returns the updated state.
- `DELETE /api/game/{gameId}`: remove a session.

## Front-end

Static assets live in `web_ui/`. The FastAPI app mounts them at `/ui/`, so after the server starts you can simply open:

```
http://localhost:8000/ui/index.html
```

The UI mirrors the phase-based rule set:

- Placement / marking / captures: click the target intersection.
- Movement: click a source stone, then its destination (clicking a different stone changes the selection).
- Forced-removal phases: click the highlighted opponent stones.
- When the rules require the special `process_removal` action, a button appears beneath the board.

Each response carries the legal moves computed by the core rule engine, so the front-end double-checks validity server-side before accepting a move.

