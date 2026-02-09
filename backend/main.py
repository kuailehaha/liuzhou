from __future__ import annotations

import os
import mimetypes
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.game_state import GameState, Player
from src.move_generator import apply_move, generate_all_legal_moves
from src.neural_network import state_to_tensor
from src.random_agent import RandomAgent
from src.evaluate import MCTSAgent

from .game_manager import GameManager, GameSession
from .model_loader import ModelLoadError, get_model
from .schemas import MoveRequest, NewGameRequest
from .utils import build_game_payload, deserialize_move, move_to_key
from .static_files import NoCacheStaticFiles


# Windows `mimetypes` maps `.js` to `text/plain` unless this is set explicitly.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_UI_DIR = BASE_DIR / "web_ui"

DEFAULT_MODEL_PATH = os.getenv(
    "MODEL_PATH", str((BASE_DIR / "models" / "model_iter_3.pt").resolve())
)
DEFAULT_DEVICE = os.getenv("MODEL_DEVICE", "cpu")
DEFAULT_SIMULATIONS = int(os.getenv("MCTS_SIMULATIONS", "160"))
DEFAULT_TEMPERATURE = float(os.getenv("MCTS_TEMPERATURE", "0.05"))

app = FastAPI(title="Liuzhou Web API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

game_manager = GameManager()

if WEB_UI_DIR.exists():
    app.mount("/ui", NoCacheStaticFiles(directory=WEB_UI_DIR, html=True), name="web_ui")


def _reset_agent(agent: Any) -> None:
    if hasattr(agent, "mcts"):
        agent.mcts.root = None  # type: ignore[attr-defined]


def _legal_moves_dict(moves: Iterable[Mapping[str, Any]]) -> Dict[Any, Mapping[str, Any]]:
    return {move_to_key(move): move for move in moves}


def _create_ai_agent(
    model_path: str | None,
    simulations: int,
    temperature: float,
    device: str,
) -> tuple[Any, bool]:
    """
    Build the agent that will control the AI player.

    Returns (agent, using_random_agent).
    """
    try:
        model = get_model(model_path, device)
    except ModelLoadError as exc:
        if model_path:
            raise HTTPException(status_code=500, detail=str(exc))
        return RandomAgent(), True

    agent = MCTSAgent(
        model=model,
        mcts_simulations=simulations,
        temperature=temperature,
        device=device,
        add_dirichlet_noise=False,
    )
    return agent, False


def _maybe_run_ai_turn(session: GameSession) -> List[Mapping[str, Any]]:
    """
    Let the AI resolve its turn (including follow-up phases that leave
    control with the same player).
    """
    if session.state.is_game_over():
        return []

    ai_moves: List[Mapping[str, Any]] = []
    while (
        not session.state.is_game_over()
        and session.state.current_player == session.ai_player
    ):
        _reset_agent(session.ai_agent)
        candidate_moves = generate_all_legal_moves(session.state)
        if not candidate_moves:
            break

        try:
            selected_move = session.ai_agent.select_move(session.state)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=f"AI could not select move: {exc}")

        legal_map = _legal_moves_dict(candidate_moves)
        key = move_to_key(selected_move)
        if key not in legal_map:
            raise HTTPException(
                status_code=500,
                detail="AI produced an illegal move.",
            )

        move_to_apply = legal_map[key]
        session.state = apply_move(session.state, move_to_apply, quiet=True)
        ai_moves.append(move_to_apply)

    _reset_agent(session.ai_agent)
    return ai_moves


def _ensure_human_turn(session: GameSession) -> None:
    if session.state.current_player != session.human_player:
        raise HTTPException(
            status_code=400,
            detail="It is not the human player's turn.",
        )


def _evaluate_position(session: GameSession) -> Dict[str, Any] | None:
    """
    Run the value head of the AI model on the current state (when available).
    Returns the estimate from the current player's perspective in the [-1, 1] range.
    """
    if torch is None:
        return None
    agent = getattr(session, "ai_agent", None)
    model = getattr(agent, "model", None)
    if model is None:
        return None

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    current_player = session.state.current_player
    try:
        from src.neural_network import wdl_to_scalar
        inputs = state_to_tensor(session.state, current_player).to(device)
        with torch.inference_mode():
            _, _, _, wdl_logits = model(inputs)
        raw_value = float(wdl_to_scalar(wdl_logits).squeeze().detach().to("cpu").item())
        wdl_probs = torch.softmax(wdl_logits, dim=-1).squeeze().detach().to("cpu").tolist()
    except Exception:
        return None

    return {
        "value": raw_value,
        "winProbability": wdl_probs[0] if len(wdl_probs) == 3 else (raw_value + 1.0) / 2.0,
        "drawProbability": wdl_probs[1] if len(wdl_probs) == 3 else 0.0,
        "lossProbability": wdl_probs[2] if len(wdl_probs) == 3 else 1.0 - (raw_value + 1.0) / 2.0,
        "perspective": current_player.name,
        "range": [-1.0, 1.0],
    }


def _prepare_payload(session: GameSession, ai_moves: List[Mapping[str, Any]] | None = None) -> Dict[str, Any]:
    legal_moves = generate_all_legal_moves(session.state) if not session.state.is_game_over() else []
    payload = build_game_payload(session.game_id, session.state, legal_moves, ai_moves)
    payload["meta"] = {
        "humanPlayer": session.human_player.name,
        "aiPlayer": session.ai_player.name,
        "usingRandomAgent": session.using_random_agent,
    }
    if session.using_random_agent:
        payload["meta"]["note"] = "Falling back to RandomAgent because no model checkpoint was configured."
    evaluation = _evaluate_position(session)
    if evaluation is not None:
        payload["evaluation"] = evaluation
    return payload


@app.post("/api/new-game")
def create_new_game(request: NewGameRequest):
    human_player = Player[request.human_player]
    simulations = request.mcts_simulations or DEFAULT_SIMULATIONS
    temperature = request.temperature or DEFAULT_TEMPERATURE
    model_path = request.model_path or DEFAULT_MODEL_PATH

    ai_agent, using_random_agent = _create_ai_agent(
        model_path=model_path,
        simulations=simulations,
        temperature=temperature,
        device=DEFAULT_DEVICE,
    )

    session = game_manager.create_session(
        human_player=human_player,
        ai_agent=ai_agent,
        using_random_agent=using_random_agent,
    )

    ai_moves = _maybe_run_ai_turn(session)
    return _prepare_payload(session, ai_moves=ai_moves or None)


@app.get("/api/game/{game_id}")
def get_game_state(game_id: str):
    try:
        session = game_manager.get_session(game_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Game not found")
    return _prepare_payload(session)


@app.post("/api/game/{game_id}/human-move")
def submit_human_move(game_id: str, request: MoveRequest):
    try:
        session = game_manager.get_session(game_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Game not found")

    if session.state.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over.")

    _ensure_human_turn(session)

    move_payload = request.model_dump(exclude_none=True)
    try:
        candidate_move = deserialize_move(move_payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid move payload: {exc}")

    legal_moves = generate_all_legal_moves(session.state)
    legal_map = _legal_moves_dict(legal_moves)
    key = move_to_key(candidate_move)
    if key not in legal_map:
        raise HTTPException(status_code=400, detail="Move is not legal in the current state.")

    canonical_move = legal_map[key]
    session.state = apply_move(session.state, canonical_move, quiet=True)
    _reset_agent(session.ai_agent)

    ai_moves = None
    if not session.state.is_game_over() and session.state.current_player == session.ai_player:
        ai_moves = _maybe_run_ai_turn(session)

    return _prepare_payload(session, ai_moves=ai_moves or None)


@app.delete("/api/game/{game_id}")
def delete_game(game_id: str):
    game_manager.remove_session(game_id)
    return {"gameId": game_id, "deleted": True}


@app.get("/")
def root():
    if WEB_UI_DIR.exists():
        return RedirectResponse(url="/ui/index.html")
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
