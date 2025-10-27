from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Optional
from uuid import uuid4

from src.game_state import GameState, Player


@dataclass
class GameSession:
    game_id: str
    state: GameState
    human_player: Player
    ai_player: Player
    ai_agent: Any
    using_random_agent: bool
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_activity = time.time()


class GameManager:
    """
    Thread-safe container that manages active game sessions.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, GameSession] = {}
        self._lock = Lock()

    def create_session(
        self,
        human_player: Player,
        ai_agent: Any,
        using_random_agent: bool = False,
        initial_state: Optional[GameState] = None,
    ) -> GameSession:
        game_id = uuid4().hex
        state = initial_state.copy() if initial_state else GameState()
        session = GameSession(
            game_id=game_id,
            state=state,
            human_player=human_player,
            ai_player=human_player.opponent(),
            ai_agent=ai_agent,
            using_random_agent=using_random_agent,
        )
        with self._lock:
            self._sessions[game_id] = session
        return session

    def get_session(self, game_id: str) -> GameSession:
        try:
            session = self._sessions[game_id]
        except KeyError as exc:
            raise KeyError(f"Unknown game id: {game_id}") from exc
        session.touch()
        return session

    def remove_session(self, game_id: str) -> None:
        with self._lock:
            self._sessions.pop(game_id, None)

