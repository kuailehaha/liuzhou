from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class NewGameRequest(BaseModel):
    human_player: Literal["BLACK", "WHITE"] = Field(
        default="BLACK",
        description="Which colour the human controls.",
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Optional override for the model checkpoint path.",
    )
    mcts_simulations: Optional[int] = Field(
        default=None,
        ge=1,
        description="Override the global default number of simulations run per AI move.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional temperature override for the AI policy head.",
    )


class MoveRequest(BaseModel):
    phase: str
    action_type: str = Field(..., alias="actionType")
    position: Optional[List[int]] = None
    from_position: Optional[List[int]] = Field(default=None, alias="fromPosition")
    to_position: Optional[List[int]] = Field(default=None, alias="toPosition")

    class Config:
        populate_by_name = True

