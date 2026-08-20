from __future__ import annotations

from pydantic import BaseModel, Field


class FirstRecommendRequest(BaseModel):
    text: str = Field(min_length=1)
    strategy: str
    strength: float = Field(0.5, ge=0.0, le=1.0)


class FirstNextRequest(BaseModel):
    play_duration_seconds: float = Field(0.0, ge=0.0)
    repeat_count: int = Field(0, ge=0)


class FirstFeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1)
    song_id: str = Field(min_length=1)
    music_preference: int = Field(ge=1, le=5)
    regulation_effect: int = Field(ge=1, le=5)
    play_duration_seconds: float = Field(0.0, ge=0.0)
    repeat_count: int = Field(0, ge=0)


class SecondStartRequest(BaseModel):
    text: str = Field(min_length=1)
    strategy: str
    user_initial_v: float = Field(ge=-1.0, le=1.0)
    user_initial_a: float = Field(ge=-1.0, le=1.0)


class SecondFeedbackRequest(BaseModel):
    user_felt_v: float = Field(ge=-1.0, le=1.0)
    user_felt_a: float = Field(ge=-1.0, le=1.0)
    strategy_rating: int = Field(ge=1, le=5)
    music_preference: int | None = Field(None, ge=1, le=5)
    play_duration_seconds: float = Field(0.0, ge=0.0)
    repeat_count: int = Field(0, ge=0)


class SecondFinishRequest(BaseModel):
    overall_strategy_fit: int | None = Field(None, ge=1, le=5)
    satisfaction: int | None = Field(None, ge=1, le=5)
    enjoyment: int | None = Field(None, ge=1, le=5)
    smoothness: int | None = Field(None, ge=1, le=5)
    willingness_to_use_again: int | None = Field(None, ge=1, le=5)
