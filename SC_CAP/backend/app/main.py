from __future__ import annotations

from functools import lru_cache
import json
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sc_cap.models import TextVAInference

from .catalog import Catalog
from .config import (CATALOG_PATH, DEVICE, FIRST_FEEDBACK_LOG_PATH, FIRST_PAPER_MUSIC_DB, FIRST_PAPER_TEXT_CHECKPOINT,
                     MSMMR_ROOT, MSMMR_TEXT_CHECKPOINT, SC_CAP_CONFIG_PATH,
                     SESSION_LOG_PATH, TRACK_METADATA_PATH)
from .first_paper import FirstPaperService
from .planner import (InfeasiblePlanError, PlannerSettings, RegulationPlanner,
                      VALID_STRATEGIES, load_planner_config)
from .schemas import (FirstFeedbackRequest, FirstNextRequest, FirstRecommendRequest, SecondFeedbackRequest,
                      SecondFinishRequest, SecondStartRequest)
from .sessions import RegulationSessionStore, STRATEGY_QUESTIONS


@lru_cache(maxsize=1)
def first_service() -> FirstPaperService:
    return FirstPaperService(FIRST_PAPER_MUSIC_DB, FIRST_PAPER_TEXT_CHECKPOINT, DEVICE)


@lru_cache(maxsize=1)
def second_settings() -> PlannerSettings:
    return load_planner_config(SC_CAP_CONFIG_PATH)


@lru_cache(maxsize=1)
def second_store() -> RegulationSessionStore:
    catalog = Catalog.load(CATALOG_PATH, TRACK_METADATA_PATH)
    planner = RegulationPlanner(catalog, second_settings())
    return RegulationSessionStore(planner, SESSION_LOG_PATH)


@lru_cache(maxsize=1)
def second_text_va() -> TextVAInference:
    return TextVAInference(MSMMR_TEXT_CHECKPOINT, MSMMR_ROOT, DEVICE, allow_fallback=False)


def create_app() -> FastAPI:
    app = FastAPI(title="Music Mood Regulation", version="2.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "text_va_checkpoint": str(MSMMR_TEXT_CHECKPOINT),
                "sequence_steps": second_settings().total_steps}

    @app.post("/api/first/recommend")
    def first_recommend(request: FirstRecommendRequest) -> dict:
        try:
            return first_service().recommend(request.text, request.strategy, request.strength)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/first/{session_id}/next")
    def first_next(session_id: str, request: FirstNextRequest) -> dict:
        try:
            return first_service().next(session_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/first/feedback")
    def first_feedback(request: FirstFeedbackRequest) -> dict:
        if request.session_id not in first_service().sessions:
            raise HTTPException(status_code=404, detail="session not found")
        FIRST_FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {**request.model_dump(), "timestamp": time.time()}
        with FIRST_FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return {"status": "recorded"}

    @app.post("/api/second/sessions")
    def second_start(request: SecondStartRequest) -> dict:
        if request.strategy not in VALID_STRATEGIES:
            raise HTTPException(status_code=400, detail="unknown strategy")
        try:
            prediction = second_text_va().predict(request.text)
            text_va = [prediction["text_pred_v"], prediction["text_pred_a"]]
            session, recommendation = second_store().start(request.text, request.strategy, text_va,
                [request.user_initial_v, request.user_initial_a])
            return {"session_id": session.session_id, "strategy": session.strategy,
                    "text_pred_va": session.text_pred_va, "user_initial_va": session.user_initial_va,
                    "recommendation": recommendation, "strategy_question": STRATEGY_QUESTIONS[session.strategy],
                    "total_steps": session.total_steps, "first_recommendation_source": "text_va_only"}
        except InfeasiblePlanError as error:
            raise HTTPException(status_code=409, detail=error.diagnostics) from error
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/second/sessions/{session_id}/feedback")
    def second_feedback(session_id: str, request: SecondFeedbackRequest) -> dict:
        try:
            session = second_store().get(session_id)
            record, recommendation = session.submit_feedback(second_store().planner,
                user_felt_va=[request.user_felt_v, request.user_felt_a], strategy_rating=request.strategy_rating,
                music_preference=request.music_preference, playback_seconds=request.play_duration_seconds,
                repeat_count=request.repeat_count)
            return {"feedback": record, "recommendation": recommendation,
                    "strategy_question": STRATEGY_QUESTIONS[session.strategy],
                    "complete": recommendation is None,
                    "planned_music_trajectory": session.planned_music_trajectory,
                    "actual_user_trajectory": session.actual_user_trajectory}
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except InfeasiblePlanError as error:
            second_store().record_infeasible(session)
            raise HTTPException(status_code=409, detail=error.diagnostics) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/second/sessions/{session_id}/finish")
    def second_finish(session_id: str, request: SecondFinishRequest) -> dict:
        try:
            session = second_store().get(session_id)
            if len(session.step_records) != session.total_steps:
                raise HTTPException(status_code=409, detail="finish is available after all four feedback entries")
            return second_store().save(session, request.model_dump())
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return app


app = create_app()
