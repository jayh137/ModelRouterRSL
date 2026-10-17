"""Authenticated RSL administration; never accept a caller-supplied pass flag."""
import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from app.core.routing_policy import load_routing_policy
from app.rsl.evaluation import activate_router, evaluate_router
from app.rsl.learning import build_candidate
from ._auth import require_admin

router = APIRouter(prefix="/rsl", tags=["admin / rsl"], dependencies=[Depends(require_admin)])


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Feedback(Input):
    attempt_id: str = Field(min_length=1, max_length=512)
    source: Literal["objective", "manual", "user", "judge", "rule"]
    source_id: str = Field(min_length=1, max_length=512)
    score: float = Field(ge=0, le=1)
    passed: bool | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)


class Profile(Input):
    profile_id: str = Field(default="__global__", min_length=1, max_length=512)


class Train(Profile):
    min_samples: int = Field(default=5, ge=1)
    prior_strength: float = Field(default=5, ge=0)
    min_confidence: float = Field(default=0.8, ge=0, le=1)


class Evaluate(Profile):
    version_id: str
    dataset_id: str = Field(min_length=1, max_length=512)
    cases: list[dict] = Field(min_length=20, max_length=10000)


class Activate(Profile):
    version_id: str
    evaluation_id: str


class Rollback(Profile):
    version_id: str


def store_for(request):
    store = getattr(request.app.state, "rsl_store", None)
    if store is None:
        raise HTTPException(404, "RSL disabled; set ROUTER_RSL_ENABLED=true")
    return store


async def run(call, *args, **kwargs):
    try:
        return await asyncio.to_thread(call, *args, **kwargs)
    except (ValueError, TypeError) as error:
        raise HTTPException(422, str(error)) from error


@router.get("/decisions")
async def decisions(request: Request, profile_id: str = "__global__", limit: int = Query(100, ge=1, le=1000)):
    return await run(store_for(request).list_decisions, profile_id, limit=limit)


@router.get("/decisions/{decision_id}")
async def decision(decision_id: str, request: Request):
    store = store_for(request)
    row = await run(store.get_decision, decision_id)
    if row is None:
        raise HTTPException(404, "unknown decision")
    attempts = await run(store.list_attempts, decision_id)
    known_cost = sum(item["cost"] for item in attempts if item["cost"] is not None)
    return {**row, "attempts": attempts, "known_cost": known_cost,
            "total_cost": known_cost if attempts and all(a["cost"] is not None for a in attempts) else None}


@router.post("/feedback", status_code=201)
async def feedback(body: Feedback, request: Request):
    return await run(store_for(request).record_feedback, **body.model_dump())


@router.get("/versions")
async def versions(request: Request, profile_id: str = "__global__"):
    store = store_for(request)
    return {"active": await run(store.get_active, profile_id),
            "versions": await run(store.list_versions, profile_id),
            "activations": await run(store.list_activations, profile_id)}


@router.post("/train", status_code=201)
async def train(body: Train, request: Request):
    store = store_for(request)
    await run(store.ensure_baseline, body.profile_id, history=request.app.state.recommender.performance_history)
    return await run(build_candidate, store, **body.model_dump())


@router.post("/evaluate", status_code=201)
async def evaluate(body: Evaluate, request: Request):
    store = store_for(request)
    policy = await load_routing_policy(request.app.state.db, body.profile_id)
    return await run(evaluate_router, store, body.profile_id, body.version_id, body.cases,
                     request.app.state.recommender, request.app.state.router, policy,
                     dataset_id=body.dataset_id)


@router.get("/evaluations/{evaluation_id}")
async def evaluation(evaluation_id: str, request: Request):
    result = await run(store_for(request).get_evaluation, evaluation_id)
    if result is None:
        raise HTTPException(404, "unknown evaluation")
    return result


@router.post("/activate")
async def activate(body: Activate, request: Request):
    store = store_for(request)
    policy = await load_routing_policy(request.app.state.db, body.profile_id)
    return await run(activate_router, store, body.profile_id, body.version_id, body.evaluation_id,
                     request.app.state.recommender, request.app.state.router, policy)


@router.post("/rollback")
async def rollback(body: Rollback, request: Request):
    return await run(store_for(request).rollback, body.profile_id, body.version_id)
