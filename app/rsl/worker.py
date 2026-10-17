"""Local RSL operations. The demo uses a temporary database and fake observations."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from app.core.config import ModelConfig, ModelPricing
from app.core.db import Database
from app.core.recommender import ModelRecommender
from app.core.router import RouterEngine
from app.core.routing_policy import RoutingPolicy, load_routing_policy
from .evaluation import activate_router, evaluate_router
from .learning import build_candidate
from .runtime import message_group_id
from .store import RSLStore


def demo_context():
    """Offline fake model catalog, deliberately separate from real providers."""
    recommender = ModelRecommender()
    recommender.model_traits = {}
    recommender.task_profiles = {}
    recommender.runtime_signals = {}
    recommender.objectives = {}
    recommender.performance_history = {}
    engine = RouterEngine(None)
    engine.providers = {"demo": SimpleNamespace(health=None)}
    engine.model_map = {
        name: ("demo", ModelConfig(name=name, routing_keys=[name], pricing=ModelPricing(prompt=1, completion=1)))
        for name in ("demo-a", "demo-b")
    }
    return recommender, engine, RoutingPolicy(recommendation_enabled=True)


def demo_training(store, recommender, profile_id="demo"):
    baseline = store.ensure_baseline(profile_id, history=recommender.performance_history)
    for index in range(8):
        messages = [{"role": "user", "content": f"Synthetic training example {index}"}]
        for model, score in (("demo-a", 0.1), ("demo-b", 0.95)):
            request_id = f"train-{index}-{model}"
            row = store.record_decision(
                request_id=request_id, profile_id=profile_id, task="general", model=model,
                features={"input_chars": len(messages[0]["content"])}, candidates=["demo-a", "demo-b"],
                policy_version=baseline["version_id"], group_id=message_group_id(messages))
            attempt = store.record_attempt(
                attempt_id=request_id, decision_id=row["decision_id"], model=model,
                latency_ms=50, cost=0.01, status="completed")
            store.record_feedback(attempt_id=attempt["attempt_id"], source="objective",
                                  source_id=request_id, score=score, passed=score >= 0.8, confidence=1)
    return baseline


def demo_cases():
    return [{
        "case_id": f"heldout-{index}",
        "messages": [{"role": "user", "content": f"Synthetic independent evaluation example {index}"}],
        "models": {
            "demo-a": {"score": 0.8, "cost": 0.02, "latency_ms": 100},
            "demo-b": {"score": 0.95, "cost": 0.01, "latency_ms": 50},
        },
    } for index in range(24)]


def run_demo():
    with tempfile.TemporaryDirectory(prefix="modelrouter-rsl-demo-") as directory:
        store = RSLStore(Path(directory) / "demo.db")
        recommender, engine, policy = demo_context()
        baseline = demo_training(store, recommender)
        candidate = build_candidate(store, "demo")
        report = evaluate_router(store, "demo", candidate["version_id"], demo_cases(),
                                 recommender, engine, policy, dataset_id="synthetic-demo-only")
        active = activate_router(store, "demo", candidate["version_id"], report["evaluation_id"],
                                 recommender, engine, policy)
        restored = store.rollback("demo", baseline["version_id"])
        return {
            "synthetic_demo_only": True, "real_providers_called": False,
            "temporary_database_removed_on_exit": True,
            "baseline_version": baseline["version_id"], "candidate_version": candidate["version_id"],
            "evaluation_passed": report["passed"], "metrics": report["report"]["metrics"],
            "activation_verified": active["version_id"] == candidate["version_id"],
            "rollback_verified": restored["version_id"] == baseline["version_id"],
        }


async def operate(args):
    store = RSLStore(args.db)
    if args.command == "versions":
        return {"active": store.get_active(args.profile_id), "versions": store.list_versions(args.profile_id),
                "activations": store.list_activations(args.profile_id)}
    if args.command == "rollback":
        return store.rollback(args.profile_id, args.version_id)
    if args.command == "feedback":
        return store.record_feedback(attempt_id=args.attempt_id, source=args.source, source_id=args.source_id,
                                      score=args.score, passed=args.passed, confidence=args.confidence)
    recommender = ModelRecommender()
    if args.command == "train":
        store.ensure_baseline(args.profile_id, history=recommender.performance_history)
        return build_candidate(store, args.profile_id, min_samples=args.min_samples,
                               prior_strength=args.prior_strength, min_confidence=args.min_confidence)
    # Loading the catalog constructs adapters but does not call a provider API.
    database = Database(args.router_db)
    await database.connect()
    try:
        await database.migrate()
        engine = RouterEngine(database)
        await engine.reload()
        policy = await load_routing_policy(database, args.profile_id)
        if args.command == "evaluate":
            dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8-sig"))
            if not isinstance(dataset, dict) or not isinstance(dataset.get("cases"), list):
                raise ValueError("dataset must contain dataset_id and cases")
            return evaluate_router(store, args.profile_id, args.version_id, dataset["cases"],
                                   recommender, engine, policy, dataset_id=dataset.get("dataset_id"))
        return activate_router(store, args.profile_id, args.version_id, args.evaluation_id,
                               recommender, engine, policy)
    finally:
        await database.close()


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--db", default=os.environ.get("ROUTER_RSL_DB", "rsl.db"))
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="synthetic temporary-database verification, no external calls")
    for name in ("train", "evaluate", "activate", "rollback", "versions"):
        command = commands.add_parser(name)
        command.add_argument("--profile-id", default="__global__")
        if name in ("evaluate", "activate", "rollback"):
            command.add_argument("--version-id", required=True)
        if name in ("evaluate", "activate"):
            command.add_argument("--router-db", default=os.environ.get("ROUTER_DB", "router.db"))
        if name == "train":
            command.add_argument("--min-samples", type=int, default=5)
            command.add_argument("--prior-strength", type=float, default=5)
            command.add_argument("--min-confidence", type=float, default=0.8)
        if name == "evaluate":
            command.add_argument("--dataset", required=True)
        if name == "activate":
            command.add_argument("--evaluation-id", required=True)
    feedback = commands.add_parser("feedback")
    feedback.add_argument("--attempt-id", required=True)
    feedback.add_argument("--source", choices=["objective", "manual", "user", "judge", "rule"], required=True)
    feedback.add_argument("--source-id", required=True)
    feedback.add_argument("--score", type=float, required=True)
    feedback.add_argument("--passed", action=argparse.BooleanOptionalAction, default=None)
    feedback.add_argument("--confidence", type=float, default=1)
    return root


def main():
    args = parser().parse_args()
    try:
        result = run_demo() if args.command == "demo" else asyncio.run(operate(args))
    except (ValueError, OSError, TypeError) as error:
        parser().exit(2, f"RSL: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
