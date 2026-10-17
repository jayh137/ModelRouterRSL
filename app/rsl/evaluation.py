"""Replay the production recommender on independently measured held-out cases."""
from __future__ import annotations

import copy

from app.schemas.chat import ChatMessage
from .learning import evaluate_candidate
from .runtime import message_group_id
from .store import digest


def routing_fingerprint(recommender, router_engine, policy) -> str:
    policy_data = policy.model_dump()
    # Switching collection to active use is the intended rollout operation.
    policy_data.pop("rsl_mode", None)
    policy_data.pop("rsl_enabled", None)
    models = {
        key: {"provider": provider, "config": config.model_dump()}
        for key, (provider, config) in router_engine.model_map.items()
    }
    return digest({
        "schema": 1, "policy": policy_data, "models": models,
        "traits": recommender.model_traits, "tasks": recommender.task_profiles,
        "objectives": recommender.objectives, "runtime_signals": recommender.runtime_signals,
        "excluded_models": sorted(recommender.excluded_models),
        "tier_adjustments": recommender.tier_adjustments,
        "scoring_parameters": {name: getattr(recommender, name) for name in (
            "price_cost_weight", "price_cost_scale", "success_rate_weight",
            "provider_health_blend_weight", "success_rate_soft_floor", "success_rate_soft_penalty",
            "success_rate_hard_floor", "success_rate_hard_penalty",
        )},
    })


def evaluate_router(store, profile_id, version_id, cases, recommender,
                    router_engine, policy, *, thresholds=None, dataset_id=None):
    if not policy.recommendation_enabled:
        raise ValueError("evaluation requires recommendation_enabled=true")
    available = set(router_engine.model_map)
    if not available:
        raise ValueError("configure candidate models before evaluating the router")
    # Freeze mutable health/availability and configuration for the paired replay.
    engine = copy.copy(router_engine)
    engine.model_map = copy.deepcopy(router_engine.model_map)
    engine.providers = {}
    from types import SimpleNamespace
    for name, provider in router_engine.providers.items():
        health = getattr(provider, "health", None)
        snapshot = copy.deepcopy(health.snapshot()) if health else None
        engine.providers[name] = SimpleNamespace(
            health=SimpleNamespace(snapshot=lambda s=snapshot: copy.deepcopy(s)) if snapshot else None)
    availability = {key: router_engine.is_model_available(key) for key in available}
    engine.is_model_available = lambda key: availability.get(key, False)
    engine.model_availability_snapshot = lambda key: None
    frozen_recommender = copy.deepcopy(recommender)
    frozen_policy = policy.model_copy(deep=True)
    candidate = store.get_version(version_id)
    active = store.get_active(profile_id)
    if candidate is None or active is None:
        raise ValueError("unknown candidate or baseline")
    training_groups = {row["group_id"] for version in (candidate, active)
                       for row in store.get_lineage(version["version_id"])}
    prepared, seen = [], set()
    for original in cases:
        case = copy.deepcopy(original)
        if not isinstance(case.get("messages"), list) or not case["messages"]:
            raise ValueError("held-out cases must contain messages for actual routing replay")
        messages = [ChatMessage.model_validate(message) for message in case["messages"]]
        group = message_group_id(messages)
        if group in training_groups or group in seen:
            raise ValueError("held-out messages overlap training data or repeat another case")
        seen.add(group)
        # Canonical fingerprints cannot be bypassed by renaming a held-out case.
        case["group_id"] = group
        declared = case.get("allowed_models")
        if declared is not None and (not isinstance(declared, list) or any(not isinstance(m, str) for m in declared)):
            raise ValueError("allowed_models must be a list of model identifiers")
        allowed = frozen_policy.effective_allowed_models(
            set(declared) if declared is not None else None, available_models=available)
        allowed = (allowed if allowed is not None else available) & available
        if not allowed:
            raise ValueError("held-out case has no eligible configured models")
        task_profile = frozen_recommender.profile_task(messages)
        eligible = set()
        for key in allowed:
            _, config = engine.model_map[key]
            traits = frozen_recommender._trait_for(key, config.name)
            reason = frozen_recommender._exclude_reason(
                routing_key=key, model_name=config.name, traits=traits, task=task_profile,
                allowed_models=allowed, policy=frozen_policy, router_engine=engine)
            if reason is None:
                eligible.add(key)
        if not eligible:
            raise ValueError("held-out case has no eligible models after capability and availability filtering")
        if not isinstance(case.get("models"), dict) or not eligible.issubset(case["models"]):
            raise ValueError("held-out matrix must include every eligible configured model")
        case["allowed_models"] = sorted(eligible)
        task = task_profile.name
        if case.get("task") is not None and case["task"] != task:
            raise ValueError("held-out task must match the production task classifier")
        case["task"] = task
        prepared.append(case)

    def rank(history, case):
        model = frozen_recommender.with_performance_history(history)
        result = model.recommend(messages=case["messages"], router_engine=engine,
                                 policy=frozen_policy, allowed_models=set(case["allowed_models"]))
        if not result.applied or not result.selected_model:
            raise ValueError("recommendation did not apply; this case cannot validate RSL routing")
        return result.selected_model

    return evaluate_candidate(
        store, profile_id, version_id, prepared, rank, thresholds=thresholds,
        dataset_id=dataset_id,
        context_fingerprint=routing_fingerprint(frozen_recommender, engine, frozen_policy),
    )


def activate_router(store, profile_id, version_id, evaluation_id, recommender, router_engine, policy):
    evaluation = store.get_evaluation(evaluation_id)
    if evaluation is None:
        raise ValueError("unknown evaluation")
    fingerprint = routing_fingerprint(recommender, router_engine, policy)
    if evaluation["report"].get("context_fingerprint") != fingerprint:
        raise ValueError("routing configuration changed after evaluation; evaluate again before activation")
    return store.activate(profile_id, version_id, evaluation_id)
