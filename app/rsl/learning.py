"""First-version RSL: confidence-weighted history and paired held-out evaluation.

Scores use the [0, 1] scale expected by ModelRecommender. Training never chooses
a model using a held-out label. The injected rank callback must run the actual
router against the supplied history; evaluation labels are removed before it runs.
"""
from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable

from .store import RSLStore, digest, identifier, number, utcnow


SOURCE_PRIORITY = {"objective": 5, "user": 4, "manual": 4, "rule": 3, "judge": 2}


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _cutoff(value: str | None) -> str:
    if value is None:
        return utcnow()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("cutoff must be an ISO 8601 timestamp with a timezone") from exc
    if parsed.tzinfo is None or parsed > datetime.now(timezone.utc):
        raise ValueError("cutoff must include a timezone and must not be in the future")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _average(records: list[dict], field: str, weight: str = "scored_cases") -> float | None:
    usable = [(record[field], record.get(weight, 0)) for record in records
              if isinstance(record.get(field), (int, float)) and not isinstance(record.get(field), bool)
              and record.get(weight, 0) > 0]
    if not usable:
        return None
    return sum(value * count for value, count in usable) / sum(count for _, count in usable)


def _summary(records: list[dict]) -> dict:
    return {
        "cases": sum(record.get("cases", 0) for record in records),
        "scored_cases": sum(record.get("scored_cases", 0) for record in records),
        "score_avg": _average(records, "score_avg"),
        "pass_rate": _average(records, "pass_rate"),
        "bad_output_rate": _average(records, "bad_output_rate"),
    }


def build_candidate(store: RSLStore, profile_id: str, *, min_samples: int = 5,
                    prior_strength: float = 5.0, prior_score: float = 0.5,
                    min_confidence: float = 0.8, bad_score_threshold: float = 0.2,
                    cutoff: str | None = None) -> dict:
    """Build an immutable snapshot; never change the active pointer.

    Each attempt contributes at most one label. Trusted objective/user labels
    take precedence over automated judging, then confidence and recency decide.
    ``min_samples`` requires distinct logical request groups in each task/model
    cell. Unknown feedback sources are collected but not used for learning.
    The original baseline supplies fixed priors, avoiding repeated counting of
    previous rounds' observations through the prior.
    """
    _positive_int(min_samples, "min_samples")
    prior_strength = number(prior_strength, "prior_strength")
    prior_score = number(prior_score, "prior_score", maximum=1)
    min_confidence = number(min_confidence, "min_confidence", maximum=1)
    bad_score_threshold = number(bad_score_threshold, "bad_score_threshold", maximum=1)
    cutoff = _cutoff(cutoff)
    baseline = store.ensure_baseline(profile_id)
    parent = store.get_active(profile_id)
    if parent["kind"] == "candidate" and cutoff < parent["cutoff"]:
        raise ValueError("cutoff cannot predate the inherited active candidate's training data")
    history = copy.deepcopy(parent["history"])
    if not isinstance(history.get("profiles", {}), dict):
        raise ValueError("baseline history.profiles must be an object")
    profiles = history.setdefault("profiles", {})
    selected: dict[str, dict] = {}
    reserved = store.heldout_hashes(profile_id)
    for row in store.feedback_rows(profile_id, cutoff):
        if row["confidence"] < min_confidence or row["confidence"] <= 0 or row["source"] not in SOURCE_PRIORITY:
            continue
        if (digest(row["group_id"]) in reserved["groups"] or digest(row["decision_id"]) in reserved["cases"]
                or digest(row["request_id"]) in reserved["cases"]):
            continue
        previous = selected.get(row["attempt_id"])
        priority = lambda item: (SOURCE_PRIORITY[item["source"]], item["confidence"], item["created_at"], item["feedback_id"])
        if previous is None or priority(row) > priority(previous):
            selected[row["attempt_id"]] = row
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in selected.values():
        cells[(row["task"], row["model"])].append(row)
    used, updated_cells = [], []
    for (task, model), rows in sorted(cells.items()):
        if len({row["group_id"] for row in rows}) < min_samples:
            continue
        base_record = baseline["history"].get("profiles", {}).get(task, {}).get("models", {}).get(model, {})
        base_score = base_record.get("score_avg")
        base_score = prior_score if base_score is None else number(base_score, "baseline score_avg", maximum=1)
        total_weight = sum(row["confidence"] for row in rows)
        score = (prior_strength * base_score + sum(row["confidence"] * row["score"] for row in rows)) / (prior_strength + total_weight)
        passed_rows = [row for row in rows if row["passed"] is not None]
        pass_rate = None
        if passed_rows:
            base_pass = base_record.get("pass_rate")
            base_pass = prior_score if base_pass is None else number(base_pass, "baseline pass_rate", maximum=1)
            pass_rate = (prior_strength * base_pass + sum(row["confidence"] * bool(row["passed"]) for row in passed_rows)) / (prior_strength + sum(row["confidence"] for row in passed_rows))
        base_bad = base_record.get("bad_output_rate")
        base_bad = 0.0 if base_bad is None else number(base_bad, "baseline bad_output_rate", maximum=1)
        bad_weight = sum(row["confidence"] for row in rows if row["score"] < bad_score_threshold or row["passed"] == 0)
        bad_rate = (prior_strength * base_bad + bad_weight) / (prior_strength + total_weight)
        profile = profiles.setdefault(task, {"models": {}})
        profile.setdefault("models", {})[model] = {
            "cases": len(rows), "scored_cases": len(rows), "score_avg": score,
            "pass_rate": pass_rate, "bad_output_rate": bad_rate,
        }
        used.extend(rows)
        updated_cells.append({"task": task, "model": model, "attempts": len(rows), "groups": len({r["group_id"] for r in rows})})
    if not used:
        raise ValueError(f"insufficient reliable feedback: need at least {min_samples} distinct groups in a task/model cell")
    all_models: dict[str, list[dict]] = defaultdict(list)
    all_records = []
    for profile in profiles.values():
        records = profile.get("models", {})
        profile["summary"] = _summary(list(records.values()))
        for model, record in records.items():
            all_models[model].append(record)
            all_records.append(record)
    history["summary"] = {**_summary(all_records), "models": {model: _summary(records) for model, records in all_models.items()}}
    history.setdefault("metadata", {})["rsl"] = {"schema": 1, "cutoff": cutoff, "profile_id": profile_id}
    metadata = {
        "min_samples": min_samples, "prior_strength": prior_strength, "prior_score": prior_score,
        "min_confidence": min_confidence, "bad_score_threshold": bad_score_threshold,
        "prior_version_id": baseline["version_id"], "updated_cells": updated_cells,
        "feedback_ids": sorted(row["feedback_id"] for row in used),
        "label_selection": "source_priority_then_confidence_then_recency; one label per attempt",
    }
    return store._create_candidate(profile_id=profile_id, parent_version_id=parent["version_id"],
                                   cutoff=cutoff, history=history, metadata=metadata, lineage=used)


@dataclass(frozen=True)
class EvaluationThresholds:
    min_cases: int = 20
    quality_tolerance: float = 0.0
    max_task_quality_drop: float = 0.0
    max_cost_ratio: float = 1.0
    max_cost_increase: float = 0.0
    max_latency_ratio: float = 1.10
    max_latency_increase_ms: float = 0.0

    def validate(self) -> dict:
        _positive_int(self.min_cases, "min_cases")
        for name in ("quality_tolerance", "max_task_quality_drop"):
            number(getattr(self, name), name, maximum=1)
        for name in ("max_cost_ratio", "max_cost_increase", "max_latency_ratio", "max_latency_increase_ms"):
            number(getattr(self, name), name)
        return asdict(self)


def _p95(values: list[float]) -> float:
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def evaluate_candidate(store: RSLStore, profile_id: str, version_id: str,
                       cases: list[dict], rank: Callable[[dict, dict], str], *,
                       thresholds: EvaluationThresholds | dict | None = None,
                       dataset_id: str | None = None, context_fingerprint: str | None = None) -> dict:
    """Evaluate actual router choices on a complete, independently measured matrix.

    Each case requires ``case_id``, ``group_id``, ``task`` and ``models`` mapping
    model IDs to measured ``score``, total ``cost`` and ``latency_ms``. Cost must
    include any retries and evaluation calls associated with that measured
    route. Optional ``messages``, ``features`` and ``allowed_models`` are supplied
    to rank(history, case), with the model observations removed. The callback
    must apply the same hard eligibility constraints as production routing.

    Integrity failures raise ValueError; quality/cost/latency failures produce
    an auditable failed evaluation. Nothing in this function activates a policy.
    """
    if thresholds is None:
        thresholds = EvaluationThresholds()
    elif isinstance(thresholds, dict):
        thresholds = EvaluationThresholds(**thresholds)
    if not isinstance(thresholds, EvaluationThresholds):
        raise ValueError("thresholds must be EvaluationThresholds or a matching dictionary")
    config = thresholds.validate()
    if dataset_id is not None:
        identifier(dataset_id, "dataset_id")
    context_fingerprint = context_fingerprint or getattr(rank, "context_fingerprint", None)
    if context_fingerprint is not None:
        identifier(context_fingerprint, "context_fingerprint")
    candidate = store.get_version(version_id)
    baseline = store.get_active(profile_id)
    if candidate is None or candidate["profile_id"] != profile_id or candidate["kind"] != "candidate" or baseline is None:
        raise ValueError("evaluation requires a candidate and an active baseline in the same profile")
    if not isinstance(cases, list) or len(cases) < thresholds.min_cases:
        raise ValueError(f"held-out evaluation requires at least {thresholds.min_cases} cases")
    lineage = store.get_lineage(version_id) + store.get_lineage(baseline["version_id"])
    training_groups = {row["group_id"] for row in lineage}
    training_cases = {row[key] for row in lineage for key in ("decision_id", "request_id")}
    case_ids, group_ids = set(), set()
    candidate_scores, baseline_scores, candidate_costs, baseline_costs = [], [], [], []
    candidate_latencies, baseline_latencies = [], []
    tasks: dict[str, list[tuple[float, float]]] = defaultdict(list)
    decisions = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each held-out case must be an object")
        case_id = identifier(case.get("case_id"), "case_id")
        group_id = identifier(case.get("group_id"), "group_id")
        task = identifier(case.get("task"), "task")
        if case_id in case_ids or group_id in group_ids:
            raise ValueError("held-out cases must have distinct case IDs and logical groups")
        if group_id in training_groups or case_id in training_cases or case.get("request_id") in training_cases:
            raise ValueError("training and held-out cases/groups overlap")
        case_ids.add(case_id)
        group_ids.add(group_id)
        matrix = case.get("models")
        if not isinstance(matrix, dict) or not matrix:
            raise ValueError("each held-out case requires measured model observations")
        allowed = case.get("allowed_models", list(matrix))
        if not isinstance(allowed, list) or not allowed or any(not isinstance(model, str) for model in allowed) or len(set(allowed)) != len(allowed):
            raise ValueError("allowed_models must be a nonempty distinct model list")
        if any(model not in matrix for model in allowed):
            raise ValueError("missing observations for one or more eligible models")
        observations = {}
        for model, observation in matrix.items():
            identifier(model, "observed model")
            if not isinstance(observation, dict):
                raise ValueError("model observation must be an object")
            observations[model] = {
                "score": number(observation.get("score"), "observed score", maximum=1),
                "cost": number(observation.get("cost"), "observed total cost"),
                "latency_ms": number(observation.get("latency_ms"), "observed latency_ms"),
            }
        # An allowlist prevents gold answers, model scores, or arbitrary labels
        # elsewhere in the case from leaking into production decision logic.
        routing_case = {key: copy.deepcopy(case[key]) for key in ("case_id", "group_id", "task", "features", "messages", "request_id") if key in case}
        routing_case["allowed_models"] = list(allowed)
        try:
            old_model = rank(copy.deepcopy(baseline["history"]), copy.deepcopy(routing_case))
            new_model = rank(copy.deepcopy(candidate["history"]), copy.deepcopy(routing_case))
        except Exception as exc:
            raise ValueError(f"router failed while evaluating case {case_id}") from exc
        if not isinstance(old_model, str) or not isinstance(new_model, str) or old_model not in allowed or new_model not in allowed:
            raise ValueError("router selected an ineligible or unobserved model; evaluation cannot omit this case")
        old, new = observations[old_model], observations[new_model]
        candidate_scores.append(new["score"])
        baseline_scores.append(old["score"])
        candidate_costs.append(new["cost"])
        baseline_costs.append(old["cost"])
        candidate_latencies.append(new["latency_ms"])
        baseline_latencies.append(old["latency_ms"])
        tasks[task].append((new["score"], old["score"]))
        decisions.append({"case_hash": digest(case_id), "group_hash": digest(group_id), "task": task,
                          "candidate_model": new_model, "baseline_model": old_model,
                          "observations_hash": digest(observations), "candidate": new, "baseline": old})
    count = len(cases)
    metrics = {
        "cases": count, "expected_cases": count, "missing_observations": 0,
        "overlapping_groups": 0, "overlapping_cases": 0,
        "candidate_quality": sum(candidate_scores) / count,
        "baseline_quality": sum(baseline_scores) / count,
        "quality_delta": sum(new - old for new, old in zip(candidate_scores, baseline_scores)) / count,
        "candidate_cost": sum(candidate_costs), "baseline_cost": sum(baseline_costs),
        "candidate_p95_latency_ms": _p95(candidate_latencies),
        "baseline_p95_latency_ms": _p95(baseline_latencies),
        "changed_decisions": sum(row["candidate_model"] != row["baseline_model"] for row in decisions),
    }
    report = {
        "schema": 1, "integrity": "complete", "thresholds": config,
        "candidate_history_hash": candidate["history_hash"], "baseline_history_hash": baseline["history_hash"],
        "dataset_id": dataset_id, "dataset_hash": digest(cases), "metrics": metrics,
        "context_fingerprint": context_fingerprint,
        "tasks": {task: {"cases": len(pairs), "quality_delta": sum(new - old for new, old in pairs) / len(pairs)} for task, pairs in tasks.items()},
        "decisions": decisions,
        "method": "paired held-out router decisions; empirical mean quality and nearest-rank p95 latency",
    }
    return store._save_evaluation(profile_id=profile_id, version_id=version_id,
                                  baseline_version_id=baseline["version_id"], report=report)
