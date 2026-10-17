from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from copy import copy, deepcopy
import re
from typing import Any

import yaml

from app.core.routing_policy import RoutingPolicy


DEFAULT_EXCLUDED_MODELS = {"codex-auto-review", "codex-review"}
DEFAULT_PERFORMANCE_HISTORY_PATH = "config/model_performance_history.yaml"
DEFAULT_TIER_ADJUSTMENTS = {
    "economy": 0.10,
    "standard": 0.03,
    "advanced": -0.06,
    "premium": -0.12,
}
LOW_SIGNAL_PROFILE_KEYWORDS = {
    "ci",
    "sql",
    "react",
    "vue",
    "node",
    "python",
    "java",
    "go",
    "rust",
    "typescript",
    "docker",
    "kubernetes",
}
TIER_RANK = {
    "economy": 0,
    "standard": 1,
    "advanced": 2,
    "premium": 3,
}
LONG_INPUT_SOFT_CHARS = 8_000
LONG_INPUT_HARD_CHARS = 20_000
LONG_CONTEXT_PROFILE_NAMES = {
    "long_context",
    "long_context_synthesis",
    "open_ended_long_form",
    "long_context_retrieval_exact",
    "few_shot_long_context_classification",
    "repo_scale_coding",
}


@dataclass(slots=True)
class TaskProfile:
    name: str
    weights: dict[str, float]
    matched_keywords: list[str] = field(default_factory=list)
    hard_requires: list[str] = field(default_factory=list)
    objective: str | None = None
    preferred_models: list[str] = field(default_factory=list)
    excluded_models: list[str] = field(default_factory=list)
    model_boosts: dict[str, float] = field(default_factory=dict)
    fallback_model: str | None = None
    fallback_margin: float = 0.0
    max_tier: str | None = None
    allowed_tiers: list[str] = field(default_factory=list)
    min_capabilities: dict[str, float] = field(default_factory=dict)
    performance_min_scored_cases: int | None = None
    performance_history_weight: float | None = None
    performance_confidence_case_scale: float | None = None
    confidence: float = 0.0


@dataclass(slots=True)
class RecommendationResult:
    applied: bool
    selected_model: str | None
    route_reason: str
    confidence: float
    task_profile: dict[str, Any]
    candidates: list[dict[str, Any]]
    excluded: list[dict[str, Any]]
    objective: dict[str, float]
    selection_policy: dict[str, Any] = field(default_factory=dict)
    eligible_models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "applied": self.applied,
            "selected_model": self.selected_model,
            "route_reason": self.route_reason,
            "confidence": round(self.confidence, 4),
            "task_profile": self.task_profile,
            "candidates": self.candidates,
            "excluded": self.excluded,
            "objective": self.objective,
            "selection_policy": self.selection_policy,
        }


class ModelRecommender:
    def __init__(
        self,
        *,
        model_traits_path: str = "config/model_traits.yaml",
        task_profiles_path: str = "config/task_profiles.yaml",
        runtime_signals_path: str = "config/model_runtime_signals.yaml",
        performance_history_path: str = DEFAULT_PERFORMANCE_HISTORY_PATH,
    ):
        self.model_traits_path = Path(model_traits_path)
        self.task_profiles_path = Path(task_profiles_path)
        self.runtime_signals_path = Path(runtime_signals_path)
        self.performance_history_path = Path(performance_history_path)
        self.model_traits: dict[str, dict[str, Any]] = {}
        self.runtime_signals: dict[str, dict[str, Any]] = {}
        self.performance_history: dict[str, Any] = {}
        self.task_profiles: dict[str, dict[str, Any]] = {}
        self.objectives: dict[str, dict[str, float]] = {}
        self.excluded_models = set(DEFAULT_EXCLUDED_MODELS)
        self.tier_adjustments = dict(DEFAULT_TIER_ADJUSTMENTS)
        self.price_cost_weight = 0.95
        self.price_cost_scale = 4.0
        self.success_rate_weight = 0.06
        self.provider_health_blend_weight = 0.30
        self.success_rate_soft_floor = 0.60
        self.success_rate_soft_penalty = 0.02
        self.success_rate_hard_floor = 0.35
        self.success_rate_hard_penalty = 0.08
        self.performance_history_enabled = True
        self.performance_history_weight = 0.12
        self.performance_history_min_scored_cases = 3
        self.performance_history_confidence_case_scale = 30.0
        self.performance_history_bad_output_penalty = 0.05
        self.reload()

    def reload(self):
        model_data = self._read_yaml(self.model_traits_path)
        task_data = self._read_yaml(self.task_profiles_path)
        runtime_data = self._read_yaml(self.runtime_signals_path)
        performance_data = self._read_yaml(self.performance_history_path)
        self.model_traits = model_data.get("models", {}) or {}
        metadata = model_data.get("metadata", {}) or {}
        runtime_metadata = runtime_data.get("metadata", {}) or {}
        self.runtime_signals = runtime_data.get("models", {}) or {}
        self.performance_history = performance_data if isinstance(performance_data, dict) else {}
        self.excluded_models = set(DEFAULT_EXCLUDED_MODELS)
        self.excluded_models.update(metadata.get("hard_excluded_models") or [])
        self.tier_adjustments = {
            **DEFAULT_TIER_ADJUSTMENTS,
            **(metadata.get("tier_adjustments") or {}),
        }
        self.price_cost_weight = float(metadata.get("price_cost_weight", 0.95))
        self.price_cost_scale = float(metadata.get("price_cost_scale", 4.0))
        self.success_rate_weight = float(
            runtime_metadata.get("success_rate_weight", metadata.get("success_rate_weight", 0.06))
        )
        self.provider_health_blend_weight = float(
            runtime_metadata.get(
                "provider_health_blend_weight",
                metadata.get("provider_health_blend_weight", 0.30),
            )
        )
        self.success_rate_soft_floor = float(
            runtime_metadata.get("success_rate_soft_floor", metadata.get("success_rate_soft_floor", 0.60))
        )
        self.success_rate_soft_penalty = float(
            runtime_metadata.get(
                "success_rate_soft_penalty",
                metadata.get("success_rate_soft_penalty", 0.02),
            )
        )
        self.success_rate_hard_floor = float(
            runtime_metadata.get("success_rate_hard_floor", metadata.get("success_rate_hard_floor", 0.35))
        )
        self.success_rate_hard_penalty = float(
            runtime_metadata.get(
                "success_rate_hard_penalty",
                metadata.get("success_rate_hard_penalty", 0.08),
            )
        )
        self._load_performance_settings()
        self.task_profiles = task_data.get("tasks", {}) or {}
        self.objectives = task_data.get("objectives", {}) or {}

    def with_performance_history(self, history: dict[str, Any]) -> "ModelRecommender":
        """A request-local snapshot; activation must never mutate the shared router."""
        snapshot = copy(self)
        snapshot.performance_history = deepcopy(history)
        snapshot._load_performance_settings()
        return snapshot

    def _load_performance_settings(self):
        performance_metadata = self.performance_history.get("metadata") or {}
        performance_adjustment = performance_metadata.get("performance_adjustment") or {}
        self.performance_history_enabled = bool(
            performance_adjustment.get("enabled", performance_metadata.get("enabled", True))
        )
        self.performance_history_weight = float(
            performance_adjustment.get("weight", performance_metadata.get("weight", 0.12))
        )
        self.performance_history_min_scored_cases = int(
            performance_adjustment.get(
                "min_scored_cases",
                performance_metadata.get("min_scored_cases", 3),
            )
        )
        self.performance_history_confidence_case_scale = float(
            performance_adjustment.get(
                "confidence_case_scale",
                performance_metadata.get("confidence_case_scale", 30),
            )
        )
        self.performance_history_bad_output_penalty = float(
            performance_adjustment.get(
                "bad_output_penalty",
                performance_metadata.get("bad_output_penalty", 0.05),
            )
        )

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
        return data if isinstance(data, dict) else {}

    def _conversation_text(self, messages: list) -> str:
        parts = []
        for msg in messages:
            role = msg.role if hasattr(msg, "role") else msg.get("role", "")
            if role in ("user", "assistant"):
                content = msg.content if hasattr(msg, "content") else msg.get("content", "")
                if content:
                    parts.append(str(content))
        return "\n".join(parts)

    def _keyword_matches(self, keyword: str, text: str, lowered: str) -> bool:
        keyword = keyword.strip()
        if not keyword:
            return False
        if keyword.isascii() and re.fullmatch(r"[A-Za-z][A-Za-z0-9_+#.-]*", keyword):
            # Short technical words like "CI" and "Go" should not match
            # ordinary words such as "specific", "decision", or "Guardian".
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(keyword)}(?![A-Za-z0-9_])"
            return re.search(pattern, text, flags=re.IGNORECASE) is not None
        return keyword.lower() in lowered

    def _keyword_weight(self, keyword: str) -> float:
        if keyword.strip().lower() in LOW_SIGNAL_PROFILE_KEYWORDS:
            return 0.35
        return 1.0

    def _objective_weights(self, policy: RoutingPolicy, task: TaskProfile) -> tuple[dict[str, float], str, str]:
        configured = policy.recommendation_weights or {}
        if configured:
            return {
                "quality": float(configured.get("quality", 0.75)),
                "cost": float(configured.get("cost", 0.15)),
                "latency": float(configured.get("latency", 0.10)),
            }, "custom", "custom"
        name = (policy.recommendation_objective or "auto").strip().lower()
        source = "policy"
        if name in {"auto", "profile", "profile_default"}:
            name = (task.objective or "budget").strip().lower()
            source = "profile"
        preset = self.objectives.get(name) or self.objectives.get("budget") or self.objectives.get("balanced") or {}
        return {
            "quality": float(preset.get("quality", 0.75)),
            "cost": float(preset.get("cost", 0.15)),
            "latency": float(preset.get("latency", 0.10)),
        }, name, source

    def profile_task(self, messages: list) -> TaskProfile:
        text = self._conversation_text(messages)
        lowered = text.lower()
        best_name = "general"
        best_score = 0.0
        best_keywords: list[str] = []
        text_len = len(text)

        for name, profile in self.task_profiles.items():
            if profile.get("default"):
                continue
            matched = []
            for keyword in profile.get("keywords") or []:
                value = str(keyword).strip()
                if self._keyword_matches(value, text, lowered):
                    matched.append(value)
            score = sum(self._keyword_weight(value) for value in matched)
            if name in LONG_CONTEXT_PROFILE_NAMES and text_len >= LONG_INPUT_SOFT_CHARS:
                score += 0.75
                if text_len >= LONG_INPUT_HARD_CHARS:
                    score += 0.75
            if score > best_score and score >= 1.0:
                best_name = name
                best_score = score
                best_keywords = matched[:8]

        selected = self.task_profiles.get(best_name) or self.task_profiles.get("general") or {}
        weights = {
            str(key): float(value)
            for key, value in (selected.get("weights") or {}).items()
        }
        if not weights:
            weights = {"reasoning": 0.35, "instruction_following": 0.35, "writing": 0.30}
        model_boosts = {
            str(key): float(value)
            for key, value in (selected.get("model_boosts") or {}).items()
        }
        min_capabilities = {
            str(key): float(value)
            for key, value in (selected.get("min_capabilities") or {}).items()
        }
        confidence = min(1.0, best_score / 4.0) if best_score else 0.2
        return TaskProfile(
            name=best_name,
            weights=weights,
            matched_keywords=best_keywords,
            hard_requires=[str(item) for item in (selected.get("hard_requires") or [])],
            objective=(str(selected.get("objective")).strip() if selected.get("objective") else None),
            preferred_models=[str(item) for item in (selected.get("preferred_models") or [])],
            excluded_models=[str(item) for item in (selected.get("excluded_models") or [])],
            model_boosts=model_boosts,
            fallback_model=(str(selected.get("fallback_model")).strip() if selected.get("fallback_model") else None),
            fallback_margin=float(selected.get("fallback_margin") or 0.0),
            max_tier=(str(selected.get("max_tier")).strip().lower() if selected.get("max_tier") else None),
            allowed_tiers=[str(item).strip().lower() for item in (selected.get("allowed_tiers") or [])],
            min_capabilities=min_capabilities,
            performance_min_scored_cases=(
                int(selected["performance_min_scored_cases"])
                if selected.get("performance_min_scored_cases") is not None
                else None
            ),
            performance_history_weight=(
                float(selected["performance_history_weight"])
                if selected.get("performance_history_weight") is not None
                else None
            ),
            performance_confidence_case_scale=(
                float(selected["performance_confidence_case_scale"])
                if selected.get("performance_confidence_case_scale") is not None
                else None
            ),
            confidence=confidence,
        )

    def _trait_for(self, routing_key: str, model_name: str) -> dict[str, Any]:
        return self.model_traits.get(routing_key) or self.model_traits.get(model_name) or {}

    def _runtime_signal_for(self, routing_key: str, model_name: str) -> dict[str, Any]:
        return self.runtime_signals.get(routing_key) or self.runtime_signals.get(model_name) or {}

    def _normalize_rate(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            text = value.strip().rstrip("%")
            if not text:
                return None
            try:
                value = float(text)
            except ValueError:
                return None
        if not isinstance(value, (int, float)):
            return None
        rate = float(value)
        if rate > 1.0:
            rate /= 100.0
        return max(0.0, min(1.0, rate))

    def _is_excluded(self, routing_key: str, model_name: str, policy: RoutingPolicy) -> bool:
        policy_excluded = set(policy.recommendation_excluded_models or [])
        return (
            routing_key in self.excluded_models
            or model_name in self.excluded_models
            or routing_key in policy_excluded
            or model_name in policy_excluded
        )

    def _provider_health_score(self, router_engine, provider_name: str) -> float:
        provider = router_engine.providers.get(provider_name)
        health = getattr(provider, "health", None)
        if not health:
            return 1.0
        snapshot = health.snapshot()
        success_rate = self._normalize_rate(snapshot.get("success_rate"))
        if success_rate is not None:
            return success_rate
        return 1.0

    def _model_success_rate_signal(
        self,
        routing_key: str,
        model_name: str,
        traits: dict[str, Any],
    ) -> tuple[float | None, str | None, str | None]:
        runtime_signal = self._runtime_signal_for(routing_key, model_name)
        for source_name, payload in (
            ("runtime_signals", runtime_signal),
            ("model_traits", traits),
        ):
            if not isinstance(payload, dict):
                continue
            for field_name in ("success_rate", "success_rate_percent"):
                rate = self._normalize_rate(payload.get(field_name))
                if rate is not None:
                    updated_at = payload.get("updated_at") or payload.get("last_updated")
                    return rate, source_name, str(updated_at) if updated_at else None
        return None, None, None

    def _reliability_signal(
        self,
        *,
        routing_key: str,
        model_name: str,
        provider_name: str,
        traits: dict[str, Any],
        router_engine,
    ) -> dict[str, Any]:
        model_success_rate, model_source, model_updated_at = self._model_success_rate_signal(
            routing_key,
            model_name,
            traits,
        )
        provider_success_rate = self._provider_health_score(router_engine, provider_name)
        blend_weight = max(0.0, min(1.0, self.provider_health_blend_weight))

        if model_success_rate is not None:
            success_score = ((1.0 - blend_weight) * model_success_rate) + (blend_weight * provider_success_rate)
            success_source = f"{model_source or 'model'}+provider_health"
        else:
            success_score = provider_success_rate
            success_source = "provider_health"

        soft_floor = max(0.0, min(1.0, self.success_rate_soft_floor))
        hard_floor = max(0.0, min(soft_floor, self.success_rate_hard_floor))
        soft_penalty = max(0.0, float(self.success_rate_soft_penalty))
        hard_penalty = max(0.0, float(self.success_rate_hard_penalty))
        penalty = 0.0

        if success_score < hard_floor and hard_floor > 0:
            gap = (hard_floor - success_score) / hard_floor
            penalty -= hard_penalty * gap
        elif success_score < soft_floor and soft_floor > 0:
            gap = (soft_floor - success_score) / soft_floor
            penalty -= soft_penalty * gap

        return {
            "score": success_score,
            "penalty": penalty,
            "model_success_rate": model_success_rate,
            "provider_success_rate": provider_success_rate,
            "source": success_source,
            "updated_at": model_updated_at,
        }

    def _capability_score(self, traits: dict[str, Any], task: TaskProfile) -> float:
        total_weight = sum(max(0.0, value) for value in task.weights.values())
        if total_weight <= 0:
            return 0.0
        score = 0.0
        for capability, weight in task.weights.items():
            score += max(0.0, weight) * (float(traits.get(capability, 50.0)) / 100.0)
        return score / total_weight

    def _pricing_unit_cost(self, model_config) -> float | None:
        pricing = getattr(model_config, "pricing", None)
        if not pricing:
            return None
        return max(0.0, float(pricing.prompt) + float(pricing.completion))

    def _pricing_score(self, model_config) -> float | None:
        blended_unit_cost = self._pricing_unit_cost(model_config)
        if blended_unit_cost is None:
            return None
        scale = max(0.1, float(self.price_cost_scale))
        return 1.0 / (1.0 + scale * blended_unit_cost)

    def _cost_score(self, traits: dict[str, Any], model_config) -> tuple[float, float | None]:
        trait_score = float(traits.get("cost_efficiency", 50.0)) / 100.0
        price_score = self._pricing_score(model_config)
        if price_score is None:
            return trait_score, None
        price_weight = max(0.0, min(1.0, self.price_cost_weight))
        return ((1.0 - price_weight) * trait_score) + (price_weight * price_score), price_score

    def _tier_adjustment(self, traits: dict[str, Any], objective: dict[str, float]) -> tuple[str, float]:
        tier = str(traits.get("tier") or "standard").strip().lower()
        raw_adjustment = float(self.tier_adjustments.get(tier, 0.0))
        economy_pressure = max(0.0, min(1.0, objective.get("cost", 0.0) + objective.get("latency", 0.0)))
        return tier, raw_adjustment * economy_pressure

    def _profile_adjustment(self, routing_key: str, model_name: str, task: TaskProfile) -> float:
        adjustment = 0.0
        if routing_key in task.preferred_models or model_name in task.preferred_models:
            adjustment += 0.02
        adjustment += float(task.model_boosts.get(routing_key, task.model_boosts.get(model_name, 0.0)))
        return adjustment

    def _as_float(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _as_int(self, value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _performance_signal(self, routing_key: str, model_name: str, task: TaskProfile) -> dict[str, Any]:
        neutral = {
            "adjustment": 0.0,
            "source": None,
            "cases": 0,
            "scored_cases": 0,
            "score_avg": None,
            "pass_rate": None,
            "confidence": 0.0,
            "quality_delta": None,
            "pass_rate_delta": None,
            "bad_output_rate": None,
        }
        if not self.performance_history_enabled:
            return {**neutral, "source": "disabled"}

        profiles = self.performance_history.get("profiles")
        if not isinstance(profiles, dict):
            return neutral

        profile = profiles.get(task.name)
        if not isinstance(profile, dict):
            return neutral

        models = profile.get("models")
        if not isinstance(models, dict):
            return neutral

        record = models.get(routing_key) or models.get(model_name)
        if not isinstance(record, dict):
            return neutral

        scored_cases = self._as_int(record.get("scored_cases"))
        cases = self._as_int(record.get("cases"))
        score_avg = self._as_float(record.get("score_avg"))
        pass_rate = self._as_float(record.get("pass_rate"))
        bad_output_rate = self._as_float(record.get("bad_output_rate")) or 0.0

        min_scored_cases = (
            task.performance_min_scored_cases
            if task.performance_min_scored_cases is not None
            else self.performance_history_min_scored_cases
        )
        if scored_cases < max(1, min_scored_cases):
            return {
                **neutral,
                "source": "insufficient_history",
                "cases": cases,
                "scored_cases": scored_cases,
                "score_avg": score_avg,
                "pass_rate": pass_rate,
                "bad_output_rate": bad_output_rate,
            }

        summary = profile.get("summary") if isinstance(profile.get("summary"), dict) else {}
        profile_score_avg = self._as_float(summary.get("score_avg"))
        profile_pass_rate = self._as_float(summary.get("pass_rate"))
        if profile_score_avg is None:
            profile_score_avg = score_avg if score_avg is not None else 0.0
        if profile_pass_rate is None:
            profile_pass_rate = pass_rate if pass_rate is not None else 0.0
        score_avg = score_avg if score_avg is not None else profile_score_avg
        pass_rate = pass_rate if pass_rate is not None else profile_pass_rate

        quality_delta = score_avg - profile_score_avg
        pass_rate_delta = pass_rate - profile_pass_rate
        confidence_case_scale = (
            task.performance_confidence_case_scale
            if task.performance_confidence_case_scale is not None
            else self.performance_history_confidence_case_scale
        )
        confidence_scale = max(1.0, float(confidence_case_scale))
        confidence = min(1.0, scored_cases / confidence_scale)
        relative_signal = (0.65 * quality_delta) + (0.35 * pass_rate_delta)
        history_weight = (
            task.performance_history_weight
            if task.performance_history_weight is not None
            else self.performance_history_weight
        )
        adjustment = (
            float(history_weight) * confidence * relative_signal
            - max(0.0, float(self.performance_history_bad_output_penalty)) * confidence * bad_output_rate
        )
        return {
            "adjustment": adjustment,
            "source": "benchmark_history",
            "cases": cases,
            "scored_cases": scored_cases,
            "score_avg": score_avg,
            "pass_rate": pass_rate,
            "confidence": confidence,
            "quality_delta": quality_delta,
            "pass_rate_delta": pass_rate_delta,
            "bad_output_rate": bad_output_rate,
        }

    def _exclude_reason(
        self,
        *,
        routing_key: str,
        model_name: str,
        traits: dict[str, Any],
        task: TaskProfile,
        allowed_models: set[str] | None,
        policy: RoutingPolicy,
        router_engine=None,
    ) -> str | None:
        if allowed_models is not None and routing_key not in allowed_models:
            return "not_in_allowed_models"
        if router_engine is not None and hasattr(router_engine, "is_model_available"):
            if not router_engine.is_model_available(routing_key):
                availability = (
                    router_engine.model_availability_snapshot(routing_key)
                    if hasattr(router_engine, "model_availability_snapshot")
                    else None
                )
                if availability:
                    return (
                        f"model_unavailable_until:{availability.get('unavailable_until_at')}"
                        f":{availability.get('reason', '')[:120]}"
                    )
                return "model_temporarily_unavailable"
        if self._is_excluded(routing_key, model_name, policy):
            return "recommendation_excluded"
        if routing_key in task.excluded_models or model_name in task.excluded_models:
            return "profile_excluded"
        tier = str(traits.get("tier") or "standard").strip().lower()
        if task.allowed_tiers and tier not in task.allowed_tiers:
            return f"tier_not_allowed:{tier}"
        if task.max_tier and TIER_RANK.get(tier, 1) > TIER_RANK.get(task.max_tier, 1):
            return f"tier_above_profile_max:{tier}"
        for capability in task.hard_requires:
            if float(traits.get(capability, 0.0)) <= 0:
                return f"missing_required_capability:{capability}"
        for capability, minimum in task.min_capabilities.items():
            if float(traits.get(capability, 0.0)) < minimum:
                return f"below_min_capability:{capability}"
        return None

    def _select_candidate(
        self,
        candidates: list[dict[str, Any]],
        task: TaskProfile,
        objective_source: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if not candidates:
            return None, {}
        selected = candidates[0]
        policy = {
            "profile": task.name,
            "objective": task.objective,
            "objective_source": objective_source,
            "fallback_model": task.fallback_model,
            "fallback_margin": task.fallback_margin,
            "fallback_applied": False,
        }
        if not task.fallback_model or objective_source != "profile":
            return selected, policy

        fallback = next(
            (
                candidate
                for candidate in candidates
                if candidate["model"] == task.fallback_model or candidate["model_name"] == task.fallback_model
            ),
            None,
        )
        if not fallback or fallback["model"] == selected["model"]:
            return selected, policy

        margin = max(0.0, task.fallback_margin)
        if float(fallback["score"]) + margin >= float(selected["score"]):
            policy.update(
                {
                    "fallback_applied": True,
                    "original_top_model": selected["model"],
                    "original_top_score": selected["score"],
                    "selected_score": fallback["score"],
                }
            )
            return fallback, policy
        return selected, policy

    def recommend(
        self,
        *,
        messages: list,
        router_engine,
        policy: RoutingPolicy,
        allowed_models: set[str] | None = None,
    ) -> RecommendationResult:
        task = self.profile_task(messages)
        objective, objective_name, objective_source = self._objective_weights(policy, task)
        candidates: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []

        for routing_key, (provider_name, model_config) in router_engine.model_map.items():
            traits = self._trait_for(routing_key, model_config.name)
            reason = self._exclude_reason(
                routing_key=routing_key,
                model_name=model_config.name,
                traits=traits,
                task=task,
                allowed_models=allowed_models,
                policy=policy,
                router_engine=router_engine,
            )
            if reason:
                excluded.append({
                    "model": routing_key,
                    "model_name": model_config.name,
                    "reason": reason,
                })
                continue

            capability_score = self._capability_score(traits, task)
            cost_score, price_score = self._cost_score(traits, model_config)
            price_unit_cost = self._pricing_unit_cost(model_config)
            latency_score = float(traits.get("latency", 50.0)) / 100.0
            health_score = self._provider_health_score(router_engine, provider_name)
            reliability = self._reliability_signal(
                routing_key=routing_key,
                model_name=model_config.name,
                provider_name=provider_name,
                traits=traits,
                router_engine=router_engine,
            )
            operator_boost = float(traits.get("operator_boost", 0.0)) / 100.0
            tier, tier_adjustment = self._tier_adjustment(traits, objective)
            profile_adjustment = self._profile_adjustment(routing_key, model_config.name, task)
            performance = self._performance_signal(routing_key, model_config.name, task)
            final_score = (
                objective["quality"] * capability_score
                + objective["cost"] * cost_score
                + objective["latency"] * latency_score
                + self.success_rate_weight * float(reliability["score"])
                + float(reliability["penalty"])
                + tier_adjustment
                + profile_adjustment
                + float(performance["adjustment"])
                + operator_boost
            )
            candidates.append(
                {
                    "model": routing_key,
                    "model_name": model_config.name,
                    "provider": provider_name,
                    "score": round(final_score, 6),
                    "capability_score": round(capability_score, 6),
                    "cost_score": round(cost_score, 6),
                    "price_score": round(price_score, 6) if price_score is not None else None,
                    "price_unit_cost": round(price_unit_cost, 6) if price_unit_cost is not None else None,
                    "latency_score": round(latency_score, 6),
                    "health_score": round(health_score, 6),
                    "model_success_rate": round(float(reliability["model_success_rate"]), 6)
                    if isinstance(reliability["model_success_rate"], (int, float))
                    else None,
                    "provider_success_rate": round(float(reliability["provider_success_rate"]), 6),
                    "reliability_score": round(float(reliability["score"]), 6),
                    "reliability_penalty": round(float(reliability["penalty"]), 6),
                    "reliability_source": reliability["source"],
                    "reliability_updated_at": reliability["updated_at"],
                    "tier": tier,
                    "tier_adjustment": round(tier_adjustment, 6),
                    "profile_adjustment": round(profile_adjustment, 6),
                    "performance_adjustment": round(float(performance["adjustment"]), 6),
                    "performance_source": performance["source"],
                    "performance_cases": performance["cases"],
                    "performance_scored_cases": performance["scored_cases"],
                    "performance_score_avg": round(float(performance["score_avg"]), 6)
                    if isinstance(performance["score_avg"], (int, float))
                    else None,
                    "performance_pass_rate": round(float(performance["pass_rate"]), 6)
                    if isinstance(performance["pass_rate"], (int, float))
                    else None,
                    "performance_confidence": round(float(performance["confidence"]), 6),
                    "performance_quality_delta": round(float(performance["quality_delta"]), 6)
                    if isinstance(performance["quality_delta"], (int, float))
                    else None,
                    "performance_pass_rate_delta": round(float(performance["pass_rate_delta"]), 6)
                    if isinstance(performance["pass_rate_delta"], (int, float))
                    else None,
                    "performance_bad_output_rate": round(float(performance["bad_output_rate"]), 6)
                    if isinstance(performance["bad_output_rate"], (int, float))
                    else None,
                }
            )

        candidates.sort(key=lambda item: (-item["score"], item["provider"], item["model"]))
        selected_candidate, selection_policy = self._select_candidate(candidates, task, objective_source)
        top_candidates = candidates[:5]
        if selected_candidate and selected_candidate not in top_candidates:
            top_candidates = top_candidates[:4] + [selected_candidate]
        selected = selected_candidate["model"] if selected_candidate else None
        confidence = 0.0
        if selected_candidate and len(candidates) == 1:
            confidence = selected_candidate["score"]
        elif selected_candidate and len(candidates) >= 2:
            best_other = next(
                (item for item in candidates if item["model_name"] != selected_candidate["model_name"]),
                None,
            )
            if best_other:
                confidence = max(0.0, float(selected_candidate["score"]) - float(best_other["score"]))

        threshold = float(policy.recommendation_confidence_threshold or 0.0)
        mode = (policy.recommendation_mode or "auto_top1").strip().lower()
        suppressed = mode in {"off", "shadow", "preview", "preview_only"}
        applied = bool(selected and confidence >= threshold and not suppressed)
        route_reason = "recommendation"
        if suppressed and selected:
            route_reason = f"recommendation_{mode}"
        elif not applied:
            route_reason = "recommendation_low_confidence"
        return RecommendationResult(
            applied=applied,
            selected_model=selected,
            route_reason=route_reason,
            confidence=confidence,
            task_profile={
                "name": task.name,
                "confidence": round(task.confidence, 4),
                "matched_keywords": task.matched_keywords,
                "weights": task.weights,
                "hard_requires": task.hard_requires,
                "objective": task.objective,
                "objective_used": objective_name,
                "objective_source": objective_source,
                "preferred_models": task.preferred_models,
                "fallback_model": task.fallback_model,
                "fallback_margin": task.fallback_margin,
                "max_tier": task.max_tier,
                "allowed_tiers": task.allowed_tiers,
                "min_capabilities": task.min_capabilities,
                "performance_min_scored_cases": task.performance_min_scored_cases,
                "performance_history_weight": task.performance_history_weight,
                "performance_confidence_case_scale": task.performance_confidence_case_scale,
            },
            candidates=top_candidates,
            excluded=excluded[:20],
            objective=objective,
            selection_policy=selection_policy,
            eligible_models=[item["model"] for item in candidates],
        )
