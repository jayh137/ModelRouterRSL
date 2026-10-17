import json
from typing import Literal

from pydantic import BaseModel, Field, model_validator


DEFAULT_MODEL_ALIASES = {"default", "auto", "profile-default"}


def _normalize_model_list(value) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        return None
    return set(items)


class RoutingPolicy(BaseModel):
    default_model: str | None = None
    preferred_providers: list[str] = Field(default_factory=list)
    allowed_models: list[str] | None = None
    deny_models: list[str] = Field(default_factory=list)
    keyword_routing_enabled: bool | None = None
    semantic_routing_enabled: bool | None = None
    cascade_enabled: bool | None = None
    cascade_override_chain: list[str] | None = None
    cost_bias: float | None = None
    latency_bias: float | None = None
    quality_bias: float | None = None
    recommendation_enabled: bool = True
    recommendation_mode: str = "auto_top1"
    recommendation_objective: str = "auto"
    recommendation_weights: dict[str, float] | None = None
    recommendation_confidence_threshold: float | None = None
    recommendation_apply_to_explicit: bool = False
    rsl_enabled: bool = False
    rsl_mode: Literal["collect", "active"] = "collect"
    recommendation_excluded_models: list[str] = Field(
        default_factory=lambda: ["codex-auto-review", "codex-review"]
    )

    @model_validator(mode="after")
    def validate_rsl(self):
        if self.rsl_enabled and (
            "recommendation_enabled" not in self.model_fields_set
            or not self.recommendation_enabled
        ):
            raise ValueError("RSL requires explicit recommendation_enabled=true")
        return self

    @property
    def cost_bias_value(self) -> float:
        return float(self.cost_bias or 0.0)

    @property
    def latency_bias_value(self) -> float:
        return float(self.latency_bias or 0.0)

    @property
    def quality_bias_value(self) -> float:
        return float(self.quality_bias or 0.0)

    @property
    def allowed_model_set(self) -> set[str] | None:
        return _normalize_model_list(self.allowed_models)

    @property
    def deny_model_set(self) -> set[str]:
        return _normalize_model_list(self.deny_models) or set()

    def effective_allowed_models(
        self,
        request_allowed_models: set[str] | None,
        available_models: set[str] | None = None,
    ) -> set[str] | None:
        allowed = None if request_allowed_models is None else set(request_allowed_models)
        policy_allowed = self.allowed_model_set
        if policy_allowed is not None:
            allowed = policy_allowed if allowed is None else allowed & policy_allowed

        denied = self.deny_model_set
        if allowed is not None:
            allowed -= denied
            return allowed

        if available_models is not None:
            return set(available_models) - denied
        return None

    def is_model_allowed(self, model: str, request_allowed_models: set[str] | None) -> bool:
        if request_allowed_models is not None and model not in request_allowed_models:
            return False
        policy_allowed = self.allowed_model_set
        if policy_allowed is not None and model not in policy_allowed:
            return False
        if model in self.deny_model_set:
            return False
        return True


async def load_routing_policy(db, routing_profile_id: str | None) -> RoutingPolicy:
    if not routing_profile_id or not str(routing_profile_id).isdigit():
        return RoutingPolicy()
    if not hasattr(db, "get_routing_profile"):
        return RoutingPolicy()
    profile = await db.get_routing_profile(int(routing_profile_id))
    if not profile:
        return RoutingPolicy()
    profile = dict(profile)
    if not profile.get("enabled", 1):
        return RoutingPolicy()
    raw = profile.get("config_json") or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    if "recommendation_enabled" not in data:
        data["recommendation_enabled"] = False
    return RoutingPolicy(**data)
