import json
import time
import logging
from datetime import datetime, timezone

from app.core.db import Database
from app.core.request_context import RequestContext
from app.core.router import RouterEngine
from app.providers.base import BaseProvider
from app.schemas.chat import ChatRequest, ChatResponse, CascadeMetadata
from app.cascade.validator import ResponseValidator
from app.cascade.local_judge import LocalJudge, LocalJudgeResult
from app.cascade.webhook import WebhookValidator

logger = logging.getLogger("model_router.cascade")


class CascadeEngine:
    def __init__(self, db: Database, router: RouterEngine):
        self.db = db
        self.router = router
        self._rules_cache: dict[tuple[str | None, str], dict] = {}

    async def reload(self):
        self._rules_cache.clear()
        rules = await self.db.list_cascade_rules(enabled_only=True)
        for rule in rules:
            rule = dict(rule)
            rule_profile_id = rule.get("routing_profile_id")
            key = (str(rule_profile_id) if rule_profile_id is not None else None, rule["source_routing_key"])
            if key not in self._rules_cache:
                self._rules_cache[key] = rule
        logger.info("Cascade engine reloaded with %d rules", len(rules))

    def get_rule(self, source_routing_key: str, routing_profile_id: str | None = None) -> dict | None:
        if routing_profile_id is not None:
            rule = self._rules_cache.get((str(routing_profile_id), source_routing_key))
            if rule:
                return rule
        return self._rules_cache.get((None, source_routing_key))

    async def resolve_chain(
        self,
        source_routing_key: str,
        cascade_config: dict | None = None,
        routing_profile_id: str | None = None,
    ) -> list[tuple[str, BaseProvider, object]] | None:
        if cascade_config and cascade_config.get("enabled") is False:
            return None

        chain_keys = None
        if cascade_config and cascade_config.get("chain"):
            chain_keys = cascade_config["chain"]
        else:
            rule = self.get_rule(source_routing_key, routing_profile_id=routing_profile_id)
            if rule:
                chain_keys = json.loads(rule["escalation_chain"])

        if not chain_keys:
            return None

        result = []
        for routing_key in chain_keys:
            if routing_key in self.router.model_map:
                provider_name, model_config = self.router.model_map[routing_key]
                if provider_name in self.router.providers:
                    provider = self.router.providers[provider_name]
                    result.append((routing_key, provider, model_config))

        return result if len(result) > 1 else None

    def get_validator(
        self,
        source_routing_key: str,
        cascade_config: dict | None = None,
        routing_profile_id: str | None = None,
    ) -> ResponseValidator:
        min_length = 10
        check_refusal = True
        refusal_patterns = None
        check_finish_reason = True

        if cascade_config:
            if cascade_config.get("min_content_length") is not None:
                min_length = cascade_config["min_content_length"]

        rule = self.get_rule(source_routing_key, routing_profile_id=routing_profile_id)
        if rule:
            min_length = rule.get("min_content_length", min_length)
            check_refusal = bool(rule.get("check_refusal", 1))
            refusal_patterns_str = rule.get("refusal_patterns", "[]")
            if isinstance(refusal_patterns_str, str):
                try:
                    refusal_patterns = json.loads(refusal_patterns_str)
                except json.JSONDecodeError:
                    refusal_patterns = None
            check_finish_reason = bool(rule.get("check_finish_reason", 1))

        return ResponseValidator(
            min_content_length=min_length,
            check_refusal=check_refusal,
            refusal_patterns=refusal_patterns,
            check_finish_reason=check_finish_reason,
        )

    def get_webhook_validator(self, source_routing_key: str, routing_profile_id: str | None = None) -> WebhookValidator | None:
        rule = self.get_rule(source_routing_key, routing_profile_id=routing_profile_id)
        if rule and rule.get("custom_validator_url"):
            return WebhookValidator(
                url=rule["custom_validator_url"],
                timeout=rule.get("custom_validator_timeout", 5000),
            )
        return None

    def get_local_judge(
        self,
        source_routing_key: str,
        cascade_config: dict | None = None,
        routing_profile_id: str | None = None,
    ) -> LocalJudge | None:
        rule = self.get_rule(source_routing_key, routing_profile_id=routing_profile_id)

        judge_model = None
        judge_threshold = 0.7
        judge_system_prompt = None

        if rule:
            judge_model = (rule.get("local_judge_model") or "").strip() or None
            if rule.get("local_judge_threshold") is not None:
                judge_threshold = rule.get("local_judge_threshold", judge_threshold)
            judge_system_prompt = rule.get("local_judge_system_prompt")

        if cascade_config:
            judge_model = (cascade_config.get("judge_model") or judge_model or "").strip() or None
            if cascade_config.get("judge_threshold") is not None:
                judge_threshold = cascade_config["judge_threshold"]
            if cascade_config.get("judge_system_prompt") is not None:
                judge_system_prompt = cascade_config.get("judge_system_prompt")

        if not judge_model:
            return None
        if judge_model not in self.router.model_map:
            logger.warning("Local judge model %s is not available for cascade rule %s", judge_model, source_routing_key)
            return None

        provider_name, _ = self.router.model_map[judge_model]
        if provider_name not in self.router.providers:
            logger.warning("Local judge provider %s is not loaded for model %s", provider_name, judge_model)
            return None

        return LocalJudge(
            router=self.router,
            judge_model=judge_model,
            threshold=judge_threshold,
            system_prompt=judge_system_prompt,
        )

    def get_max_escalations(
        self,
        source_routing_key: str,
        cascade_config: dict | None = None,
        routing_profile_id: str | None = None,
    ) -> int:
        if cascade_config and cascade_config.get("max_escalations") is not None:
            return cascade_config["max_escalations"]
        rule = self.get_rule(source_routing_key, routing_profile_id=routing_profile_id)
        if rule:
            return rule.get("max_escalations", 3)
        return 3

    async def execute_cascade(
        self,
        request: ChatRequest,
        original_model: str,
        chain: list[tuple[str, BaseProvider, object]],
        validator: ResponseValidator,
        local_judge: LocalJudge | None,
        webhook_validator: WebhookValidator | None,
        max_escalations: int,
        api_key_id: int = 0,
        rsl_trace=None,
    ) -> tuple[ChatResponse, CascadeMetadata]:
        models_attempted = []
        validation_reasons = []
        last_response = None
        last_provider = None
        last_model_config = None
        last_judge_result = None
        last_response_model = None

        start_time = time.monotonic()

        for idx, (routing_key, provider, model_config) in enumerate(chain):
            if idx >= max_escalations + 1:
                break

            models_attempted.append(routing_key)

            try:
                if rsl_trace:
                    response = await rsl_trace.complete(provider, request, model_config, routing_key)
                else:
                    response = await provider.chat_completions(request, model_config)
            except Exception as e:
                logger.warning("Cascade step %d failed for %s: %s", idx, routing_key, e)
                validation_reasons.append(f"provider_error:{str(e)[:50]}")
                continue

            last_response = response
            last_response_model = routing_key
            last_provider = provider
            last_model_config = model_config

            is_valid, reason, judge_result = await self.evaluate_response(
                response,
                request,
                validator,
                local_judge,
                webhook_validator,
            )
            last_judge_result = judge_result
            if rsl_trace:
                await rsl_trace.feedback(rsl_trace.last_generation_attempt_id, is_valid, reason, judge_result)

            validation_reasons.append(f"{routing_key}:{reason}")

            if is_valid:
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                cascade_meta = CascadeMetadata(
                    enabled=True,
                    level=idx,
                    escalated_from=original_model if idx > 0 else None,
                    final_model=routing_key,
                    validation_passed=True,
                    escalation_reason=validation_reasons[0] if idx > 0 else None,
                    models_attempted=models_attempted,
                    judge_model=local_judge.judge_model if local_judge else None,
                    judge_score=judge_result.score if judge_result else None,
                    judge_reason=judge_result.reason if judge_result else None,
                )
                response.cascade = cascade_meta
                return response, cascade_meta

            if idx < len(chain) - 1:
                logger.info(
                    "Cascade escalation: %s -> %s (reason: %s)",
                    routing_key, chain[idx + 1][0], reason,
                )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        final_model = last_response_model or (models_attempted[-1] if models_attempted else original_model)
        cascade_meta = CascadeMetadata(
            enabled=True,
            level=len(models_attempted) - 1,
            escalated_from=original_model if len(models_attempted) > 1 else None,
            final_model=final_model,
            validation_passed=False,
            escalation_reason="; ".join(validation_reasons),
            models_attempted=models_attempted,
            judge_model=local_judge.judge_model if local_judge else None,
            judge_score=last_judge_result.score if last_judge_result else None,
            judge_reason=last_judge_result.reason if last_judge_result else None,
        )
        if last_response:
            last_response.cascade = cascade_meta
        return last_response, cascade_meta

    async def evaluate_response(
        self,
        response: ChatResponse,
        request: ChatRequest,
        validator: ResponseValidator,
        local_judge: LocalJudge | None,
        webhook_validator: WebhookValidator | None,
    ) -> tuple[bool, str, LocalJudgeResult | None]:
        is_valid, reason = validator.validate(response)
        judge_result = None
        if is_valid and local_judge:
            judge_result = await local_judge.evaluate(response, request)
            if judge_result.valid and judge_result.score >= local_judge.threshold:
                is_valid = True
                reason = f"judge_pass:{judge_result.score:.2f}:{judge_result.reason}"
            else:
                is_valid = False
                reason = f"judge_rejected:{judge_result.score:.2f}:{judge_result.reason}"
        if is_valid and webhook_validator:
            is_valid, reason = await webhook_validator.validate(response, request)
        return is_valid, reason, judge_result

    async def log_cascade(
        self,
        original_model: str,
        cascade_meta: CascadeMetadata,
        total_cost: float | None,
        total_latency_ms: int,
        request_context: RequestContext | None = None,
        status: str = "success",
    ):
        request_context = request_context or RequestContext(subject_type="internal", subject_id="system")
        stat = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **request_context.log_fields(),
            "original_model": original_model,
            "models_attempted": json.dumps(cascade_meta.models_attempted or []),
            "final_model": cascade_meta.final_model or original_model,
            "total_escalations": cascade_meta.level,
            "validation_reasons": cascade_meta.escalation_reason,
            "total_cost": total_cost,
            "total_latency_ms": total_latency_ms,
            "status": status,
        }
        await self.db.insert_cascade_stat(stat)
