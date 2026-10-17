import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from contextlib import aclosing

from fastapi import APIRouter, Request, BackgroundTasks, Response
from fastapi.responses import StreamingResponse

from app.core.auth import resolve_request_context
from app.core.ensemble import ensemble_enabled, execute_ensemble
from app.core.exceptions import (
    ModelRouterError,
    ModelAccessError,
    ProviderError,
)
from app.core.routing_policy import DEFAULT_MODEL_ALIASES, load_routing_policy
from app.core.request_context import RequestContext
from app.core.routing_settings import load_effective_routing_settings
from app.schemas.chat import (
    ChatRequest,
    Usage,
    ChatResponse,
    CascadeMetadata,
    ChatChoice,
    ChatMessage,
    RoutingRetryMetadata,
)
from app.core.logging import log_request
from app.rsl.runtime import prepare_runtime

router = APIRouter()


def _calc_cost(pricing, pt: int, ct: int) -> float:
    if not pricing:
        return 0.0
    return (pt / 1_000_000) * pricing.prompt + (ct / 1_000_000) * pricing.completion


def _model_unavailable_retry_limit() -> int:
    raw = __import__("os").environ.get("ROUTER_MODEL_UNAVAILABLE_RETRIES", "3").strip()
    try:
        return max(0, min(10, int(raw)))
    except ValueError:
        return 3


async def _write_log(
    db,
    body,
    provider_name,
    model_config,
    request_context: RequestContext,
    elapsed,
    usage=None,
    error=None,
    cost_override: float | None = None,
    model_override: str | None = None,
):
    pt = usage.prompt_tokens if usage else 0
    ct = usage.completion_tokens if usage else 0
    cost = cost_override if cost_override is not None else (
        _calc_cost(model_config.pricing, pt, ct) if model_config and model_config.pricing else 0.0
    )
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **request_context.log_fields(),
        "model": model_override or body.model,
        "provider": provider_name,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "cost": round(cost, 6),
        "latency_ms": elapsed,
        "status": "error" if error else "success",
        "error": error,
    }
    await db.insert_usage(row)
    asyncio.create_task(asyncio.to_thread(log_request, row))


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    response: Response = None,
):
    request_context = await resolve_request_context(request)
    db = request.app.state.db
    routing_policy = await load_routing_policy(db, request_context.routing_profile_id)
    routing_settings = await load_effective_routing_settings(
        db,
        routing_profile_id=request_context.routing_profile_id,
        keyword_default=getattr(request.app.state, "keyword_routing_enabled", False),
        semantic_default=getattr(request.app.state, "semantic_routing_enabled", False),
        cascade_default=getattr(request.app.state, "cascade_enabled", False),
    )
    router_engine = request.app.state.router
    cascade_engine = request.app.state.cascade
    original_model = body.model
    requested_target_model = original_model
    intent_category = None
    ensemble_requested = ensemble_enabled(body)
    available_models = set(router_engine.model_map.keys())
    allowed_models = routing_policy.effective_allowed_models(
        request_context.allowed_models,
        available_models=available_models,
    )
    recommendation_applied = False
    recommendation = None
    recommender = getattr(request.app.state, "recommender", None)
    rsl_trace = None
    if not ensemble_requested:
        recommender, rsl_trace = await prepare_runtime(
            getattr(request.app.state, "rsl_store", None), recommender, routing_policy,
            request_context.routing_profile_id, body.messages, request_context.request_id,
        )

    async def start_rsl(model, candidates):
        if rsl_trace is not None:
            await rsl_trace.start(model, candidates)
            if response is not None:
                response.headers.update(rsl_trace.headers)

    if not ensemble_requested:
        requested_is_auto = original_model in DEFAULT_MODEL_ALIASES
        if not requested_is_auto:
            requested_target_model = router_engine.resolve_model_target(
                original_model,
                default_model=routing_policy.default_model,
                preferred_providers=routing_policy.preferred_providers,
                cost_bias=routing_policy.cost_bias_value,
                latency_bias=routing_policy.latency_bias_value,
                quality_bias=routing_policy.quality_bias_value,
            )

        recommendation_requested = requested_is_auto or routing_policy.recommendation_apply_to_explicit
        if routing_policy.recommendation_enabled and recommendation_requested and recommender:
            recommendation = recommender.recommend(
                messages=body.messages,
                router_engine=router_engine,
                policy=routing_policy,
                allowed_models=allowed_models,
            )
            if recommendation.applied and recommendation.selected_model:
                body.model = recommendation.selected_model
                recommendation_applied = True
            else:
                if requested_is_auto:
                    requested_target_model = router_engine.resolve_model_target(
                        original_model,
                        default_model=routing_policy.default_model,
                        preferred_providers=routing_policy.preferred_providers,
                        cost_bias=routing_policy.cost_bias_value,
                        latency_bias=routing_policy.latency_bias_value,
                        quality_bias=routing_policy.quality_bias_value,
                    )
                body.model = requested_target_model
        else:
            if requested_is_auto:
                requested_target_model = router_engine.resolve_model_target(
                    original_model,
                    default_model=routing_policy.default_model,
                    preferred_providers=routing_policy.preferred_providers,
                    cost_bias=routing_policy.cost_bias_value,
                    latency_bias=routing_policy.latency_bias_value,
                    quality_bias=routing_policy.quality_bias_value,
                )
            body.model = requested_target_model

    if not ensemble_requested and not recommendation_applied:
        keyword_enabled = routing_settings["keyword_routing_enabled"]
        if keyword_enabled:
            matched_key = await router_engine.match_keyword(
                body.messages,
                routing_profile_id=request_context.routing_profile_id,
            )
            if matched_key and matched_key in router_engine.model_map:
                body.model = matched_key

        if body.model == original_model:
            semantic_enabled = routing_settings["semantic_routing_enabled"]
            if semantic_enabled:
                embedder = getattr(request.app.state, "embedder", None)
                result = await router_engine.match_semantic(body.messages, embedder)
                if result:
                    target_key, category = result
                    body.model = target_key
                    intent_category = category

    cascade_enabled_global = routing_settings["cascade_enabled"]
    cascade_config = None
    if not ensemble_requested:
        if body.cascade:
            cascade_config = body.cascade.model_dump()
        elif cascade_enabled_global:
            cascade_config = {"enabled": True}
        if routing_policy.cascade_override_chain:
            cascade_config = {
                **(cascade_config or {"enabled": True}),
                "chain": list(routing_policy.cascade_override_chain),
            }

    if not ensemble_requested and requested_target_model in router_engine.model_map and not routing_policy.is_model_allowed(
        requested_target_model,
        request_context.allowed_models,
    ):
        raise ModelAccessError(requested_target_model)
    if not ensemble_requested and not routing_policy.is_model_allowed(
        body.model,
        request_context.allowed_models,
    ):
        raise ModelAccessError(body.model)

    start = time.monotonic()
    elapsed = lambda: int((time.monotonic() - start) * 1000)

    if ensemble_requested:
        try:
            resp, total_cost, judge_model_config = await execute_ensemble(
                body,
                router_engine,
                allowed_models=allowed_models,
            )
            ms = elapsed()
            background_tasks.add_task(
                _write_log,
                db,
                body,
                "Ensemble",
                judge_model_config,
                request_context,
                ms,
                resp.usage,
                None,
                total_cost,
                original_model,
            )
            return resp
        except ModelRouterError:
            raise
        except Exception as e:
            ms = elapsed()
            background_tasks.add_task(
                _write_log,
                db,
                body,
                "Ensemble",
                None,
                request_context,
                ms,
                None,
                str(e),
                None,
                original_model,
            )
            raise

    if cascade_config and cascade_config.get("enabled") is not False:
        chain = await cascade_engine.resolve_chain(
            body.model,
            cascade_config,
            routing_profile_id=request_context.routing_profile_id,
        )
        if chain and allowed_models is not None:
            chain = [entry for entry in chain if entry[0] in allowed_models]
        if chain:
            validator = cascade_engine.get_validator(
                body.model,
                cascade_config,
                routing_profile_id=request_context.routing_profile_id,
            )
            local_judge = cascade_engine.get_local_judge(
                body.model,
                cascade_config,
                routing_profile_id=request_context.routing_profile_id,
            )
            if local_judge and allowed_models is not None and local_judge.judge_model not in allowed_models:
                raise ModelAccessError(local_judge.judge_model)
            if local_judge:
                local_judge.rsl_trace = rsl_trace
            webhook_validator = cascade_engine.get_webhook_validator(
                body.model,
                routing_profile_id=request_context.routing_profile_id,
            )
            max_escalations = cascade_engine.get_max_escalations(
                body.model,
                cascade_config,
                routing_profile_id=request_context.routing_profile_id,
            )
            await start_rsl(chain[0][0], [entry[0] for entry in chain])

            if body.stream:
                return StreamingResponse(
                    _stream_cascade(
                        body, chain, validator, local_judge, webhook_validator, max_escalations,
                        db, cascade_engine, request_context, elapsed, original_model, intent_category,
                        rsl_trace=rsl_trace,
                    ),
                    media_type="text/event-stream",
                    headers=rsl_trace.headers if rsl_trace else None,
                )

            try:
                resp, cascade_meta = await cascade_engine.execute_cascade(
                    body, original_model, chain, validator, local_judge, webhook_validator,
                    max_escalations, request_context.api_key_id,
                    rsl_trace=rsl_trace,
                )
                ms = elapsed()
                final_model_config = next(
                    (mc for rk, _, mc in chain if rk == cascade_meta.final_model),
                    chain[-1][2],
                )

                if resp and resp.usage:
                    background_tasks.add_task(
                        _write_log, db, body, "Cascade",
                        final_model_config, request_context, ms, resp.usage,
                    )

                background_tasks.add_task(
                    cascade_engine.log_cascade,
                    original_model, cascade_meta, rsl_trace.total_cost if rsl_trace else 0.0, ms, request_context,
                )

                if resp:
                    resp.actual_model = cascade_meta.final_model
                    resp.model = original_model
                    if intent_category:
                        resp.intent_category = intent_category
                return resp
            except ModelRouterError:
                raise
            except Exception as e:
                ms = elapsed()
                background_tasks.add_task(
                    _write_log, db, body, "Cascade",
                    chain[-1][2], request_context, ms, error=str(e),
                )
                raise

    await start_rsl(body.model, recommendation.eligible_models if recommendation else [body.model])

    if body.stream:
        provider, model_config = await router_engine.route(body.model)
        return StreamingResponse(
            _stream(
                body,
                provider,
                model_config,
                db,
                request_context,
                elapsed,
                original_model,
                intent_category,
                router_engine,
                rsl_trace=rsl_trace,
            ),
            media_type="text/event-stream",
            headers=rsl_trace.headers if rsl_trace else None,
        )

    retry_history: list[str] = []
    retry_limit = _model_unavailable_retry_limit()
    can_retry_model = (
        recommendation_applied
        and original_model in DEFAULT_MODEL_ALIASES
        and not (cascade_config and cascade_config.get("enabled") is not False)
        and recommender is not None
    )

    while True:
        provider, model_config = await router_engine.route(body.model)
        provider_label = getattr(provider, "provider_name", provider.__class__.__name__)
        attempted_model = body.model
        try:
            if rsl_trace is not None:
                resp = await rsl_trace.complete(provider, body, model_config, attempted_model)
            else:
                resp = await provider.chat_completions(body, model_config)
            ms = elapsed()
            usage = resp.usage
            if body.model != original_model:
                resp.actual_model = body.model
                resp.model = original_model
            if retry_history:
                resp.routing_retry = RoutingRetryMetadata(
                    retry_count=len(retry_history),
                    attempted_models=[*retry_history, body.model],
                    reason="model_temporarily_unavailable",
                )
            if intent_category:
                resp.intent_category = intent_category
            background_tasks.add_task(
                _write_log, db, body, provider_label,
                model_config, request_context, ms, usage,
            )
            return resp
        except ProviderError as e:
            quarantined = router_engine.mark_model_error(
                attempted_model,
                model_config.name,
                e,
            )
            if not quarantined or not can_retry_model or len(retry_history) >= retry_limit:
                raise

            retry_history.append(attempted_model)
            background_tasks.add_task(
                _write_log,
                db,
                body,
                provider_label,
                model_config,
                request_context,
                elapsed(),
                error=str(e),
                model_override=attempted_model,
            )
            next_recommendation = recommender.recommend(
                messages=body.messages,
                router_engine=router_engine,
                policy=routing_policy,
                allowed_models=allowed_models,
            )
            next_model = next_recommendation.selected_model if next_recommendation.applied else None
            if not next_model or next_model == attempted_model:
                raise
            body.model = next_model
        except ModelRouterError:
            raise
        except Exception as e:
            ms = elapsed()
            background_tasks.add_task(
                _write_log, db, body, provider_label,
                model_config, request_context, ms, error=str(e),
            )
            raise


async def _stream(
    body,
    provider,
    model_config,
    db,
    request_context,
    elapsed,
    original_model=None,
    intent_category=None,
    router_engine=None,
    rsl_trace=None,
):
    usage: Usage | None = None
    error = None

    try:
        chunks = (rsl_trace.stream(provider, body, model_config, body.model) if rsl_trace
                  else provider.chat_completions_stream(body, model_config))
        async with aclosing(chunks):
            async for chunk in chunks:
                if chunk.usage:
                    usage = chunk.usage
                if body.model != original_model and original_model is not None:
                    chunk.actual_model = body.model
                    chunk.model = original_model
                if intent_category:
                    chunk.intent_category = intent_category
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"
    except ProviderError as e:
        if router_engine is not None:
            router_engine.mark_model_error(body.model, model_config.name, e)
        error = str(e)
        yield f"data: {{\"error\":\"{error}\"}}\n\n"
        yield "data: [DONE]\n\n"
    except ModelRouterError as e:
        error = str(e)
        yield f"data: {{\"error\":\"{error}\"}}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        error = str(e)
        yield f"data: {{\"error\":\"{error}\"}}\n\n"
        yield "data: [DONE]\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        error = "cancelled"
        raise
    finally:
        ms = elapsed()
        await _write_log(
            db, body, getattr(provider, "provider_name", provider.__class__.__name__),
            model_config, request_context, ms, usage,
            error=error,
        )


async def _stream_cascade(
    body, chain, validator, local_judge, webhook_validator, max_escalations,
    db, cascade_engine, request_context, elapsed, original_model, intent_category,
    rsl_trace=None,
):
    models_attempted = []
    validation_reasons = []
    last_usage = None
    last_provider_name = None
    last_model_config = None
    last_judge_result = None
    start_time = time.monotonic()

    for idx, (routing_key, provider, model_config) in enumerate(chain):
        if idx >= max_escalations + 1:
            break

        models_attempted.append(routing_key)
        last_provider_name = getattr(provider, "provider_name", provider.__class__.__name__)
        last_model_config = model_config

        buffered_chunks = []
        usage = None
        error = None

        try:
            chunks = (rsl_trace.stream(provider, body, model_config, routing_key) if rsl_trace
                      else provider.chat_completions_stream(body, model_config))
            async with aclosing(chunks):
                async for chunk in chunks:
                    buffered_chunks.append(chunk)
                    if chunk.usage:
                        usage = chunk.usage
                        last_usage = chunk.usage
        except Exception as e:
            error = str(e)
            buffered_chunks = []
            validation_reasons.append(f"{routing_key}:provider_error:{str(e)[:50]}")
            continue

        if error:
            continue

        content = ""
        finish_reason = None
        for chunk in buffered_chunks:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    content += delta.content
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

        temp_resp = ChatResponse(
            id=buffered_chunks[0].id if buffered_chunks else str(uuid.uuid4()),
            created=buffered_chunks[0].created if buffered_chunks else int(time.time()),
            model=routing_key,
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )
        temp_req = ChatRequest(
            model=body.model,
            messages=body.messages,
        )
        is_valid, reason, judge_result = await cascade_engine.evaluate_response(
            temp_resp,
            temp_req,
            validator,
            local_judge,
            webhook_validator,
        )
        last_judge_result = judge_result
        if rsl_trace:
            await rsl_trace.feedback(rsl_trace.last_generation_attempt_id, is_valid, reason, judge_result)

        validation_reasons.append(f"{routing_key}:{reason}")

        if is_valid:
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

            for chunk in buffered_chunks:
                if routing_key != original_model and original_model is not None:
                    chunk.actual_model = routing_key
                    chunk.model = original_model
                if intent_category:
                    chunk.intent_category = intent_category
                chunk.cascade = cascade_meta
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"

            ms = elapsed()
            if usage:
                await _write_log(
                    db, body, last_provider_name,
                    last_model_config, request_context, ms, usage,
                )
            await cascade_engine.log_cascade(
                original_model, cascade_meta, rsl_trace.total_cost if rsl_trace else 0.0, ms, request_context,
            )
            return

        if idx < len(chain) - 1:
            logger = __import__("logging").getLogger("model_router.cascade")
            logger.info(
                "Cascade streaming escalation: %s -> %s (reason: %s)",
                routing_key, chain[idx + 1][0], reason,
            )

    if buffered_chunks:
        cascade_meta = CascadeMetadata(
            enabled=True,
            level=len(models_attempted) - 1,
            escalated_from=original_model if len(models_attempted) > 1 else None,
            final_model=models_attempted[-1] if models_attempted else original_model,
            validation_passed=False,
            escalation_reason="; ".join(validation_reasons),
            models_attempted=models_attempted,
            judge_model=local_judge.judge_model if local_judge else None,
            judge_score=last_judge_result.score if last_judge_result else None,
            judge_reason=last_judge_result.reason if last_judge_result else None,
        )
        for chunk in buffered_chunks:
            if routing_key != original_model and original_model is not None:
                chunk.actual_model = routing_key
                chunk.model = original_model
            if intent_category:
                chunk.intent_category = intent_category
            chunk.cascade = cascade_meta
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"
    else:
        error_msg = {"error": "All cascade models failed"}
        yield f"data: {json.dumps(error_msg)}\n\n"
        yield "data: [DONE]\n\n"

    ms = elapsed()
    if last_usage:
        await _write_log(
            db, body, last_provider_name,
            last_model_config, request_context, ms, last_usage,
        )
    cascade_meta = CascadeMetadata(
        enabled=True,
        level=len(models_attempted) - 1,
        escalated_from=original_model if len(models_attempted) > 1 else None,
        final_model=models_attempted[-1] if models_attempted else original_model,
        validation_passed=False,
        escalation_reason="; ".join(validation_reasons),
        models_attempted=models_attempted,
        judge_model=local_judge.judge_model if local_judge else None,
        judge_score=last_judge_result.score if last_judge_result else None,
        judge_reason=last_judge_result.reason if last_judge_result else None,
    )
    await cascade_engine.log_cascade(
        original_model, cascade_meta, rsl_trace.total_cost if rsl_trace else 0.0, ms, request_context, status="error",
    )
