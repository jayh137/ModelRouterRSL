from fastapi import APIRouter, Request, Depends

from app.core.route_preview import build_route_preview, parse_allowed_models
from app.core.routing_policy import load_routing_policy
from app.core.routing_settings import load_effective_routing_settings
from app.schemas.admin import RoutePreviewRequest
from app.rsl.runtime import prepare_runtime
from ._auth import require_admin

router = APIRouter(tags=["admin / inspect"])


@router.post("/route/preview")
async def preview_route(body: RoutePreviewRequest, request: Request, _=Depends(require_admin)):
    allowed_models = parse_allowed_models(body.allowed_models)
    routing_profile_id = body.routing_profile_id
    if body.user_id and not routing_profile_id:
        profile = await request.app.state.db.get_routing_profile_for_user(body.user_id)
        if profile:
            routing_profile_id = str(profile["id"])
    if body.api_key:
        key = await request.app.state.db.get_api_key(body.api_key)
        if key:
            allowed_models = parse_allowed_models(key["allowed_models"])
            if not routing_profile_id:
                profile = await request.app.state.db.get_routing_profile_for_api_key(key["id"])
                if profile:
                    routing_profile_id = str(profile["id"])

    routing_settings = await load_effective_routing_settings(
        request.app.state.db,
        routing_profile_id=routing_profile_id,
        keyword_default=getattr(request.app.state, "keyword_routing_enabled", False),
        semantic_default=getattr(request.app.state, "semantic_routing_enabled", False),
        cascade_default=getattr(request.app.state, "cascade_enabled", False),
    )
    routing_policy = await load_routing_policy(
        request.app.state.db,
        routing_profile_id,
    )
    allowed_models = routing_policy.effective_allowed_models(
        allowed_models,
        available_models=set(request.app.state.router.model_map.keys()),
    )

    recommender, trace = await prepare_runtime(
        getattr(request.app.state, "rsl_store", None),
        getattr(request.app.state, "recommender", None),
        routing_policy, routing_profile_id, body.messages,
    )
    result = await build_route_preview(
        body,
        request.app.state.router,
        request.app.state.cascade,
        routing_policy=routing_policy,
        routing_profile_id=routing_profile_id,
        keyword_routing_enabled=routing_settings["keyword_routing_enabled"],
        semantic_routing_enabled=routing_settings["semantic_routing_enabled"],
        embedder=getattr(request.app.state, "embedder", None),
        recommendation_enabled=routing_policy.recommendation_enabled,
        recommender=recommender,
        cascade_enabled=routing_settings["cascade_enabled"],
        allowed_models=allowed_models,
    )
    result["rsl"] = {
        "enabled": routing_policy.rsl_enabled,
        "mode": routing_policy.rsl_mode,
        "policy_version": trace.version_id if trace else None,
    }
    return result
