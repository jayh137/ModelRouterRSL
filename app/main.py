import logging
import asyncio
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import load_config
from app.core.db import Database
from app.core.router import RouterEngine
from app.core.embedder import SemanticEmbedder
from app.core.cascade import CascadeEngine
from app.core.recommender import ModelRecommender
from app.rsl.store import RSLStore
from app.core.model_discovery import sync_all_providers_from_remote
from app.core.rate_limit import FixedWindowRateLimiter
from app.core.exceptions import (
    InvalidRequestError,
    ModelNotFoundError,
    AuthError,
    ProviderError,
    AllKeysFailedError,
    ModelAccessError,
    RateLimitError,
)
from app.api import chat, models
from app.api.admin import router as admin_router

VALID_RUNTIME_MODES = {"internal", "standalone"}


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", ""}


def _runtime_mode() -> str:
    return os.environ.get("ROUTER_RUNTIME_MODE", "standalone").strip().lower() or "standalone"


def validate_runtime_settings(
    *,
    runtime_mode: str,
    enable_admin_api: bool,
    enable_standalone_keys: bool,
    internal_auth_key: str,
    admin_key: str,
):
    if runtime_mode not in VALID_RUNTIME_MODES:
        raise RuntimeError(
            f"Invalid ROUTER_RUNTIME_MODE '{runtime_mode}'. Expected one of: {sorted(VALID_RUNTIME_MODES)}"
        )

    if runtime_mode == "internal" and enable_standalone_keys:
        raise RuntimeError(
            "Unsafe router configuration: internal mode requires ROUTER_ENABLE_STANDALONE_KEYS=false."
        )

    if runtime_mode == "internal" and not internal_auth_key.strip():
        raise RuntimeError(
            "Unsafe router configuration: internal mode requires ROUTER_INTERNAL_API_KEY."
        )

    if enable_admin_api and not admin_key.strip():
        raise RuntimeError(
            "Unsafe router configuration: Admin API requires ADMIN_API_KEY."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    runtime_mode = _runtime_mode()
    internal_mode = runtime_mode == "internal"
    enable_admin_api = _env_flag("ROUTER_ENABLE_ADMIN_API", not internal_mode)
    enable_standalone_keys = _env_flag("ROUTER_ENABLE_STANDALONE_KEYS", not internal_mode)
    enable_usage_api = _env_flag("ROUTER_ENABLE_USAGE_API", not internal_mode)
    auth_key = config.auth.get("api_key", "")
    internal_auth_key = os.environ.get("ROUTER_INTERNAL_API_KEY", auth_key).strip()
    admin_key = os.environ.get("ADMIN_API_KEY", "").strip()

    validate_runtime_settings(
        runtime_mode=runtime_mode,
        enable_admin_api=enable_admin_api,
        enable_standalone_keys=enable_standalone_keys,
        internal_auth_key=internal_auth_key,
        admin_key=admin_key,
    )

    db_path = os.environ.get("ROUTER_DB", "router.db")
    db = Database(db_path)
    await db.connect()
    await db.migrate()

    count = await db._fetchone("SELECT COUNT(*) FROM providers")
    if count and count[0] == 0:
        await db.seed_from_yaml("config/providers.yaml")

    auto_sync_models = os.environ.get("AUTO_SYNC_PROVIDER_MODELS", "false").lower() not in {
        "0",
        "false",
        "no",
    }
    if auto_sync_models:
        sync_timeout = int(os.environ.get("AUTO_SYNC_PROVIDER_MODELS_TIMEOUT", "10"))
        await sync_all_providers_from_remote(db, timeout=sync_timeout)

    router = RouterEngine(db)
    await router.reload()

    cascade = CascadeEngine(db, router)
    await cascade.reload()
    recommender = ModelRecommender()

    app.state.db = db
    app.state.router = router
    app.state.cascade = cascade
    app.state.recommender = recommender
    app.state.rsl_store = (
        await asyncio.to_thread(RSLStore, os.environ.get("ROUTER_RSL_DB", "rsl.db"))
        if _env_flag("ROUTER_RSL_ENABLED", False) else None
    )
    app.state.runtime_mode = runtime_mode
    app.state.enable_admin_api = enable_admin_api
    app.state.enable_standalone_keys = enable_standalone_keys
    app.state.enable_usage_api = enable_usage_api
    app.state.auth_key = auth_key
    app.state.internal_auth_key = internal_auth_key
    app.state.admin_key = admin_key
    app.state.api_key_limiter = FixedWindowRateLimiter()
    app.state.keyword_routing_enabled = (await db.get_setting("keyword_routing_enabled", "false")) == "true"
    app.state.cascade_enabled = (await db.get_setting("cascade_enabled", "false")) == "true"

    semantic_cfg = getattr(config, "semantic_routing", None) or {}
    sem_enabled_db = (await db.get_setting("semantic_routing_enabled", "false")) == "true"
    sem_enabled = sem_enabled_db or semantic_cfg.get("enabled", False)
    embedder = SemanticEmbedder(
        model_name=semantic_cfg.get("model", "all-MiniLM-L6-v2"),
        threshold=semantic_cfg.get("threshold", 0.4),
        cache_dir=semantic_cfg.get("cache_dir", ".embed_cache"),
        categories=semantic_cfg.get("categories"),
    )
    if sem_enabled:
        embedder.load()
    app.state.embedder = embedder
    app.state.semantic_routing_enabled = embedder.enabled and sem_enabled

    if not router.model_map:
        print("WARNING: No models loaded. Use Admin API or add config to config/providers.yaml")
        print("  Admin API: POST /admin/providers, POST /admin/models, etc.")
        print()

    yield

    await db.close()


app = FastAPI(title="Model Router", version="0.2.0", lifespan=lifespan)

app.include_router(chat.router)
app.include_router(models.router)
if _env_flag("ROUTER_ENABLE_ADMIN_API", _runtime_mode() != "internal"):
    app.include_router(admin_router)


@app.exception_handler(ModelNotFoundError)
async def handle_model_not_found(_, exc):
    return JSONResponse(status_code=404, content={"error": str(exc), "available": exc.available})


@app.exception_handler(InvalidRequestError)
async def handle_invalid_request(_, exc):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(AuthError)
async def handle_auth_error(_, exc):
    return JSONResponse(status_code=401, content={"error": str(exc)})


@app.exception_handler(ModelAccessError)
async def handle_model_access_error(_, exc):
    return JSONResponse(status_code=403, content={"error": str(exc)})


@app.exception_handler(RateLimitError)
async def handle_rate_limit_error(_, exc):
    return JSONResponse(status_code=429, content={"error": str(exc)})


@app.exception_handler(ProviderError)
async def handle_provider_error(_, exc):
    return JSONResponse(status_code=502, content={"error": str(exc)})


@app.exception_handler(AllKeysFailedError)
async def handle_all_keys_failed(_, exc):
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(Exception)
async def handle_unexpected_error(_, exc):
    return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint for monitoring."""
    try:
        # Check database connectivity
        db = request.app.state.db
        await db._fetchone("SELECT 1")
        
        # Check router engine
        router = request.app.state.router
        model_count = len(router.model_map)
        provider_count = len(router.providers)
        
        return {
            "status": "healthy",
            "version": "0.2.0",
            "runtime_mode": getattr(request.app.state, "runtime_mode", "standalone"),
            "providers": provider_count,
            "models": model_count,
            "database": "connected",
            "features": {
                "keyword_routing": getattr(request.app.state, "keyword_routing_enabled", False),
                "semantic_routing": getattr(request.app.state, "semantic_routing_enabled", False),
                "recommendation": bool(getattr(request.app.state, "recommender", None)),
                "rsl": getattr(request.app.state, "rsl_store", None) is not None,
                "cascade": getattr(request.app.state, "cascade_enabled", False),
                "admin_api": getattr(request.app.state, "enable_admin_api", True),
                "standalone_keys": getattr(request.app.state, "enable_standalone_keys", True),
                "usage_api": getattr(request.app.state, "enable_usage_api", True),
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "database": "disconnected"
            }
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    import uvicorn
    config = load_config()
    uvicorn.run(
        "app.main:app",
        host=config.server.get("host", "0.0.0.0"),
        port=config.server.get("port", 8000),
        reload=False,
    )


if __name__ == "__main__":
    main()
