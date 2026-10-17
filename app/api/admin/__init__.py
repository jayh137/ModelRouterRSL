from fastapi import APIRouter
from . import providers, models, keys, usage, status, keyword, cascade, inspect, routing_profiles, rsl

router = APIRouter(prefix="/admin")
router.include_router(providers.router)
router.include_router(models.router)
router.include_router(keys.router)
router.include_router(usage.router)
router.include_router(status.router)
router.include_router(keyword.router)
router.include_router(cascade.router)
router.include_router(inspect.router)
router.include_router(routing_profiles.router)
router.include_router(rsl.router)
