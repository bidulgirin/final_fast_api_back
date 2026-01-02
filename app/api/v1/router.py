from fastapi import APIRouter
from app.api.v1.endpoints import stt, classify

router = APIRouter()
router.include_router(stt.router, prefix="/stt", tags=["stt"])
# router.include_router(classify.router, prefix="/classify", tags=["vision"])