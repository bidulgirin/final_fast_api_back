from fastapi import APIRouter
from app.api.v1.endpoints import stt, mfcc

router = APIRouter()
# router.include_router(classify.router, prefix="/classify", tags=["vision"])
router.include_router(stt.router, prefix="/stt", tags=["stt"])
router.include_router(mfcc.router, prefix="/mfcc", tags=["mfcc"])