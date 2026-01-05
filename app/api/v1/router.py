from fastapi import APIRouter
from app.api.v1.endpoints import stt, mfcc, voice_phising_number

router = APIRouter()
router.include_router(stt.router, prefix="/stt", tags=["stt"])
router.include_router(mfcc.router, prefix="/mfcc", tags=["mfcc"])
router.include_router(voice_phising_number.router, prefix="/voice_phising_number", tags=["voice_phising_number"])