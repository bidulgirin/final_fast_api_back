from fastapi import APIRouter
from app.api.v1.endpoints import real_time_check, stt, voice_phising_number
from app.routers.chat import router as chat_router

# 전체 API에 /api prefix 부여
router = APIRouter()

# 기존 라우터들
router.include_router(stt.router, prefix="/stt", tags=["stt"])
router.include_router(real_time_check.router, prefix="/mfcc", tags=["mfcc"])
router.include_router(voice_phising_number.router, prefix="/voice_phising_number_list", tags=["voice_phising_number_list"])

# chat_router가 내부에서 prefix="/v1/chat"를 가지고 있다면
# 최종 경로는 /api + /v1/chat + /send = /api/v1/chat/send 가 됨
router.include_router(chat_router)
