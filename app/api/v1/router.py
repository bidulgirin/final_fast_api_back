from fastapi import APIRouter

from app.api.v1.endpoints.chat import chat_faiss, chat_guide
from app.api.v1.endpoints import faiss_keywords
from app.api.v1.endpoints.phising_docs import router as phising_docs_router
from app.api.v1.endpoints.admin_faiss import router as admin_faiss_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints import voice_phising_number
from app.api.v1.endpoints import real_time_check
from app.api.v1.endpoints import stt

router = APIRouter()

router.include_router(chat_faiss.router)
router.include_router(chat_guide.router)
router.include_router(faiss_keywords.router)
router.include_router(voice_phising_number.router)
router.include_router(real_time_check.router)
router.include_router(stt.router)

router.include_router(phising_docs_router)
router.include_router(admin_faiss_router)

router.include_router(health_router)
