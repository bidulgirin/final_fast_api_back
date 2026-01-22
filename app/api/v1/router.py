from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.endpoints.admin_faiss import router as admin_faiss_router
from app.api.v1.endpoints.chat.chat_faiss import router as chat_faiss_router
from app.api.v1.endpoints.chat.chat_guide import router as chat_guide_router
from app.api.v1.endpoints.faiss_keywords import router as faiss_keywords_router
from app.api.v1.endpoints.safe_faiss_keywords import router as safe_faiss_keywords_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.phising_docs import router as phising_docs_router
from app.api.v1.endpoints.real_time_check import router as real_time_check_router
from app.api.v1.endpoints.stt import router as stt_router
from app.api.v1.endpoints.voice_phising_number import router as voice_phising_number_router
from app.api.v1.users import router as users_router
from app.routers.chat import router as chat_router

router = APIRouter()

# public
router.include_router(chat_router)
router.include_router(chat_faiss_router)
router.include_router(chat_guide_router)
router.include_router(faiss_keywords_router)
router.include_router(safe_faiss_keywords_router)
router.include_router(voice_phising_number_router)
router.include_router(real_time_check_router)
router.include_router(stt_router)

# docs/admin/health
router.include_router(phising_docs_router)
router.include_router(admin_faiss_router)
router.include_router(health_router)

# auth/users
router.include_router(auth_router, prefix="/auth", tags=["auth"])
router.include_router(users_router, prefix="/users", tags=["users"])
