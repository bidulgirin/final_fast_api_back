from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI

from app.api.v1.router import router as v1_router
from app.db.base import Base
from app.db.session import engine
import app.db.models  # noqa: F401

from app.faiss.faiss_store import FaissStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    if os.getenv("AUTO_CREATE_TABLES", "false").lower() == "true":
        Base.metadata.create_all(bind=engine)

    faiss_index_path = os.getenv("FAISS_INDEX_PATH", "./data/index.faiss")
    embed_model_name = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    data_dir = Path(faiss_index_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    # 전역 객체는 app.state에 보관 (import 사이드 이펙트 방지)
    app.state.faiss_store = FaissStore(index_path=faiss_index_path, model_name=embed_model_name)
    app.state.data_dir = data_dir
    app.state.faiss_index_path = faiss_index_path

    yield

    # shutdown (필요 시 리소스 정리)
    # 예: 파일 close, GPU 메모리 정리 등


app = FastAPI(lifespan=lifespan)

# 모든 API는 v1_router에서만 include 하도록 통일
app.include_router(v1_router, prefix="/api/v1")
