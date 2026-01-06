from fastapi import FastAPI

from app.api.v1.router import router as v1_router
from app.routers.test_db import router as test_db

# DB 관련 import (Base / engine)
from app.db.base import Base
from app.db.session import engine

# 모델들을 등록하기 위해 import (Base.metadata에 모델이 올라가도록)
import app.db.models  # noqa: F401

app = FastAPI()

# 여기서 /api/v1을 붙일 거면, v1_router 내부에는 /api나 /v1 prefix를 또 붙이지 않는 게 안전
app.include_router(v1_router, prefix="/api/v1")

# 테스트 라우터는 루트에 붙임 (prefix는 굳이 "" 안 줘도 됨)
app.include_router(test_db)

@app.on_event("startup")
def on_startup():
    # 개발 단계 편의용: 테이블 자동 생성
    Base.metadata.create_all(bind=engine)

@app.get("/health")
def health():
    return {"ok": True}
