from fastapi import FastAPI
from app.api.v1.router import router as v1_router
from app.routers.test_db import router as test_db

app = FastAPI()
app.include_router(v1_router, prefix="/api/v1")
app.include_router(test_db, prefix="") # 테스트 등록

@app.get("/health")
def health():
    return {"ok": True}