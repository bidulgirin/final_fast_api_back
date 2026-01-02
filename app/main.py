from fastapi import FastAPI
from app.api.v1.router import router as v1_router
from app.api.v1.endpoints.stt import router as stt_router

app = FastAPI()
app.include_router(v1_router, prefix="/api/v1")

@app.get("/health")
def health():
    return {"ok": True}