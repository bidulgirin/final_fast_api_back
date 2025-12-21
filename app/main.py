from fastapi import FastAPI
from app.routers import test_db

app = FastAPI()
app.include_router(test_db.router)