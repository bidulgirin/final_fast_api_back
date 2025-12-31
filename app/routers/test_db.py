# app/routers/test_db.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.db.session import get_db

router = APIRouter(prefix="/test", tags=["test"])

@router.get("/db")
def test_db(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"result": "PostgreSQL connected"}