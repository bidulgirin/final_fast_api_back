# db 연결 예시 코드
from fastapi import Header, HTTPException
from jose import JWTError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from sqlalchemy.orm import Session
from app.core import jwt_utils
from app.core.config import settings

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    echo=False,   # SQL 로그 출력
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# DB 세션 Dependency
def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user_id(authorization: str = Header(..., alias="Authorization")) -> str:
    """
    Authorization: Bearer <token> 에서 JWT를 검증하고 user_id(sub)를 반환.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.replace("Bearer ", "").strip()
    try:
        payload = jwt_utils.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG])
        return payload["sub"]
    except (JWTError, KeyError):
        raise HTTPException(status_code=401, detail="Invalid access token")