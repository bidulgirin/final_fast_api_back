# db 연결 예시 코드
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from sqlalchemy.orm import Session

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    echo=True,   # SQL 로그 출력 (테스트용)
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