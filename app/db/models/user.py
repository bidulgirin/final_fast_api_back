# app/models/user.py
import uuid
from sqlalchemy import String, DateTime, func, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # 구글 계정 고유 식별자(sub) - 일반 회원가입을 위해 nullable로 변경
    google_sub: Mapped[str | None] = mapped_column(String(255), unique=True, index=True, nullable=True)

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    # 비밀번호 해시 (일반 회원 가입용)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # 이메일 검증 여부
    is_verified: Mapped[bool] = mapped_column(Boolean(), server_default="false", nullable=False)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    picture: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    nickname: Mapped[str | None] = mapped_column(String(50), nullable=True)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
