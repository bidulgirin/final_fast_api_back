from __future__ import annotations

from sqlalchemy import String, Text, Integer, DateTime, func, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base

class PhisingCaseDocs(Base):
    __tablename__ = "phising_case_docs"
    __table_args__ = (
        UniqueConstraint("file_id", "interval", "case_name", name="uq_docs_group"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    file_id: Mapped[int] = mapped_column(Integer, index=True)
    interval: Mapped[int] = mapped_column(Integer, index=True)
    case_name: Mapped[str] = mapped_column(String(255), index=True)

    text: Mapped[str] = mapped_column(Text)

    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
