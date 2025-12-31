# app/models/report.py
import uuid
from sqlalchemy import Column, String, TIMESTAMP, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base

class Report(Base):
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))

    report_category = Column(String(50))
    reported_at = Column(TIMESTAMP, server_default="now()")
    is_blocked = Column(Boolean, default=False)