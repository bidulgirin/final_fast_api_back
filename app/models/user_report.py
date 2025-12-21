# app/models/user_report.py
import uuid
from sqlalchemy import Column, String, Text, TIMESTAMP, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base

class UserReport(Base):
    __tablename__ = "user_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    analysis_result_id = Column(UUID(as_uuid=True), ForeignKey("analysis_results.id", ondelete="CASCADE"))

    report_type = Column(String(30))
    summary = Column(Text)
    action_guide = Column(Text)
    created_at = Column(TIMESTAMP, server_default="now()")