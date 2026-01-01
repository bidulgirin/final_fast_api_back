# app/models/analysis_result.py
import uuid
from sqlalchemy import Column, String, TIMESTAMP, ForeignKey, Numeric
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    recording_id = Column(UUID(as_uuid=True), ForeignKey("recordings.id", ondelete="CASCADE"))

    phishing_risk_score = Column(Numeric(5,2))
    deepvoice_risk_score = Column(Numeric(5,2))
    final_decision = Column(String(20))

    threshold = Column(Numeric(5,2))
    model_version = Column(String(50))
    analyzed_at = Column(TIMESTAMP, server_default="now()")