# app/models/chunk_analysis.py
import uuid
from sqlalchemy import Column, Text, TIMESTAMP, ForeignKey, Numeric
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base

class ChunkAnalysis(Base):
    __tablename__ = "chunk_analysis"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    recording_id = Column(UUID(as_uuid=True), ForeignKey("recordings.id", ondelete="CASCADE"))
    analysis_result_id = Column(UUID(as_uuid=True), ForeignKey("analysis_results.id", ondelete="CASCADE"))

    stt_text = Column(Text, nullable=False)
    chunk_score = Column(Numeric(5,2))
    created_at = Column(TIMESTAMP, server_default="now()")