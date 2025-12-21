# app/models/recording.py
import uuid
from sqlalchemy import Column, Integer, TIMESTAMP, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base

class Recording(Base):
    __tablename__ = "recordings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    duration_sec = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP, server_default="now()")