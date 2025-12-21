# app/models/user.py
import uuid
from sqlalchemy import Column, String, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    phone_hash = Column(String, nullable=False)
    platform = Column(String(20), nullable=False)
    device_model = Column(String(50))
    app_version = Column(String(20))
    created_at = Column(TIMESTAMP, server_default="now()")