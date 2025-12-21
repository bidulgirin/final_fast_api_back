# app/models/post_incident_management.py
from sqlalchemy import Column, String, Integer, TIMESTAMP, Boolean
from app.db.base import Base

class PostIncidentManagement(Base):
    __tablename__ = "post_incident_management"

    phone_hash = Column(String, primary_key=True)
    report_count = Column(Integer, default=0)
    last_reported_at = Column(TIMESTAMP)
    is_banned = Column(Boolean, default=False)