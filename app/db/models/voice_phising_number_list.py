import uuid

from sqlalchemy import Column, String, Text, TIMESTAMP, Integer, func
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base

# 보이스피싱 명단 테이블임 (사용자가 제보하기 눌렀으면 여기에 insert)
# get 할때 보이스피싱 명단 업데이트해서 보이스피싱 판별
class VoicePhisingNumberList(Base):
    __tablename__ = "voice_phising_number_list"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    number = Column(String(20), nullable=False, index=True)
    description = Column(Text)
    report_count = Column(Integer, nullable=False, server_default="1")  
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now()) 
