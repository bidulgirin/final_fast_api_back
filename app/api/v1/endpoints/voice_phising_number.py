from app.db.models.voice_phising_number_list import VoicePhisingNumberList
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel
import uuid
from app.db.session import get_db
from datetime import datetime

router = APIRouter()

# 스키마 :::: API 입출력 데이터 계약
class VoicePhisingCreate(BaseModel): # 
    number: str
    description: str | None = None

class VoicePhisingOut(BaseModel):
    id: uuid.UUID
    number: str
    description: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True




@router.post("", response_model=VoicePhisingOut, status_code=201)
def insert_number(payload: VoicePhisingCreate, db: Session = Depends(get_db)):
    row = VoicePhisingNumberList(number=payload.number, description=payload.description)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="이미 등록된 번호입니다.")
    db.refresh(row)
    return row