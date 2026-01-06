from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from datetime import datetime

class SendChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    user_text: str = Field(min_length=1)

    call_id: Optional[int] = None
    summary_text: Optional[str] = None
    call_text: Optional[str] = None
    

class SendChatResponse(BaseModel):
    conversation_id: str
    assistant_text: str

class LogMessageRequest(BaseModel):
    conversation_id: Optional[str] = None
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

class LogMessageResponse(BaseModel):
    conversation_id: str
    
# 조회 응답 스키마
class MessageOut(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime

    class Config:
        from_attributes = True  # SQLAlchemy ORM -> Pydantic 변환

class ConversationOut(BaseModel):
    conversation_id: str
    created_at: datetime
    messages: List[MessageOut]

    class Config:
        from_attributes = True
