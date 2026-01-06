from sqlalchemy.orm import Session
from app.db.models.chat import Conversation, Message
from sqlalchemy import select

def get_or_create_conversation(db: Session, conversation_id: str | None) -> Conversation:
    if conversation_id:
        conv = db.get(Conversation, conversation_id)
        if conv:
            return conv

    conv = Conversation()
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv

def add_message(db: Session, conversation: Conversation, role: str, content: str) -> Message:
    msg = Message(conversation_id=conversation.id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg

def get_recent_messages(db: Session, conversation_id: str, limit: int = 30) -> list[Message]:
    conv = db.get(Conversation, conversation_id)
    if not conv:
        return []
    # relationship order_by로 정렬되어 있지만, 안전하게 슬라이싱
    return conv.messages[-limit:]

# 대화 조회 (messages 전체 또는 limit만)
def get_conversation(db: Session, conversation_id: str) -> Conversation | None:
    return db.get(Conversation, conversation_id)

def get_messages_by_conversation(db: Session, conversation_id: str, limit: int = 200) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())
