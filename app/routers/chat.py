from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import logging
import traceback

from app.db.session import get_db
from app.schemas.chat import (
    SendChatRequest, SendChatResponse,
    LogMessageRequest, LogMessageResponse,
    ConversationOut, MessageOut
)
from app.crud.chat import get_or_create_conversation, add_message, get_recent_messages
from app.services.openai_service import ask_openai
from app.crud.chat import get_conversation, get_messages_by_conversation

router = APIRouter(prefix="/chat", tags=["chat"])
# logger = logging.getLogger(__name__)

@router.post("/log", response_model=LogMessageResponse)
def log_message(payload: LogMessageRequest, db: Session = Depends(get_db)):
    conv = get_or_create_conversation(db, payload.conversation_id)
    add_message(db, conv, payload.role, payload.content)
    return LogMessageResponse(conversation_id=conv.id)

@router.post("/send", response_model=SendChatResponse)
def send(payload: SendChatRequest, db: Session = Depends(get_db)):
    conv = get_or_create_conversation(db, payload.conversation_id)

    # 1) 유저 메시지 저장
    add_message(db, conv, "user", payload.user_text)

    # 2) 최근 히스토리 로드 -> OpenAI 입력 구성
    recent = get_recent_messages(db, conv.id, limit=30)
    history_for_openai = [{"role": m.role, "content": m.content} for m in recent[:-1]]  # 마지막은 방금 user_text

    # 3) OpenAI 호출
    try:
        assistant_text = ask_openai(history_for_openai, payload)  # payload 통째로
        if not assistant_text:
            assistant_text = "답변 생성에 실패했습니다. 잠시 후 다시 시도해주세요."
    except Exception as e:
        # 무조건 콘솔에 남기기
        # logger.exception("OpenAI error in /chat/send")
        # traceback.print_exc()

        # 클라이언트에도 타입/메시지 내려서 Android에서 확인 가능하게
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )

    add_message(db, conv, "assistant", assistant_text)
    return SendChatResponse(conversation_id=conv.id, assistant_text=assistant_text)

@router.get("/{conversation_id}", response_model=ConversationOut)
def get_chat_history(
    conversation_id: str,
    limit: int = Query(200, ge=1, le=1000),  # 너무 커지면 앱/서버 부담
    db: Session = Depends(get_db),
):
    conv = get_conversation(db, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = get_messages_by_conversation(db, conversation_id, limit=limit)

    return ConversationOut(
        conversation_id=conv.id,
        created_at=conv.created_at,
        messages=[MessageOut.model_validate(m) for m in messages],
    )
