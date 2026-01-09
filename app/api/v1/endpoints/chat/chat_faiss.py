# api/v1/endpoints/chat_faiss.py

from __future__ import annotations

import uuid
import json
import threading
from datetime import datetime, timezone
from typing import Any, Optional, List, Dict, Tuple

import numpy as np
import faiss

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body, Query
from pydantic import BaseModel, Field

from sqlalchemy import Column, Integer, String, Text, DateTime, select, ForeignKey
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import JSONB

import re

# ---------------------------------------------------------  유틸 

_RE_PHONE = re.compile(r"(01[016789])[-\s]?\d{3,4}[-\s]?\d{4}")
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_RE_RRN = re.compile(r"\b\d{6}[-\s]?\d{7}\b")  # 주민번호 형태
_RE_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")  # 카드번호(대략)
_RE_ACCOUNT = re.compile(r"\b\d{2,4}[-\s]?\d{2,4}[-\s]?\d{2,6}[-\s]?\d{1,4}\b")  # 계좌(대략)
_RE_URL = re.compile(r"(https?://\S+|www\.\S+)")

def mask_pii(text: str) -> str:
    t = text
    t = _RE_EMAIL.sub("[이메일]", t)
    t = _RE_PHONE.sub("[전화번호]", t)
    t = _RE_RRN.sub("[주민번호]", t)
    t = _RE_CARD.sub("[카드번호]", t)
    t = _RE_URL.sub("[URL]", t)

    # 계좌는 오탐이 있을 수 있어 “다른 것들 후”에 마스킹
    t = _RE_ACCOUNT.sub("[계좌번호]", t)
    return t


# ---------------------------------------------------------
from app.db.base import Base        # <- 프로젝트 구조에 맞게 수정
from app.db.session import engine, get_db   # <- 프로젝트 구조에 맞게 수정

from openai import OpenAI

from sqlalchemy.orm import relationship

from app.api.v1.endpoints.chat.chat_guide import chat_guide_store

# --- (chat_faiss.py 상단 import들 근처에 추가) ---
import os
import json
from typing import Any, Optional, List, Dict

from fastapi import HTTPException
from pydantic import BaseModel, Field

from openai import OpenAI


# --- OpenAI Client (앱 시작 시 1회 생성 권장) ---
_openai_client: Optional[OpenAI] = None


router = APIRouter(prefix="/chat-faiss", tags=["chat-faiss"])


# ---------- OpenAI client ----------
_openai_client: Optional[OpenAI] = None
def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다.")
        _openai_client = OpenAI(api_key=key)
    return _openai_client

# ---------- request/response ----------
class ChatReq(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(min_length=1)

    k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.4, ge=-1.0, le=1.0)  # cosine 기준이면 0~1 사이를 주로 씀
    # category는 당분간 안 쓴다 했지만, 서버는 선택적으로 받을 수 있게 둠(클라는 빼면 됨)
    category: Optional[str] = None

    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    model: Optional[str] = None

    # 통화 컨텍스트(클라에서 보내면 LLM 입력에 포함)
    call_id: Optional[int] = None
    summary_text: Optional[str] = None
    call_text: Optional[str] = None

class ChatCaseCard(BaseModel):
    id: int
    score: float
    category: str
    user_query: str
    answer: str
    metadata: Optional[Dict[str, Any]] = None

class GuideCard(BaseModel):
    id: int
    score: float
    key: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class ChatResp(BaseModel):
    session_id: str
    risk_level: str
    final_answer: str
    matched_cases: List[ChatCaseCard]
    follow_up_questions: List[str]
    disclaimer: str

class HistoryMessage(BaseModel):
    role: str
    content: str
    created_at: str


class HistoryResp(BaseModel):
    session_id: str
    messages: List[HistoryMessage]

def risk_level_from(top_score: float) -> str:
    if top_score >= 0.78:
        return "HIGH"
    if top_score >= 0.62:
        return "MEDIUM"
    return "LOW"

# ---------- Structured Output schema (Responses API text.format) ----------
CHAT_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "phishing_chatbot_response",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
            "final_answer": {"type": "string"},
            "follow_up_questions": {"type": "array", "items": {"type": "string"}},
            "disclaimer": {"type": "string"},
        },
        "required": ["risk_level", "final_answer", "follow_up_questions", "disclaimer"],
    },
    "strict": True,
}

def build_context(cases: List[ChatCaseCard]) -> str:
    payload = []
    for c in cases[:5]:
        payload.append({
            "id": c.id,
            "score": round(c.score, 4),
            "category": c.category,
            "user_query": c.user_query,
            "answer": c.answer,
            "metadata": c.metadata or {},
        })
    return json.dumps(payload, ensure_ascii=False)

# 가이드 빌더임
def build_guides_context(guides: List[GuideCard]) -> str:
    payload = []
    for g in guides[:5]:
        payload.append({
            "id": g.id,
            "score": round(g.score, 4),
            "key": g.key,
            "title": g.title,
            "content": g.content,
            "metadata": g.metadata or {},
        })
    return json.dumps(payload, ensure_ascii=False)

def ensure_session(db: Session, session_id: Optional[str]) -> ChatSession:
    if session_id:
        s = db.get(ChatSession, session_id)
        if s:
            return s
    s = ChatSession()
    db.add(s)
    db.commit()
    return s

def save_message(
    db: Session,
    session: ChatSession,
    role: str,
    content_raw: str,
    retrieval: Optional[Dict[str, Any]] = None,
    risk_level: Optional[str] = None,
    model: Optional[str] = None,
    call_context: Optional[Dict[str, Any]] = None,
):
    msg = ChatMessage(
        session_id=session.id,
        role=role,
        content_masked=mask_pii(content_raw),
        retrieval=retrieval,
        risk_level=risk_level,
        model=model,
        call_context=call_context,
    )
    db.add(msg)
    session.updated_at = datetime.now(timezone.utc)
    db.commit()

def ensure_session(db: Session, session_id: Optional[str]) -> ChatSession:
    if session_id:
        s = db.get(ChatSession, session_id)
        if s:
            return s
    s = ChatSession()
    db.add(s)
    db.commit()
    return s

def call_llm_strict_json(
    client: OpenAI,
    model: str,
    instructions: str,
    user_input: str,
    temperature: float,
) -> Dict[str, Any]:
    """
    1) json_schema 시도
    2) 실패 시 json_object로 폴백(유효 JSON만 보장)
    """
    try:
        resp = client.responses.create(
            model=model,
            instructions=instructions,
            input=user_input,
            temperature=temperature,
            store=False,
            text={"format": CHAT_JSON_SCHEMA},
        )
        return json.loads(resp.output_text)
    except Exception:
        resp = client.responses.create(
            model=model,
            instructions=instructions + "\n반드시 JSON object 1개만 출력해라.",
            input=user_input,
            temperature=temperature,
            store=False,
            text={"format": {"type": "json_object"}},
        )
        return json.loads(resp.output_text)

def clip(text: Optional[str], limit: int) -> str:
    if not text:
        return ""
    t = text.strip()
    return t[:limit] + ("…(생략)" if len(t) > limit else "")

def normalize_model(m: Optional[str]) -> Optional[str]:
    if not m:
        return None
    m = m.strip()
    if not m or m.lower() in ("string", "null", "none"):
        return None
    return m


@router.post("/chat", response_model=ChatResp)
def chat(req: ChatReq, db: Session = Depends(get_db)):
    # 0) 세션 확보
    session = ensure_session(db, req.session_id)

    # 1) 무조건 FAISS 검색부터 수행 (근거 강제의 시작)
    hits = chat_faiss_store.search(
        db=db,
        query=req.message,
        k=req.k,
        category=req.category, # 클라이언트는 이값 안보내도됨 근데 보내면 더 정확하게 검색함...
        min_score=req.min_score,
    )
    matched_cases = [
        ChatCaseCard(
            id=h.id,
            score=h.score,
            category=h.category,
            user_query=h.user_query,
            answer=h.answer,
            metadata=h.metadata,
        )
        for h in hits
    ]

     # 2) 가이드 검색(FAISS) - 통화요약/전사를 일부 섞어서 recall 강화
    guide_query = req.message
    if req.summary_text:
        guide_query += "\n" + clip(req.summary_text, 500)
    if req.call_text:
        guide_query += "\n" + clip(req.call_text, 1000)

    guide_hits = chat_guide_store.search(
        db=db,
        query=guide_query,
        k=min(req.k, 10),
        min_score=0.0,  # 필요하면 req.min_score랑 분리해도 됨
    )
    matched_guides = [
        GuideCard(
            id=g.id,
            score=g.score,
            key=g.key,
            title=g.title,
            content=g.content,
            metadata=g.metadata,
        )
        for g in guide_hits
    ]

    retrieval_log = {
        "cases": {
            "k": req.k,
            "min_score": req.min_score,
            "category": req.category,
            "top": [{"doc_id": c.id, "score": c.score, "category": c.category} for c in matched_cases[:5]],
        },
        "guides": {
            "k": min(req.k, 10),
            "min_score": 0.0,
            "top": [{"guide_id": g.id, "score": g.score, "key": g.key} for g in matched_guides[:5]],
        }
    }

    call_ctx = {
        "call_id": req.call_id,
        "summary_text": mask_pii(req.summary_text) if req.summary_text else None,
        "call_text": mask_pii(req.call_text) if req.call_text else None,
    }
    # 값이 전부 None이면 저장 의미가 없으니 None 처리
    if all(v is None for v in call_ctx.values()):
        call_ctx = None


    # 2) 유저 메시지 저장(마스킹)
    save_message(
        db=db,
        session=session,
        role="user",
        content_raw=req.message,
        retrieval=retrieval_log,
        risk_level=None,
        model=req.model,
        call_context=call_ctx,
    )

    # 3) 사례가 없으면: "없다" + 단정 금지 + 안전 안내 (LLM 호출 X)
    if not matched_cases:
        final_answer = (
            "현재 저장된 보이스피싱 사례 데이터에서 **유사한 패턴을 찾지 못했습니다.**\n"
            "그래도 실제 보이스피싱일 가능성을 배제할 수는 없으니, 아래를 확인해 주세요:\n"
            "- 비밀번호/인증번호/신분증/계좌정보 요구는 제공하지 않기\n"
            "- 링크 클릭/앱 설치 요청은 거절하기\n"
            "- 통화를 종료하고 해당 기관의 **공식 대표번호**로 직접 재확인하기"
        )
        disclaimer = "본 안내는 내부 사례 검색 결과 기반 참고용이며, 확정 판단이 아닙니다."

        save_message(
            db=db,
            session=session,
            role="assistant",
            content_raw=final_answer,
            retrieval=retrieval_log,
            risk_level="LOW",
            model=None,
            call_context=None,
        )

        return ChatResp(
            session_id=session.id,
            risk_level="LOW",
            final_answer=final_answer,
            matched_cases=[],
            follow_up_questions=[
                "상대가 어떤 기관/은행이라고 했나요?",
                "링크 클릭이나 앱 설치를 요구했나요?",
                "금전 이체/현금 전달을 요구했나요?",
            ],
            disclaimer=disclaimer,
        )

    # 4) 사례가 있으면: 여기서부터 LLM 생성 (근거 컨텍스트 포함)
    top_score = matched_cases[0].score
    risk_level = risk_level_from(top_score)

    # “근거 밖 단정 금지”를 아주 강하게
    system_instructions = (
        "너는 보이스피싱/스미싱 안전 안내 챗봇이다.\n"
        "사용자가 방금 보낸 [사용자 입력]이 '질문'이며, 반드시 그 질문에 직접 답해야 한다.\n"
        "[통화 요약/전사]는 참고 정보일 뿐이며 질문과 무관하면 무시한다.\n"
        "반드시 [유사사례 컨텍스트(JSON)]에 포함된 내용만 근거로 사용한다.\n"
        "컨텍스트에 없는 사실은 단정하지 말고 follow_up_questions로 확인 질문을 한다.\n"
        "사례가 없으면 '유사 사례를 찾지 못했다'고 명시하되, 가이드가 있으면 가이드를 근거로 안내할 수 있다.\n"
        "출력은 JSON 하나만 출력하고 스키마를 엄격히 준수한다."
    )

    extra_ctx = ""
    if req.summary_text:
        extra_ctx += f"\n[통화 요약]\n{req.summary_text}\n"
    if req.call_text:
        extra_ctx += f"\n[통화 전사/내용]\n{req.call_text}\n"

    user_input = (
        f"[사용자 입력]\n{req.message}\n"
        f"{extra_ctx}\n"
        f"[유사사례 컨텍스트(JSON)]\n{build_context(matched_cases)}\n\n"
        f"[가이드 컨텍스트(JSON)]\n{build_guides_context(matched_guides)}\n\n"
        f"[참고]\n- top_score: {top_score:.4f}\n- internal_risk_hint: {risk_level}\n"
    )

    client = get_openai_client()
    model = req.model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    try:
        resp = client.responses.create(
            model=model,
            instructions=system_instructions,
            input=user_input,
            temperature=req.temperature,
            store=False,
            text={"format": CHAT_JSON_SCHEMA},
        )
        out = json.loads(resp.output_text)
    except Exception as e:
        # LLM 실패 시에도 "근거 기반" 원칙 유지: 가장 유사 사례 answer를 그대로 사용(재작성 X)
        # fallback = (
        #     f"유사 사례가 발견되어 참고 정보를 제공합니다(단정 아님).\n\n"
        #     f"[가장 유사한 사례: {matched_cases[0].category}, score={matched_cases[0].score:.2f}]\n"
        #     f"{matched_cases[0].answer}\n\n"
        #     f"추가로, 비밀번호/인증번호/개인정보는 제공하지 말고 통화를 종료한 뒤 공식번호로 재확인하세요."
        # )
        parts = []
        if matched_cases:
            parts.append(f"[유사 사례: {matched_cases[0].category}, score={matched_cases[0].score:.2f}]\n{matched_cases[0].answer}")
        if matched_guides:
            g = matched_guides[0]
            parts.append(f"[추천 가이드: {g.title}]\n{g.content}")
        fallback = "유사 근거 기반으로 참고 안내를 제공합니다(단정 아님).\n\n" + "\n\n".join(parts)
        
        out = {
            "risk_level": risk_level,
            "final_answer": fallback,
            "follow_up_questions": [],
            "disclaimer": "본 안내는 내부 사례 검색 결과 기반 참고용이며, 확정 판단이 아닙니다.",
        }

    # 5) 어시스턴트 메시지 저장(마스킹)
    save_message(
        db=db,
        session=session,
        role="assistant",
        content_raw=out["final_answer"],
        retrieval=retrieval_log,
        risk_level=out.get("risk_level", risk_level),
        model=model,
        call_context=None,
    )

    return ChatResp(
        session_id=session.id,
        risk_level=out.get("risk_level", risk_level),
        # final_answer=out["final_answer"],
        final_answer=out.get("final_answer", ""),
        matched_cases=matched_cases[:req.k],
        follow_up_questions=out.get("follow_up_questions", []),
        disclaimer=out.get("disclaimer", "본 안내는 내부 사례 기반 참고용이며 확정 판단이 아닙니다."),
    )





# =========================================================
# 0) 세션/메시지 저장용 DB 모델
# =========================================================
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True, nullable=False)

    role = Column(String(16), nullable=False)  # "user" | "assistant"
    content_masked = Column(Text, nullable=False)

    # 어떤 검색 근거로 답했는지(운영/디버깅)
    retrieval = Column(JSONB, nullable=True)   # {"k":..,"min_score":..,"top":[...]}
    risk_level = Column(String(16), nullable=True)  # LOW|MEDIUM|HIGH
    model = Column(String(64), nullable=True)

    # (선택) 통화 컨텍스트도 저장하고 싶으면 여기에 넣어도 됨(마스킹 후)
    call_context = Column(JSONB, nullable=True)  # {"call_id":..,"summary_text":..,"call_text":..}

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    session = relationship("ChatSession", back_populates="messages")

# =========================================================
# 1) DB Model (문서 저장: id/category/user_query/answer/metadata)
# =========================================================
class ChatFaissDoc(Base):
    __tablename__ = "chat_faiss_docs"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(120), index=True, nullable=False)
    user_query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)

    # Postgres면 JSONB 권장, SQLite면 JSON으로 바꿔도 됨
    # 파이썬 속성명(meta) / DB 컬럼명(metadata)
    meta = Column("metadata", JSONB, nullable=True) #

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))


# =========================================================
# 2) Pydantic Schemas
# =========================================================
class ChatFaissDocIn(BaseModel):
    id: int
    category: str
    user_query: str
    answer: str
    metadata: Optional[Dict[str, Any]] = None


class IngestResp(BaseModel):
    inserted: int
    updated: int
    total_in_db: int
    indexed_count: int


class SearchReq(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    category: Optional[str] = None
    min_score: Optional[float] = None  # cosine similarity threshold (0~1 정도가 보통)


class SearchHit(BaseModel):
    id: int
    score: float
    category: str
    user_query: str
    answer: str
    metadata: Optional[Dict[str, Any]] = None


class SearchResp(BaseModel):
    results: List[SearchHit]


class StatsResp(BaseModel):
    total_in_db: int
    indexed_count: int
    dim: int
    metric: str
    last_built_at: Optional[str]


# =========================================================
# 3) Embedding (예시: sentence-transformers)
#    - 너가 이미 임베딩 파이프라인이 있으면 여기만 바꿔주면 됨
# =========================================================
_embedder_lock = threading.Lock()
_embedder = None

def get_embedder():
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise RuntimeError(
                    "sentence-transformers가 설치되어 있지 않습니다. "
                    "pip install sentence-transformers 로 설치하거나, 프로젝트 임베딩 함수를 embed_texts에 연결하세요."
                ) from e
            # 한국어 포함 멀티링구얼 모델 예시(필요에 맞게 교체)
            _embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return _embedder

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    return shape: (n, dim) float32
    cosine 유사도용으로 L2 normalize 해서 반환
    """
    model = get_embedder()
    vecs = model.encode(texts, normalize_embeddings=True)  # 이미 normalize 됨
    vecs = np.asarray(vecs, dtype=np.float32)
    return vecs


# =========================================================
# 4) FAISS Store (IndexIDMap2 + IndexFlatIP) : cosine similarity
# =========================================================
class ChatFaissStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._index: Optional[faiss.Index] = None
        self._dim: Optional[int] = None
        self._last_built_at: Optional[str] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("FAISS index not initialized")
        return self._dim

    def _ensure_index(self, dim: int):
        """
        cosine similarity = normalized vectors + inner product
        IndexIDMap2를 쓰면 id로 add/remove 가능
        """
        if self._index is None:
            base = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIDMap2(base)
            self._dim = dim

    def build_from_db(self, db: Session):
        with self._lock:
            docs = db.execute(select(ChatFaissDoc)).scalars().all()
            if not docs:
                # 빈 인덱스 생성(차원은 아직 모름)
                self._index = None
                self._dim = None
                self._last_built_at = datetime.now(timezone.utc).astimezone().isoformat()
                return

            texts = [d.user_query for d in docs]
            ids = np.array([d.id for d in docs], dtype=np.int64)

            vecs = embed_texts(texts)
            dim = vecs.shape[1]
            self._ensure_index(dim)

            # 새로 빌드
            self._index.reset()
            self._index.add_with_ids(vecs, ids)

            self._last_built_at = datetime.now(timezone.utc).astimezone().isoformat()

    def upsert_many(self, db: Session, docs_in: List[ChatFaissDocIn]) -> Tuple[int, int]:
        """
        DB upsert 후, FAISS 인덱스에도 반영
        반환: (inserted, updated)
        """
        with self._lock:
            inserted = 0
            updated = 0

            # 1) DB upsert
            ids = [d.id for d in docs_in]
            existing = db.execute(select(ChatFaissDoc).where(ChatFaissDoc.id.in_(ids))).scalars().all()
            existing_map = {d.id: d for d in existing}

            now = datetime.now(timezone.utc)

            for d in docs_in:
                row = existing_map.get(d.id)
                if row is None:
                    row = ChatFaissDoc(
                        id=d.id,
                        category=d.category,
                        user_query=d.user_query,
                        answer=d.answer,
                        metadata=d.metadata,
                        created_at=now,
                        updated_at=now,
                    )
                    db.add(row)
                    inserted += 1
                else:
                    row.category = d.category
                    row.user_query = d.user_query
                    row.answer = d.answer
                    row.meta = d.metadata # 이 키 맞춰줘야 잘들어감
                    row.updated_at = now
                    updated += 1

            db.commit()

            # 2) FAISS 반영
            # - IndexIDMap2 + remove_ids로 업데이트 가능
            texts = [d.user_query for d in docs_in]
            vecs = embed_texts(texts)
            dim = vecs.shape[1]
            self._ensure_index(dim)

            # 기존 id들 제거(있으면) 후 추가
            remove_ids = np.array([d.id for d in docs_in], dtype=np.int64)
            try:
                self._index.remove_ids(remove_ids)
            except Exception:
                # 일부 faiss 빌드/설정에서 remove가 예외일 수 있으니 최후엔 rebuild 권장
                pass

            self._index.add_with_ids(vecs, remove_ids)
            self._last_built_at = datetime.now(timezone.utc).astimezone().isoformat()

            return inserted, updated

    def delete_one(self, db: Session, doc_id: int) -> bool:
        with self._lock:
            doc = db.get(ChatFaissDoc, doc_id)
            if not doc:
                return False

            db.delete(doc)
            db.commit()

            if self._index is not None:
                try:
                    self._index.remove_ids(np.array([doc_id], dtype=np.int64))
                except Exception:
                    # remove 실패 시, 운영 정책에 따라 rebuild로 정합성 맞추는 것도 방법
                    pass

            self._last_built_at = datetime.now(timezone.utc).astimezone().isoformat()
            return True

    def search(self, db: Session, query: str, k: int, category: Optional[str], min_score: Optional[float]) -> List[SearchHit]:
        with self._lock:
            # 인덱스가 없으면 DB로부터 빌드 시도
            if self._index is None:
                self.build_from_db(db)

            if self._index is None:
                return []

            qv = embed_texts([query])  # (1, dim)
            # 카테고리 필터가 있으면 후보를 넉넉히 뽑아 post-filter
            overfetch = min(max(k * 10, k), 200)
            D, I = self._index.search(qv, overfetch)

            ids = [int(x) for x in I[0].tolist() if int(x) != -1]
            scores = [float(x) for x in D[0].tolist()]

            if not ids:
                return []

            # DB 조회
            docs = db.execute(select(ChatFaissDoc).where(ChatFaissDoc.id.in_(ids))).scalars().all()
            doc_map = {d.id: d for d in docs}

            results: List[SearchHit] = []
            for score, doc_id in zip(scores, I[0].tolist()):
                doc_id = int(doc_id)
                if doc_id == -1:
                    continue
                doc = doc_map.get(doc_id)
                if not doc:
                    continue

                if category and doc.category != category:
                    continue
                if min_score is not None and score < min_score:
                    continue

                results.append(
                    SearchHit(
                        id=doc.id,
                        score=score,
                        category=doc.category,
                        user_query=doc.user_query,
                        answer=doc.answer,
                        metadata=doc.meta,
                    )
                )
                if len(results) >= k:
                    break

            return results

    def stats(self, db: Session) -> StatsResp:
        with self._lock:
            total = db.execute(select(ChatFaissDoc.id)).scalars().all()
            total_in_db = len(total)
            indexed = int(self._index.ntotal) if self._index is not None else 0
            dim = self._dim or 0
            return StatsResp(
                total_in_db=total_in_db,
                indexed_count=indexed,
                dim=dim,
                metric="cosine (normalized vectors + inner product)",
                last_built_at=self._last_built_at,
            )


# 싱글톤 스토어
chat_faiss_store = ChatFaissStore()


# =========================================================
# 5) Helpers: 업로드 JSON 파싱
# =========================================================
def parse_docs_json(payload: Any) -> List[ChatFaissDocIn]:
    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail="업로드 데이터는 JSON 배열(list)이어야 합니다.")
    docs: List[ChatFaissDocIn] = []
    for i, item in enumerate(payload):
        try:
            docs.append(ChatFaissDocIn.model_validate(item))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"{i}번째 항목 형식이 올바르지 않습니다: {e}")
    return docs


# =========================================================
# 6) APIs
# =========================================================

# (A) JSON 파일 업로드로 적재 (multipart)
@router.post("/ingest/file", response_model=IngestResp)
async def ingest_file(
    file: UploadFile = File(..., description="문서 목록 JSON 파일"),
    db: Session = Depends(get_db),
):
    if file.content_type not in ("application/json", "text/plain"):
        raise HTTPException(status_code=400, detail="JSON 파일만 업로드 해주세요. (application/json)")

    raw = await file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON 파싱 실패: {e}")

    docs_in = parse_docs_json(payload)
    inserted, updated = chat_faiss_store.upsert_many(db, docs_in)
    stats = chat_faiss_store.stats(db)

    return IngestResp(
        inserted=inserted,
        updated=updated,
        total_in_db=stats.total_in_db,
        indexed_count=stats.indexed_count,
    )


# (B) JSON body로 바로 적재 (프론트에서 파일 없이 바로 보내는 경우)
@router.post("/ingest", response_model=IngestResp)
def ingest_json(
    docs: List[ChatFaissDocIn] = Body(...),
    db: Session = Depends(get_db),
):
    inserted, updated = chat_faiss_store.upsert_many(db, docs)
    stats = chat_faiss_store.stats(db)
    return IngestResp(
        inserted=inserted,
        updated=updated,
        total_in_db=stats.total_in_db,
        indexed_count=stats.indexed_count,
    )


# (C) 검색 (유사 사례 찾기)
@router.post("/search", response_model=SearchResp)
def search(req: SearchReq, db: Session = Depends(get_db)):
    results = chat_faiss_store.search(
        db=db,
        query=req.query,
        k=req.k,
        category=req.category,
        min_score=req.min_score,
    )
    return SearchResp(results=results)


# (D) 전체 인덱스 재빌드 (운영에서 강추)
@router.post("/rebuild", response_model=StatsResp)
def rebuild(db: Session = Depends(get_db)):
    chat_faiss_store.build_from_db(db)
    return chat_faiss_store.stats(db)


# (E) 통계/상태
@router.get("/stats", response_model=StatsResp)
def stats(db: Session = Depends(get_db)):
    return chat_faiss_store.stats(db)


# (F) 문서 단건 조회
@router.get("/docs/{doc_id}", response_model=ChatFaissDocIn)
def get_doc(doc_id: int, db: Session = Depends(get_db)):
    doc = db.get(ChatFaissDoc, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return ChatFaissDocIn(
        id=doc.id,
        category=doc.category,
        user_query=doc.user_query,
        answer=doc.answer,
        metadata=doc.meta,
    )


# (G) 문서 삭제 (DB + 인덱스)
@router.delete("/docs/{doc_id}")
def delete_doc(doc_id: int, db: Session = Depends(get_db)):
    ok = chat_faiss_store.delete_one(db, doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return {"deleted": True, "id": doc_id}


# (H) 카테고리 목록(필터 UI용)
@router.get("/categories")
def list_categories(
    db: Session = Depends(get_db),
    limit: int = Query(default=200, ge=1, le=2000),
):
    rows = db.execute(select(ChatFaissDoc.category)).scalars().all()
    uniq = sorted(list({r for r in rows}))[:limit]
    return {"categories": uniq}

# =========================================================
# 7) 히스토리 조회 API (마스킹된 내용만 반환)
# GET /chat-faiss/sessions/{session_id}/messages?limit=200
# =========================================================
@router.get("/sessions/{session_id}/messages", response_model=HistoryResp)
def get_session_messages(
    session_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    s = db.get(ChatSession, session_id)
    if not s:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    rows = db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
    ).scalars().all()

    msgs = [
        HistoryMessage(
            role=r.role,
            content=r.content_masked,
            created_at=r.created_at.astimezone().isoformat() if r.created_at else "",
        )
        for r in rows
    ]
    return HistoryResp(session_id=session_id, messages=msgs)

