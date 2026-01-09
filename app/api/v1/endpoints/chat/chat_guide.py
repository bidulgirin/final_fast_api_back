# app/api/v1/endpoints/chat/chat_guide.py
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Optional, List, Dict, Tuple

import numpy as np
import faiss

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body, Query
from pydantic import BaseModel, Field

from sqlalchemy import Column, Integer, String, Text, DateTime, select
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base import Base
from app.db.session import get_db

# =========================================================
# 1) DB Model: 가이드 저장
# =========================================================
class ChatGuideDoc(Base):
    __tablename__ = "chat_guide_docs"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(64), index=True, unique=True, nullable=False)   # "피해신고" 같은 key
    title = Column(Text, nullable=False)                                # "🚨 즉시 피해 신고"
    content = Column(Text, nullable=False)                              # 안내 문구

    meta = Column(JSONB, nullable=True)  # 확장용(선택): {"tags":..., "urgency":...} 등

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))


# =========================================================
# 2) Pydantic Schemas
# =========================================================
class ChatGuideDocIn(BaseModel):
    id: int
    key: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None  # 요청 JSON의 확장 필드

class IngestResp(BaseModel):
    inserted: int
    updated: int
    total_in_db: int
    indexed_count: int

class SearchReq(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    min_score: Optional[float] = None  # cosine similarity threshold

class SearchHit(BaseModel):
    id: int
    score: float
    key: str
    title: str
    content: str
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
# 3) Embedding
#    - chat_faiss.py랑 동일 모델을 쓴다는 전제(중복 로딩될 수 있음)
#    - 최적화하려면 embed_texts를 공용 모듈로 빼는 걸 추천
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
                    "pip install sentence-transformers"
                ) from e
            _embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return _embedder

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)

 # chat_guide.py (일부 발췌 / 교체)
def _flatten_meta(meta: Optional[Dict[str, Any]]) -> str:
    """
    메타데이터를 검색/임베딩에 도움 되는 문자열로 펼침
    (FAISS 임베딩용: 너무 길면 품질/비용에 악영향 -> 적당히 제한)
    """
    if not meta:
        return ""

    def _take_list(key: str, limit: int = 20) -> List[str]:
        v = meta.get(key)
        if isinstance(v, list):
            out = []
            for x in v[:limit]:
                if isinstance(x, str):
                    out.append(x)
                elif isinstance(x, dict):
                    # dict는 key:value 형태로 간단히
                    out.append(" ".join([f"{k}:{x[k]}" for k in x.keys() if x.get(k) is not None])[:200])
            return out
        return []

    tags = _take_list("tags", 30)
    stage = _take_list("stage", 10)
    kws = _take_list("trigger_keywords", 50)

    # contacts/links/actions는 dict list일 가능성이 큼
    contacts = _take_list("contacts", 10)
    links = _take_list("links", 10)
    actions = _take_list("actions", 10)

    risk = meta.get("risk_level")
    priority = meta.get("priority")

    chunks = []
    if tags: chunks.append("TAGS: " + ", ".join(tags))
    if stage: chunks.append("STAGE: " + ", ".join(stage))
    if kws: chunks.append("KW: " + ", ".join(kws))
    if contacts: chunks.append("CONTACTS: " + " | ".join(contacts))
    if links: chunks.append("LINKS: " + " | ".join(links))
    if actions: chunks.append("ACTIONS: " + " | ".join(actions))
    if risk: chunks.append(f"RISK:{risk}")
    if priority is not None: chunks.append(f"PRIORITY:{priority}")

    # 너무 길어지면 truncate
    s = "\n".join(chunks)
    return s[:1500]

# =========================================================
# 4) FAISS Store (IndexIDMap2 + IndexFlatIP): cosine
# =========================================================
class ChatGuideStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._index: Optional[faiss.Index] = None
        self._dim: Optional[int] = None
        self._last_built_at: Optional[str] = None

    def _ensure_index(self, dim: int):
        if self._index is None:
            base = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIDMap2(base)
            self._dim = dim

    def _doc_to_text(self, d: ChatGuideDoc) -> str:
        meta_txt = _flatten_meta(d.meta)
        return f"{d.title}\n{d.content}\nKEY:{d.key}\n{meta_txt}".strip()

    def build_from_db(self, db: Session):
        with self._lock:
            docs = db.execute(select(ChatGuideDoc)).scalars().all()
            if not docs:
                self._index = None
                self._dim = None
                self._last_built_at = datetime.now(timezone.utc).astimezone().isoformat()
                return

            texts = [self._doc_to_text(d) for d in docs]
            ids = np.array([d.id for d in docs], dtype=np.int64)

            vecs = embed_texts(texts)
            dim = vecs.shape[1]
            self._ensure_index(dim)

            self._index.reset()
            self._index.add_with_ids(vecs, ids)
            self._last_built_at = datetime.now(timezone.utc).astimezone().isoformat()

    def upsert_many(self, db: Session, docs_in: List[ChatGuideDocIn]) -> Tuple[int, int]:
        with self._lock:
            inserted = 0
            updated = 0

            ids = [d.id for d in docs_in]
            existing = db.execute(select(ChatGuideDoc).where(ChatGuideDoc.id.in_(ids))).scalars().all()
            existing_map = {d.id: d for d in existing}

            # FAISS 반영 시도: upsert된 docs_in 기반으로 임베딩 텍스트 생성
            texts = []

            now = datetime.now(timezone.utc)
            for d in docs_in:
                row = existing_map.get(d.id)

                if row is None:
                    row = ChatGuideDoc(
                        id=d.id,
                        key=d.key,
                        title=d.title,
                        content=d.content,
                        meta=d.metadata,
                        created_at=now,
                        updated_at=now,
                    )
                    db.add(row)
                    inserted += 1
                else:
                    row.key = d.key
                    row.title = d.title
                    row.content = d.content
                    row.meta = d.metadata
                    row.updated_at = now
                    updated += 1
            

            db.commit()

            # FAISS 반영
            texts = []
            for d in docs_in:
                meta_txt = _flatten_meta(d.metadata)
                # texts.append(f"{d.title}\n{d.content}\nKEY:{d.key}")
                texts.append(f"{d.title}\n{d.content}\nKEY:{d.key}\n{meta_txt}".strip())
            vecs = embed_texts(texts)
            dim = vecs.shape[1]
            self._ensure_index(dim)

            remove_ids = np.array([d.id for d in docs_in], dtype=np.int64)
            try:
                self._index.remove_ids(remove_ids)
            except Exception:
                pass

            self._index.add_with_ids(vecs, remove_ids)
            self._last_built_at = datetime.now(timezone.utc).astimezone().isoformat()

            return inserted, updated

    def delete_one(self, db: Session, doc_id: int) -> bool:
        with self._lock:
            doc = db.get(ChatGuideDoc, doc_id)
            if not doc:
                return False

            db.delete(doc)
            db.commit()

            if self._index is not None:
                try:
                    self._index.remove_ids(np.array([doc_id], dtype=np.int64))
                except Exception:
                    pass

            self._last_built_at = datetime.now(timezone.utc).astimezone().isoformat()
            return True

    def search(self, db: Session, query: str, k: int, min_score: Optional[float]) -> List[SearchHit]:
        with self._lock:
            if self._index is None:
                self.build_from_db(db)
            if self._index is None:
                return []

            qv = embed_texts([query])
            overfetch = min(max(k * 10, k), 200)
            D, I = self._index.search(qv, overfetch)

            ids = [int(x) for x in I[0].tolist() if int(x) != -1]
            scores = [float(x) for x in D[0].tolist()]
            if not ids:
                return []

            docs = db.execute(select(ChatGuideDoc).where(ChatGuideDoc.id.in_(ids))).scalars().all()
            doc_map = {d.id: d for d in docs}

            results: List[SearchHit] = []
            for score, doc_id in zip(scores, I[0].tolist()):
                doc_id = int(doc_id)
                if doc_id == -1:
                    continue
                if min_score is not None and score < min_score:
                    continue

                doc = doc_map.get(doc_id)
                if not doc:
                    continue

                results.append(
                    SearchHit(
                        id=doc.id,
                        score=score,
                        key=doc.key,
                        title=doc.title,
                        content=doc.content,
                        metadata=doc.meta,
                    )
                )
                if len(results) >= k:
                    break

            return results

    def stats(self, db: Session) -> StatsResp:
        with self._lock:
            total_in_db = db.execute(select(ChatGuideDoc.id)).scalars().all()
            indexed = int(self._index.ntotal) if self._index is not None else 0
            return StatsResp(
                total_in_db=len(total_in_db),
                indexed_count=indexed,
                dim=self._dim or 0,
                metric="cosine (normalized vectors + inner product)",
                last_built_at=self._last_built_at,
            )
   
chat_guide_store = ChatGuideStore()


# =========================================================
# 5) 업로드 JSON 파서: dict 또는 list 모두 지원
# =========================================================
def parse_guides_payload(payload: Any) -> List[ChatGuideDocIn]:
    """
    지원 형식:
    1) dict 형태:
       {
         "1": {"key":"...", "title":"...", "content":"..."},
         "2": {...}
       }
    2) list 형태:
       [
         {"id":1, "key":"...", "title":"...", "content":"..."},
         ...
       ]
    """
    if isinstance(payload, dict):
        docs: List[ChatGuideDocIn] = []
        for k, v in payload.items():
            if not isinstance(v, dict):
                raise HTTPException(status_code=400, detail="dict payload의 값은 object여야 합니다.")
            try:
                doc_id = int(k)
            except Exception:
                raise HTTPException(status_code=400, detail=f"키 '{k}'는 int로 변환 가능해야 합니다.")
            item = {"id": doc_id, **v}
            # v 안에 metadata 같은 확장 필드가 들어올 수 있으니 그대로 받되, 스키마에 맞게 매핑
            # 여기서는 metadata 키가 없다면 None
            if "metadata" not in item and "meta" in item:
                item["metadata"] = item["meta"]

            try:
                docs.append(ChatGuideDocIn.model_validate(item))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"id={doc_id} 항목 형식 오류: {e}")
        return docs

    if isinstance(payload, list):
        docs: List[ChatGuideDocIn] = []
        for i, item in enumerate(payload):
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail=f"{i}번째 항목은 object여야 합니다.")
            try:
                docs.append(ChatGuideDocIn.model_validate(item))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{i}번째 항목 형식 오류: {e}")
        return docs

    raise HTTPException(status_code=400, detail="업로드 데이터는 dict 또는 list 형식이어야 합니다.")


# =========================================================
# 6) Router / APIs
# =========================================================
router = APIRouter(prefix="/chat-guide", tags=["chat-guide"])


@router.post("/ingest", response_model=IngestResp)
def ingest_guides(
    payload: Any = Body(...),
    db: Session = Depends(get_db),
):
    docs_in = parse_guides_payload(payload)
    inserted, updated = chat_guide_store.upsert_many(db, docs_in)
    stats = chat_guide_store.stats(db)
    return IngestResp(
        inserted=inserted,
        updated=updated,
        total_in_db=stats.total_in_db,
        indexed_count=stats.indexed_count,
    )


@router.post("/ingest/file", response_model=IngestResp)
async def ingest_guides_file(
    file: UploadFile = File(..., description="가이드 JSON 파일 (dict 또는 list)"),
    db: Session = Depends(get_db),
):
    if file.content_type not in ("application/json", "text/plain"):
        raise HTTPException(status_code=400, detail="JSON 파일만 업로드 해주세요.")

    raw = await file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON 파싱 실패: {e}")

    docs_in = parse_guides_payload(payload)
    inserted, updated = chat_guide_store.upsert_many(db, docs_in)
    stats = chat_guide_store.stats(db)
    return IngestResp(
        inserted=inserted,
        updated=updated,
        total_in_db=stats.total_in_db,
        indexed_count=stats.indexed_count,
    )


@router.post("/search", response_model=SearchResp)
def search_guides(req: SearchReq, db: Session = Depends(get_db)):
    results = chat_guide_store.search(
        db=db,
        query=req.query,
        k=req.k,
        min_score=req.min_score,
    )
    return SearchResp(results=results)


@router.post("/rebuild", response_model=StatsResp)
def rebuild(db: Session = Depends(get_db)):
    chat_guide_store.build_from_db(db)
    return chat_guide_store.stats(db)


@router.get("/stats", response_model=StatsResp)
def stats(db: Session = Depends(get_db)):
    return chat_guide_store.stats(db)


@router.get("/guides/{doc_id}", response_model=ChatGuideDocIn)
def get_guide(doc_id: int, db: Session = Depends(get_db)):
    doc = db.get(ChatGuideDoc, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="가이드를 찾을 수 없습니다.")
    return ChatGuideDocIn(
        id=doc.id,
        key=doc.key,
        title=doc.title,
        content=doc.content,
        metadata=doc.meta,
    )


@router.delete("/guides/{doc_id}")
def delete_guide(doc_id: int, db: Session = Depends(get_db)):
    ok = chat_guide_store.delete_one(db, doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="가이드를 찾을 수 없습니다.")
    return {"deleted": True, "id": doc_id}


@router.get("/keys")
def list_keys(
    db: Session = Depends(get_db),
    limit: int = Query(default=200, ge=1, le=2000),
):
    rows = db.execute(select(ChatGuideDoc.key)).scalars().all()
    uniq = sorted(list({r for r in rows}))[:limit]
    return {"keys": uniq}
