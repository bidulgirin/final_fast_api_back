from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.db.models.phising_sign import kw_store

router = APIRouter(prefix="/faiss/keywords", tags=["faiss-keywords"])


class KeywordsPayload(BaseModel):
    keywords: List[str]


class SearchResponse(BaseModel):
    q: str
    hits: list


@router.post("/rebuild")
def rebuild_keywords(payload: KeywordsPayload):
    try:
        n = kw_store.rebuild(payload.keywords)
        return {"ok": True, "rebuilt": n, "total": kw_store.index.ntotal}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upsert")
def upsert_keywords(payload: KeywordsPayload):
    try:
        result = kw_store.upsert(payload.keywords)
        return {"ok": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("")
def delete_keywords(payload: KeywordsPayload):
    try:
        removed = kw_store.remove(payload.keywords)
        return {"ok": True, "removed": removed, "total": kw_store.index.ntotal}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
def search(q: str, k: int = 5, min_sim: float = 0.35):
    """
    텍스트로 검색해서 키워드 후보 반환(디버깅/운영 확인용)
    """
    try:
        hits = kw_store.search(q, topk=k, min_sim=min_sim)
        return {"ok": True, "q": q, "hits": hits}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
