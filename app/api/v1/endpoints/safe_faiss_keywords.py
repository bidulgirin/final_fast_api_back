from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.services.text_infer import TextInferConfig, get_safe_kw_store

router = APIRouter(prefix="/faiss/safe-keywords", tags=["faiss-safe-keywords"])

_CFG = TextInferConfig()


class KeywordsPayload(BaseModel):
    keywords: List[str]


@router.post("/rebuild")
def rebuild_keywords(payload: KeywordsPayload):
    try:
        store = get_safe_kw_store(_CFG)
        n = store.rebuild(payload.keywords)
        return {"ok": True, "rebuilt": n, "total": store.index.ntotal}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upsert")
def upsert_keywords(payload: KeywordsPayload):
    try:
        store = get_safe_kw_store(_CFG)
        result = store.upsert(payload.keywords)
        return {"ok": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("")
def delete_keywords(payload: KeywordsPayload):
    try:
        store = get_safe_kw_store(_CFG)
        removed = store.remove(payload.keywords)
        return {"ok": True, "removed": removed, "total": store.index.ntotal}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
def search(q: str, k: Optional[int] = None, min_sim: Optional[float] = None):
    try:
        store = get_safe_kw_store(_CFG)
        topk = _CFG.safe_faiss_topk if k is None else int(k)
        threshold = _CFG.safe_faiss_min_sim if min_sim is None else float(min_sim)
        hits = store.search(q, topk=topk, min_sim=threshold)
        return {"ok": True, "q": q, "hits": hits}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
def stats():
    try:
        store = get_safe_kw_store(_CFG)
        return {
            "ok": True,
            "total": int(store.index.ntotal),
            "meta_keywords": len(store.kw_to_id),
            "index_path": str(store.index_path),
            "meta_path": str(store.meta_path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
