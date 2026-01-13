import os
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.session import get_db
from app.api.v1.deps import get_faiss_store
from app.db.models.phising_case import PhisingCaseDocs

router = APIRouter(prefix="/admin")


def rebuild_faiss_from_db(db: Session, faiss_store):
    docs = db.execute(select(PhisingCaseDocs)).scalars().all()
    if not docs:
        return

    faiss_store.index = faiss_store._load_or_create()
    faiss_store.add([d.id for d in docs], [d.text for d in docs])
    faiss_store.save()


@router.post("/rebuild-faiss")
def admin_rebuild_faiss(
    db: Session = Depends(get_db),
    faiss_store=Depends(get_faiss_store),
):
    rebuild_faiss_from_db(db, faiss_store)
    return {"status": "ok"}


@router.get("/faiss-info")
def faiss_info(faiss_store=Depends(get_faiss_store)):
    path = getattr(faiss_store, "index_path", None) or os.getenv("FAISS_INDEX_PATH", "./data/index.faiss")
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0

    return {
        "faiss_index_path": path,
        "exists": exists,
        "file_size_bytes": size,
        "ntotal": int(faiss_store.index.ntotal),
        "dim": int(faiss_store.index.d),
        "is_trained": bool(faiss_store.index.is_trained),
        "index_type": str(type(faiss_store.index)),
    }
