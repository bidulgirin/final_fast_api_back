import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.session import get_db
from app.api.v1.deps import get_faiss_store, get_data_dir

from app.db.models.phising_case import PhisingCaseDocs
from app.schemas.phising_case import (
    DocCreate, DocUpdate, DocOut,
    SearchReq, SearchResp, SearchHit,
    UploadResp
)
from app.loader.xlsx_loader import load_grouped_docs_from_xlsx

router = APIRouter()


def rebuild_faiss_from_db(db: Session, faiss_store):
    docs = db.execute(select(PhisingCaseDocs)).scalars().all()
    if not docs:
        return

    faiss_store.index = faiss_store._load_or_create()
    faiss_store.add([d.id for d in docs], [d.text for d in docs])
    faiss_store.save()


@router.post("/docs", response_model=DocOut)
def create_doc(
    payload: DocCreate,
    db: Session = Depends(get_db),
    faiss_store=Depends(get_faiss_store),
):
    doc = PhisingCaseDocs(
        file_id=payload.file_id,
        interval=payload.interval,
        case_name=payload.case_name,
        text=payload.text,
    )
    db.add(doc)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"DB 저장 실패: {str(e)}")

    db.refresh(doc)

    try:
        faiss_store.add([doc.id], [doc.text])
        faiss_store.save()
    except Exception as e:
        db.delete(doc)
        db.commit()
        raise HTTPException(status_code=500, detail=f"FAISS 적재 실패: {str(e)}")

    return doc


@router.get("/docs/{doc_id}", response_model=DocOut)
def get_doc(doc_id: int, db: Session = Depends(get_db)):
    doc = db.get(PhisingCaseDocs, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return doc


@router.get("/docs", response_model=list[DocOut])
def list_docs(
    file_id: int | None = None,
    case_name: str | None = None,
    db: Session = Depends(get_db),
):
    stmt = select(PhisingCaseDocs)
    if file_id is not None:
        stmt = stmt.where(PhisingCaseDocs.file_id == file_id)
    if case_name is not None:
        stmt = stmt.where(PhisingCaseDocs.case_name == case_name)

    docs = db.execute(
        stmt.order_by(PhisingCaseDocs.file_id, PhisingCaseDocs.interval, PhisingCaseDocs.id)
    ).scalars().all()
    return docs


@router.put("/docs/{doc_id}", response_model=DocOut)
def update_doc(
    doc_id: int,
    payload: DocUpdate,
    db: Session = Depends(get_db),
    faiss_store=Depends(get_faiss_store),
):
    doc = db.get(PhisingCaseDocs, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")

    if payload.case_name is not None:
        doc.case_name = payload.case_name
    if payload.text is not None:
        doc.text = payload.text

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"DB 업데이트 실패: {str(e)}")

    db.refresh(doc)

    if payload.text is not None:
        try:
            faiss_store.upsert(doc.id, doc.text)
            faiss_store.save()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"FAISS 업데이트 실패: {str(e)}")

    return doc


@router.delete("/docs/{doc_id}")
def delete_doc(
    doc_id: int,
    db: Session = Depends(get_db),
    faiss_store=Depends(get_faiss_store),
):
    doc = db.get(PhisingCaseDocs, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")

    try:
        db.delete(doc)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"DB 삭제 실패: {str(e)}")

    try:
        faiss_store.remove([doc_id])
        faiss_store.save()
    except Exception as e:
        rebuild_faiss_from_db(db, faiss_store)
        raise HTTPException(status_code=500, detail=f"FAISS 삭제 실패: {str(e)}")

    return {"deleted": doc_id}


@router.post("/upload-xlsx", response_model=UploadResp)
async def upload_xlsx(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    faiss_store=Depends(get_faiss_store),
    data_dir: Path = Depends(get_data_dir),
):
    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="xlsx 파일만 업로드 가능합니다.")

    save_path = data_dir / file.filename
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        grouped = load_grouped_docs_from_xlsx(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"xlsx 파싱 실패: {str(e)}")

    created_docs = 0

    for g in grouped:
        doc = PhisingCaseDocs(file_id=g.file_id, interval=g.interval, case_name=g.case_name, text=g.text)
        db.add(doc)
        try:
            db.commit()
        except Exception:
            db.rollback()
            continue

        db.refresh(doc)

        try:
            faiss_store.add([doc.id], [doc.text])
            created_docs += 1
        except Exception:
            db.delete(doc)
            db.commit()

    faiss_store.save()
    return UploadResp(created_docs=created_docs)


@router.post("/search", response_model=SearchResp)
def search(
    req: SearchReq,
    db: Session = Depends(get_db),
    faiss_store=Depends(get_faiss_store),
):
    if req.k <= 0 or req.k > 50:
        raise HTTPException(status_code=400, detail="k는 1~50 범위로 설정하세요.")

    D, I = faiss_store.search(req.query, req.k)

    ids = [int(x) for x in I[0].tolist() if int(x) != -1]
    scores = D[0].tolist()

    if not ids:
        return SearchResp(results=[])

    doc_map = {
        d.id: d
        for d in db.execute(select(PhisingCaseDocs).where(PhisingCaseDocs.id.in_(ids))).scalars().all()
    }

    hits: list[SearchHit] = []
    for score, doc_id in zip(scores, I[0].tolist()):
        doc_id = int(doc_id)
        if doc_id == -1:
            continue
        doc = doc_map.get(doc_id)
        if not doc:
            continue

        hits.append(
            SearchHit(
                id=doc.id,
                score=float(score),
                file_id=doc.file_id,
                interval=doc.interval,
                case_name=doc.case_name,
                text=doc.text,
            )
        )

    return SearchResp(results=hits)
