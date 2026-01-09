from fastapi import FastAPI


import os
from pathlib import Path
import shutil
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.api.v1.router import router as v1_router
from app.api.v1.endpoints.chat import chat_faiss,chat_guide

# DB 관련 import (Base / engine)
from app.db.base import Base
from app.db.session import engine, get_db


# 모델들을 등록하기 위해 import (Base.metadata에 모델이 올라가도록)
import app.db.models  # noqa: F401

# faiss 모델들고 오기
from app.db.models.phising_case import PhisingCaseDocs
# 스키마
from app.schemas.phising_case import (
    DocCreate, DocUpdate, DocOut,
    SearchReq, SearchResp, SearchHit,
    UploadResp
)

# faiss 연결 + 문서 
from app.faiss.faiss_store import FaissStore
from app.loader.xlsx_loader import load_grouped_docs_from_xlsx

# faiss 관련 임포트
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/index.faiss")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


DATA_DIR = Path(FAISS_INDEX_PATH).parent
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

faiss_store = FaissStore(index_path=FAISS_INDEX_PATH, model_name=EMBED_MODEL_NAME)

# 여기서 /api/v1을 붙일 거면, v1_router 내부에는 /api나 /v1 prefix를 또 붙이지 않는 게 안전
app.include_router(v1_router, prefix="/api/v1")

# 테스트 라우터는 루트에 붙임 (prefix는 굳이 "" 안 줘도 됨)
# chat-faiss 라우터 
app.include_router(chat_faiss.router)
# chat-guide 라우터임
app.include_router(chat_guide.router)



@app.on_event("startup")
def on_startup():
    # 개발 단계 편의용: 테이블 자동 생성
    Base.metadata.create_all(bind=engine)


# 여기부터 faiss 시작
def rebuild_faiss_from_db(db: Session):
    docs = db.execute(select(PhisingCaseDocs)).scalars().all()
    if not docs:
        return

    faiss_store.index = faiss_store._load_or_create()
    faiss_store.add([d.id for d in docs], [d.text for d in docs])
    faiss_store.save()

@app.post("/docs", response_model=DocOut)
def create_doc(payload: DocCreate, db: Session = Depends(get_db)):
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

@app.get("/docs/{doc_id}", response_model=DocOut)
def get_doc(doc_id: int, db: Session = Depends(get_db)):
    doc = db.get(PhisingCaseDocs, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return doc

@app.get("/docs", response_model=list[DocOut])
def list_docs(file_id: int | None = None, case_name: str | None = None, db: Session = Depends(get_db)):
    stmt = select(PhisingCaseDocs)
    if file_id is not None:
        stmt = stmt.where(PhisingCaseDocs.file_id == file_id)
    if case_name is not None:
        stmt = stmt.where(PhisingCaseDocs.case_name == case_name)
    docs = db.execute(stmt.order_by(PhisingCaseDocs.file_id, PhisingCaseDocs.interval, PhisingCaseDocs.id)).scalars().all()
    return docs

@app.put("/docs/{doc_id}", response_model=DocOut)
def update_doc(doc_id: int, payload: DocUpdate, db: Session = Depends(get_db)):
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

@app.delete("/docs/{doc_id}")
def delete_doc(doc_id: int, db: Session = Depends(get_db)):
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
        rebuild_faiss_from_db(db)
        raise HTTPException(status_code=500, detail=f"FAISS 삭제 실패: {str(e)}")

    return {"deleted": doc_id}

@app.post("/upload-xlsx", response_model=UploadResp)
async def upload_xlsx(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="xlsx 파일만 업로드 가능합니다.")

    save_path = DATA_DIR / file.filename
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

@app.post("/search", response_model=SearchResp)
def search(req: SearchReq, db: Session = Depends(get_db)):
    # k 값 검증: 1~50 범위가 아니면 400 에러
    if req.k <= 0 or req.k > 50:
        raise HTTPException(status_code=400, detail="k는 1~50 범위로 설정하세요.")

    # FAISS 벡터 검색 수행
    # D: 거리/유사도 스코어(보통 distance거나 -cosine 같은 형태)
    # I: 매칭된 문서(벡터) ID 인덱스 (-1은 빈자리/미매칭 의미)
    D, I = faiss_store.search(req.query, req.k)

    hits: list[SearchHit] = []

    # FAISS 결과에서 -1이 아닌 ID만 뽑아 DB 조회용 리스트로 준비
    ids = [int(x) for x in I[0].tolist() if int(x) != -1]

    # 스코어 리스트 (I[0]과 같은 길이; -1 자리에도 스코어 값은 있을 수 있음)
    scores = D[0].tolist()

    # 유효한 ID가 하나도 없으면 빈 결과 반환
    if not ids:
        return SearchResp(results=[])

    # DB에서 ids에 해당하는 문서들을 한 번에 조회한 뒤,
    # {doc_id: doc} 형태로 맵을 만들어 O(1)로 꺼낼 수 있게 함 (N+1 방지)
    doc_map = {
        d.id: d
        for d in db.execute(
            select(PhisingCaseDocs).where(PhisingCaseDocs.id.in_(ids))
        ).scalars().all()
    }

    # FAISS 결과 순서를 유지하기 위해 스코어와 doc_id를 zip으로 같이 순회
    for score, doc_id in zip(scores, I[0].tolist()):
        doc_id = int(doc_id)

        # -1은 FAISS에서 결과가 없음을 의미하므로 스킵
        if doc_id == -1:
            continue

        # DB에서 가져온 문서가 있으면 사용, 없으면 스킵 (동기화 문제 대비)
        doc = doc_map.get(doc_id)
        if not doc:
            continue

        # 응답 모델(SearchHit)로 변환해서 결과에 추가
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

    # 최종 결과 반환
    return SearchResp(results=hits)


@app.post("/admin/rebuild-faiss")
def admin_rebuild_faiss(db: Session = Depends(get_db)):
    rebuild_faiss_from_db(db)
    return {"status": "ok"}

@app.get("/admin/faiss-info")
def faiss_info():
    path = FAISS_INDEX_PATH
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

# 연결 체크
@app.get("/health")
def health():
    return {"ok": True}
