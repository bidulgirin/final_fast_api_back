# DB + FAISS 의존성 제공
from fastapi import Request
from sqlalchemy.orm import Session

from app.db.session import get_db


def get_faiss_store(request: Request):
    return request.app.state.faiss_store


def get_data_dir(request: Request):
    return request.app.state.data_dir
