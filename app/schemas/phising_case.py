from __future__ import annotations

from pydantic import BaseModel, Field
# phising_case 와 관련된 doc 만들기요

class DocCreate(BaseModel):
    file_id: int
    interval: int
    case_name: str = Field(max_length=255)
    text: str

class DocUpdate(BaseModel):
    case_name: str | None = Field(default=None, max_length=255)
    text: str | None = None

class DocOut(BaseModel):
    id: int
    file_id: int
    interval: int
    case_name: str
    text: str

    class Config:
        from_attributes = True

class SearchReq(BaseModel):
    query: str
    k: int = 5

class SearchHit(BaseModel):
    id: int
    score: float
    file_id: int
    interval: int
    case_name: str
    text: str

class SearchResp(BaseModel):
    results: list[SearchHit]

class UploadResp(BaseModel):
    created_docs: int
