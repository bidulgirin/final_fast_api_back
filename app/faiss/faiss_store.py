from __future__ import annotations

import os
import threading
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
# FAISS 스토어 (IndexIDMap2 + remove/update 지원)
class FaissStore:
    def __init__(self, index_path: str, model_name: str):
        self.index_path = index_path
        self.model = SentenceTransformer(model_name)

        self.dim = self.model.get_sentence_embedding_dimension()
        self.lock = threading.Lock()

        self.index = self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.index_path):
            idx = faiss.read_index(self.index_path)
            return idx

        base = faiss.IndexFlatIP(self.dim)
        idx = faiss.IndexIDMap2(base)
        return idx

    def save(self):
        with self.lock:
            faiss.write_index(self.index, self.index_path)

    def embed(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(texts, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(emb)
        return emb

    def add(self, doc_ids: list[int], texts: list[str]):
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids와 texts 길이가 다릅니다.")

        vecs = self.embed(texts)
        ids = np.array(doc_ids, dtype="int64")

        with self.lock:
            self.index.add_with_ids(vecs, ids)

    def remove(self, doc_ids: list[int]):
        ids = np.array(doc_ids, dtype="int64")
        with self.lock:
            self.index.remove_ids(ids)

    def upsert(self, doc_id: int, text: str):
        with self.lock:
            self.index.remove_ids(np.array([doc_id], dtype="int64"))
        self.add([doc_id], [text])

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        qv = self.embed([query])
        with self.lock:
            D, I = self.index.search(qv, k)
        return D, I
