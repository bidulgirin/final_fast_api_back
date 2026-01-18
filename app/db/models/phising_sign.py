# app/db/models/phising_sign.py

from __future__ import annotations

import re
import os
import json
import hashlib
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

import faiss  # pip install faiss-cpu
from typing import List


# ----------------------------
# 1) AE 모델 구조
# ----------------------------
class PhishingFilterAE(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 256), nn.ReLU(),
            nn.Linear(256, 1024), nn.ReLU(),
            nn.Linear(1024, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ----------------------------
# 2) 마스킹(전처리)
# ----------------------------
def advanced_deidentify(text: str) -> str:
    if not isinstance(text, str):
        return ""
    titles = r"님|씨|과장|팀장|대리|부장|차장|주임|선생님|교수님"
    text = re.sub(rf'([가-힣]{{2,4}})({titles})', r'[NAME]\2', text)
    text = re.sub(r'([가-힣]{{2,4}})\s*(수사관|검사|사무관|조사관|드림|올림)', r'[NAME] \2', text)
    text = re.sub(r'\d{2,3}-\d{3,4}-\d{4}', '[TEL]', text)
    text = re.sub(r'\d{10,14}', '[ACC]', text)
    text = re.sub(r'http[s]?://\S+', '[URL]', text)
    text = re.sub(r'\d{4,}', '[NUM]', text)
    return text


# ----------------------------
# 3) FAISS 키워드 스토어
#    - TFIDF(vec)로 키워드 임베딩 만들고
#    - IndexIDMap2 + FlatIP(코사인) 사용
# ----------------------------
def _stable_id_from_text(s: str) -> int:
    """
    keyword -> stable int64 id
    (Python hash는 프로세스마다 달라질 수 있어 사용 금지)
    """
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    # 63-bit 양수로 줄임
    return int(h[:16], 16) & 0x7FFFFFFFFFFFFFFF


def _l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + eps)


class FaissKeywordStore:
    def __init__(
        self,
        vec,  # TfidfVectorizer
        dim: int,
        index_path: str | Path,
        meta_path: str | Path,
    ):
        self.vec = vec
        self.dim = int(dim)
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)

        self._lock = threading.RLock()

        # id <-> keyword
        self.id_to_kw: Dict[int, str] = {}
        self.kw_to_id: Dict[str, int] = {}

        self.index = self._create_empty_index()
        self.load()

    def _create_empty_index(self) -> faiss.Index:
        base = faiss.IndexFlatIP(self.dim)  # cosine similarity = normalized vec + inner product
        return faiss.IndexIDMap2(base)

    def _embed_keywords(self, keywords: List[str]) -> np.ndarray:
        # TF-IDF는 sparse -> dense float32
        cleaned = [advanced_deidentify(k) for k in keywords]
        x = self.vec.transform(cleaned).toarray().astype("float32")
        x = _l2_normalize(x)
        return x

    def _embed_sentence(self, sentence: str) -> np.ndarray:
        cleaned = advanced_deidentify(sentence)
        x = self.vec.transform([cleaned]).toarray().astype("float32")
        x = _l2_normalize(x)
        return x

    def load(self) -> None:
        with self._lock:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))

            if self.meta_path.exists():
                data = json.loads(self.meta_path.read_text(encoding="utf-8"))
                # json은 key가 str로 저장되므로 int로 변환
                self.id_to_kw = {int(k): v for k, v in data.get("id_to_kw", {}).items()}
                self.kw_to_id = {k: int(v) for k, v in data.get("kw_to_id", {}).items()}

    def save(self) -> None:
        with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
            payload = {
                "id_to_kw": {str(k): v for k, v in self.id_to_kw.items()},
                "kw_to_id": self.kw_to_id,
            }
            self.meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def rebuild(self, keywords: List[str]) -> int:
        """
        전체 재적재(리빌드): 기존 데이터 모두 교체
        """
        keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
        if not keywords:
            with self._lock:
                self.index = self._create_empty_index()
                self.id_to_kw.clear()
                self.kw_to_id.clear()
                self.save()
            return 0

        with self._lock:
            self.index = self._create_empty_index()
            self.id_to_kw.clear()
            self.kw_to_id.clear()

            ids = np.array([_stable_id_from_text(k) for k in keywords], dtype="int64")
            vecs = self._embed_keywords(keywords)

            self.index.add_with_ids(vecs, ids)

            for kw, _id in zip(keywords, ids.tolist()):
                self.id_to_kw[_id] = kw
                self.kw_to_id[kw] = _id

            self.save()
            return len(keywords)

    def upsert(self, keywords: List[str]) -> Dict[str, Any]:
        """
        키워드 추가/업데이트:
        - 이미 존재하면 remove_ids 후 add_with_ids로 교체
        """
        keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
        if not keywords:
            return {"added": 0, "updated": 0, "total": self.index.ntotal}

        with self._lock:
            add_list: List[str] = []
            add_ids: List[int] = []
            updated = 0
            added = 0

            # 먼저 remove 할 id들 수집
            to_remove: List[int] = []
            for kw in keywords:
                _id = _stable_id_from_text(kw)
                if _id in self.id_to_kw:
                    to_remove.append(_id)
                    updated += 1
                else:
                    added += 1
                add_list.append(kw)
                add_ids.append(_id)

            if to_remove:
                selector = faiss.IDSelectorBatch(np.array(to_remove, dtype="int64"))
                self.index.remove_ids(selector)

            vecs = self._embed_keywords(add_list)
            ids = np.array(add_ids, dtype="int64")
            self.index.add_with_ids(vecs, ids)

            for kw, _id in zip(add_list, add_ids):
                self.id_to_kw[_id] = kw
                self.kw_to_id[kw] = _id

            self.save()
            return {"added": added, "updated": updated, "total": self.index.ntotal}

    def remove(self, keywords: List[str]) -> int:
        keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
        if not keywords:
            return 0

        with self._lock:
            ids = []
            for kw in keywords:
                _id = _stable_id_from_text(kw)
                if _id in self.id_to_kw:
                    ids.append(_id)

            if not ids:
                return 0

            selector = faiss.IDSelectorBatch(np.array(ids, dtype="int64"))
            removed = self.index.remove_ids(selector)

            for _id in ids:
                kw = self.id_to_kw.pop(_id, None)
                if kw:
                    self.kw_to_id.pop(kw, None)

            self.save()
            return int(removed)

    def search(self, sentence: str, topk: int = 10, min_sim: float = 0.25) -> List[Dict[str, Any]]:
        """
        sentence -> topK 키워드 (cosine similarity)
        """
        if self.index.ntotal == 0:
            return []

        x = self._embed_sentence(sentence)
        with self._lock:
            sims, ids = self.index.search(x, topk)

        out = []
        for sim, _id in zip(sims[0].tolist(), ids[0].tolist()):
            if _id == -1:
                continue
            if sim < min_sim:
                continue
            kw = self.id_to_kw.get(int(_id))
            if not kw:
                continue
            out.append({"keyword": kw, "sim": float(sim), "id": int(_id)})
        return out


# ----------------------------
# 4) AE 탐지기 + (FAISS 키워드)
# ----------------------------
class PhishingDetectorAE:
    """
    - ckpt 포맷:
      {'vec': TfidfVectorizer, 'input_dim': int, 'state': model_state_dict}
    - keywords 파라미터가 없으면 FAISS에서 자동 검색
    """

    def __init__(self, model_path: str | Path, kw_store: Optional[FaissKeywordStore] = None):
        self.model_path = str(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(self.model_path, map_location=self.device)

        self.vec = ckpt["vec"]
        self.input_dim = int(ckpt["input_dim"])
        self.model = PhishingFilterAE(self.input_dim).to(self.device)
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()

        self.kw_store = kw_store  # 나중에 attach 가능

    def attach_kw_store(self, kw_store: FaissKeywordStore) -> None:
        self.kw_store = kw_store

    def _score(self, sentence: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        cleaned = advanced_deidentify(sentence)
        x_vec = self.vec.transform([cleaned]).toarray()
        x_tensor = torch.FloatTensor(x_vec).to(self.device)

        with torch.no_grad():
            pred = self.model(x_tensor)
            base_loss = torch.abs(pred - x_tensor).sum().item()

        detected_kw = keywords or []
        penalty = 1.0
        for _ in detected_kw:
            penalty *= 20.0

        final_score = base_loss * penalty

        if final_score > 150:
            label = "🚨 차단"
        elif final_score > 80:
            label = "⚠️ 주의"
        else:
            label = "✅ 정상"

        return {
            "result": label,
            "score": round(float(final_score), 2),
            "base_loss": round(float(base_loss), 4),
            "keywords": detected_kw,
            "cleaned": cleaned,
        }

    def predict(
        self,
        sentence: str,
        keywords: Optional[List[str]] = None,
        *,
        faiss_topk: int = 10,
        faiss_min_sim: float = 0.25,
    ) -> Dict[str, Any]:
        if not sentence or not isinstance(sentence, str):
            return {"result": "✅ 정상", "score": 0.0, "keywords": []}

        # keywords가 안 들어오면 FAISS에서 자동으로 뽑아 사용
        if (not keywords) and self.kw_store is not None:
            faiss_hits = self.kw_store.search(sentence, topk=faiss_topk, min_sim=faiss_min_sim)
            keywords = [h["keyword"] for h in faiss_hits]

        out = self._score(sentence, keywords=keywords)
        out["faiss_hits"] = faiss_hits
        return out


# ----------------------------
# 5) 싱글톤 초기화
# ----------------------------
DEFAULT_AE_PATH = Path("assets/models/final_ae.pth")

# FAISS 저장 경로
DEFAULT_FAISS_INDEX = Path("assets/faiss/keyword.index")
DEFAULT_FAISS_META = Path("assets/faiss/keyword_meta.json")

# 먼저 AE 로드
ae_detector = PhishingDetectorAE(DEFAULT_AE_PATH)

# AE의 vec/input_dim으로 FAISS store 생성/로드
kw_store = FaissKeywordStore(
    vec=ae_detector.vec,
    dim=ae_detector.input_dim,
    index_path=DEFAULT_FAISS_INDEX,
    meta_path=DEFAULT_FAISS_META,
)

# AE에 attach
ae_detector.attach_kw_store(kw_store)


def get_keywords_from_faiss(sentence: str, topk: int = 5, min_sim: float = 0.35) -> List[str]:
    """
    sentence를 FAISS에서 검색해 관련 키워드 topK를 반환
    """
    hits = kw_store.search(sentence, topk=topk, min_sim=min_sim)
    return [h["keyword"] for h in hits]
