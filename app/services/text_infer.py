# app/services/text_infer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import re
import json
import hashlib
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.feature_extraction.text  # noqa: F401
from transformers import BertForSequenceClassification, AutoTokenizer

import faiss  # pip install faiss-cpu


# torch load 시 vectorizer(TfidfVectorizer) 안전 글로벌 등록
torch.serialization.add_safe_globals([sklearn.feature_extraction.text.TfidfVectorizer])


# ----------------------------
# 1) AE 모델 구조
# ----------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 256), nn.ReLU(),
            nn.Linear(256, 1024), nn.ReLU(),
            nn.Linear(1024, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ----------------------------
# 2) 마스킹(전처리)
# ----------------------------
# def advanced_deidentify(text: str) -> str:
#     """
#     키워드/문장 임베딩 전에 개인정보를 마스킹해서
#     TF-IDF 공간에서 유사도가 의미있게 나오도록 처리
#     """
#     if not isinstance(text, str):
#         return ""
#     titles = r"님|씨|과장|팀장|대리|부장|차장|주임|선생님|교수님"
#     text = re.sub(rf'([가-힣]{{2,4}})({titles})', r'[NAME]\2', text)
#     text = re.sub(r'([가-힣]{{2,4}})\s*(수사관|검사|사무관|조사관|드림|올림)', r'[NAME] \2', text)
#     text = re.sub(r'\d{2,3}-\d{3,4}-\d{4}', '[TEL]', text)
#     text = re.sub(r'\d{10,14}', '[ACC]', text)
#     text = re.sub(r'http[s]?://\S+', '[URL]', text)
#     text = re.sub(r'\d{4,}', '[NUM]', text)
#     return text


def advanced_deidentify(text):
    if not isinstance(text, str): return ""
    
    # 1. 상세 이름 및 호칭/직위 결합 (성함, 고객님, 수사관 등)
    titles = r"님|씨|과장|팀장|대리|부장|차장|주임|선생님|교수님|수사관|검사|사무관|조사관"
    text = re.sub(rf'([가-힣]{{2,4}})\s*({titles})', r'[NAME] \2', text)
    
    # 2. 주소 (도로명 주소 및 지번 주소 패턴)
    # ~~시/도 ~~구/군 ~~동/읍/면/로/길 및 번지수 대응
    addr_pattern = r'([가-힣]+[시도]\s+)?[가-힣]+[구군]\s+[가-힣\d]+(동|읍|면|로|길)(\s+\d+(-?\d+)?)?'
    text = re.sub(addr_pattern, '[ADDR]', text)
    
    # 3. 주민등록번호 (앞뒤 13자리 및 성별 구분자 포함)
    text = re.sub(r'\d{6}-[1-4]\d{6}', '[ID_NUM]', text)
    
    # 4. 전화번호 및 계좌번호
    text = re.sub(r'\d{2,3}-\d{3,4}-\d{4}', '[TEL]', text)
    text = re.sub(r'\d{10,14}', '[ACC]', text)
    
    # 5. 기타 (URL, 4자리 이상의 연속 숫자)
    text = re.sub(r'http[s]?://\S+', '[URL]', text)
    text = re.sub(r'\d{4,}', '[NUM]', text)
    
    return text

def _stable_id_from_text(s: str) -> int:
    """
    keyword -> stable int64 id
    """
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16) & 0x7FFFFFFFFFFFFFFF


def _l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + eps)


RISK_SAFE = "NORMAL"
RISK_WARNING = "WARNING"
RISK_DANGER = "DANGER"

CATEGORY_KEYWORDS = [
    ("기관사칭", ["검찰", "검사", "검찰청", "경찰", "경찰청", "수사관", "금감원", "금융감독원", "국세청", "법원", "검거", "출석", "압수수색", "계좌동결"]),
    ("광고", ["광고", "대출", "저금리", "한도", "승인", "상담", "이자", "특별금리", "즉시", "신용", "캐피탈", "대부"]),
    ("투자사기", ["투자", "수익", "코인", "주식", "리딩", "상장", "상한가", "물타기", "수익률", "원금", "보장"]),
    ("채용빙자", ["채용", "면접", "입사", "지원", "이력서", "채용공고", "합격", "인사팀"]),
    ("납치협박", ["납치", "협박", "몸값", "구속", "체포", "구금", "감금"]),
    ("가족,지인사칭", ["가족", "지인", "친구", "어마", "아빠", "아들", "딸", "동생", "언니", "오빠", "형", "누나"]),
]

def _detect_category(text: str) -> str | None:
    if not text:
        return None
    for category, keywords in CATEGORY_KEYWORDS:
        if any(kw in text for kw in keywords):
            return category
    return None

# ----------------------------
# 3) FAISS 키워드 스토어
# ----------------------------
class FaissKeywordStore:
    """
    - vectorizer(TfidfVectorizer)로 키워드 임베딩
    - cosine similarity = normalize + inner product
    - index + meta를 파일로 저장/로드
    """
    def __init__(self, vec, dim: int, index_path: str | Path, meta_path: str | Path):
        self.vec = vec
        self.dim = int(dim)
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)

        self._lock = threading.RLock()

        self.id_to_kw: Dict[int, str] = {}
        self.kw_to_id: Dict[str, int] = {}

        self.index = self._create_empty_index()
        self.load()

    def _create_empty_index(self) -> faiss.Index:
        base = faiss.IndexFlatIP(self.dim)
        return faiss.IndexIDMap2(base)

    def _embed_keywords(self, keywords: List[str]) -> np.ndarray:
        cleaned = [advanced_deidentify(k) for k in keywords]
        x = self.vec.transform(cleaned).toarray().astype("float32")
        return _l2_normalize(x)

    def _embed_sentence(self, sentence: str) -> np.ndarray:
        cleaned = advanced_deidentify(sentence)
        x = self.vec.transform([cleaned]).toarray().astype("float32")
        return _l2_normalize(x)

    def load(self) -> None:
        with self._lock:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
            if self.meta_path.exists():
                data = json.loads(self.meta_path.read_text(encoding="utf-8"))
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
        keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
        with self._lock:
            self.index = self._create_empty_index()
            self.id_to_kw.clear()
            self.kw_to_id.clear()

            if not keywords:
                self.save()
                return 0

            ids = np.array([_stable_id_from_text(k) for k in keywords], dtype="int64")
            vecs = self._embed_keywords(keywords)
            self.index.add_with_ids(vecs, ids)

            for kw, _id in zip(keywords, ids.tolist()):
                self.id_to_kw[_id] = kw
                self.kw_to_id[kw] = _id

            self.save()
            return len(keywords)

    def upsert(self, keywords: List[str]) -> Dict[str, Any]:
        keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
        if not keywords:
            return {"added": 0, "updated": 0, "total": int(self.index.ntotal)}

        with self._lock:
            add_list: List[str] = []
            add_ids: List[int] = []
            updated = 0
            added = 0
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
            return {"added": added, "updated": updated, "total": int(self.index.ntotal)}

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
        if self.index.ntotal == 0:
            return []
        x = self._embed_sentence(sentence)
        with self._lock:
            sims, ids = self.index.search(x, topk)

        out: List[Dict[str, Any]] = []
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
# 4) Config
# ----------------------------
@dataclass
class TextInferConfig:
    device: str = "cpu"  # "cuda" or "cpu"
    ae_path: str = "assets/models/final_ae.pth"
    kobert_path: str = "assets/models/kobert"
    threshold: float = 5500.0
    buffer_size: int = 3

    # koBERT 후처리 설정
    temp: float = 5.0
    danger_low: float = 29.5
    danger_high: float = 31.5

    safe_words: Optional[List[str]] = None

    # FAISS 키워드 설정
    faiss_index_path: str = "assets/faiss/keyword.index"
    faiss_meta_path: str = "assets/faiss/keyword_meta.json"
    faiss_topk: int = 10
    faiss_min_sim: float = 0.25

    # 키워드가 잡힐 때 위험도 반영 룰
    keyword_warn_count: int = 1        # 키워드 1개만 잡혀도 WARN 최소 보장
    keyword_critical_count: int = 2    # 키워드 2개 이상이면 CRITICAL 쪽으로 강하게
    keyword_force_risk: float = 0.45   # AE가 SAFE여도 키워드 잡히면 risk_score 최소치
    keyword_bonus_per_hit: float = 0.15  # hit 하나마다 risk_score 가산(최대 1.0 클램프)


# ----------------------------
# 5) TextInfer
# ----------------------------
class TextInfer:
    """
    1) AE loss로 이상 여부 판단
    2) FAISS 키워드 히트로 위험 신호 반영
    3) 이상이면 최근 N개를 koBERT로 상세 분석
    4) status / loss / details / risk_score(0~1) 반환
    """

    def __init__(self, cfg: TextInferConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

        # ---- AE + Vectorizer 로드 ----
        ckpt = torch.load(cfg.ae_path, map_location=self.device, weights_only=False)
        self.input_dim: int = int(ckpt.get("input_dim", 8000))
        self.vectorizer = ckpt.get("vec")
        if self.vectorizer is None:
            raise RuntimeError("final_ae.pth checkpoint에 'vec'(TfidfVectorizer)가 없습니다.")

        self.ae = Autoencoder(self.input_dim).to(self.device)
        state = ckpt.get("state", None)
        if state is None:
            state = ckpt
        self.ae.load_state_dict(state, strict=False)
        self.ae.eval()

        # ---- koBERT 로드 ----
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.kobert_path, use_fast=False)
        self.bert = BertForSequenceClassification.from_pretrained(cfg.kobert_path).to(self.device)
        self.bert.eval()

        self.safe_words = cfg.safe_words or ["점심", "저녁", "먹자", "카페", "친구", "고생", "사랑해", "반가워"]

        # ---- FAISS 키워드 스토어 로드 ----
        self.kw_store = FaissKeywordStore(
            vec=self.vectorizer,
            dim=self.input_dim,
            index_path=cfg.faiss_index_path,
            meta_path=cfg.faiss_meta_path,
        )

    # ---- 키워드 관리 API (서버 라우터에서 그대로 호출 가능) ----
    def upsert_keywords(self, keywords: List[str]) -> Dict[str, Any]:
        return self.kw_store.upsert(keywords)

    def rebuild_keywords(self, keywords: List[str]) -> Dict[str, Any]:
        total = self.kw_store.rebuild(keywords)
        return {"total": total}

    def remove_keywords(self, keywords: List[str]) -> Dict[str, Any]:
        removed = self.kw_store.remove(keywords)
        return {"removed": removed}

    def faiss_stats(self) -> Dict[str, Any]:
        return {
            "total": int(self.kw_store.index.ntotal),
            "meta_keywords": len(self.kw_store.kw_to_id),
            "index_path": str(self.kw_store.index_path),
            "meta_path": str(self.kw_store.meta_path),
        }

    # ---- AE / BERT ----
    def ae_loss(self, text: str) -> float:
        vec = self.vectorizer.transform([advanced_deidentify(text)]).toarray()
        x = torch.FloatTensor(vec).to(self.device)
        with torch.no_grad():
            recon = self.ae(x)
            loss = torch.mean((recon - x) ** 2).item()
        return float(loss)

    def bert_analyze(self, text: str) -> Dict[str, Any]:
        category = _detect_category(text)
        category_label = category or ""
        if any(w in text for w in self.safe_words):
            return {"result": category_label, "risk_label": RISK_SAFE, "category": category_label, "prob": 0.0, "msg": "일상 대화 필터링"}

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            logits = self.bert(**inputs).logits
            probs = F.softmax(logits / self.cfg.temp, dim=-1)[0].detach().cpu().numpy()

        prob = round(float(probs[1] * 100), 2)  # 0~100
        if self.cfg.danger_low <= prob <= self.cfg.danger_high:
            risk_label = RISK_DANGER
            msg = "피싱 패턴 감지"
        elif 28.0 <= prob < self.cfg.danger_low:
            risk_label = RISK_WARNING
            msg = "의심 정황 포착"
        else:
            risk_label = RISK_SAFE
            msg = "정상 문맥"
        return {"result": category_label, "risk_label": risk_label, "category": category_label, "prob": prob, "msg": msg}

    # ---- FAISS 키워드 검색 ----
    def faiss_keywords(self, sentence: str) -> List[Dict[str, Any]]:
        if not sentence:
            return []
        return self.kw_store.search(sentence, topk=self.cfg.faiss_topk, min_sim=self.cfg.faiss_min_sim)

    # ----------------------------
    # 6) 최종 예측
    # ----------------------------
    def predict(self, buffered_texts: List[str]) -> Dict[str, Any]:
        """
        buffered_texts: call_id 기준 최근 N개(예: 3개) 텍스트
        """
        if not buffered_texts:
            return {
                "status": "SAFE",
                "loss": 0.0,
                "details": None,
                "risk_score": 0.0,
                "keywords": [],
                "faiss_hits": [],
            }

        # 키워드는 "최근 chunk" + "전체 합친 문장" 둘 다 검색
        latest = buffered_texts[-1]
        merged = " ".join([t for t in buffered_texts if isinstance(t, str)])
        # Bias toward SAFE if safe words are present.
        safe_present = False
        if self.safe_words:
            safe_present = any(w in merged for w in self.safe_words)

        warn_threshold = self.cfg.keyword_warn_count + (1 if safe_present else 0)
        min_chunks = self.cfg.buffer_size

        hits_latest = self.faiss_keywords(latest)
        hits_merged = self.faiss_keywords(merged)

        # 중복 제거(키워드 기준)
        seen = set()
        faiss_hits: List[Dict[str, Any]] = []
        for h in (hits_latest + hits_merged):
            kw = h.get("keyword")
            if not kw or kw in seen:
                continue
            seen.add(kw)
            faiss_hits.append(h)

        detected_kw = [h["keyword"] for h in faiss_hits]
        kw_count = len(detected_kw)

        # ---- AE 판단 ----
        loss = self.ae_loss(latest)
        ae_suspicious = loss > self.cfg.threshold

        # ---- 키워드 기반 위험도 보정 ----
        # AE가 SAFE라도 키워드가 잡히면 risk_score를 최소/가산 처리
        keyword_risk = 0.0
        if kw_count >= warn_threshold:
            keyword_risk = max(keyword_risk, self.cfg.keyword_force_risk)
            keyword_risk = min(1.0, keyword_risk + (kw_count * self.cfg.keyword_bonus_per_hit))
        if safe_present:
            keyword_risk *= 0.3

        # ---- AE가 SAFE이고, 키워드도 없으면 바로 SAFE ----
        if (not ae_suspicious) and kw_count < warn_threshold:
            return {
                "status": "SAFE",
                "loss": loss,
                "details": None,
                "risk_score": 0.0,
                "keywords": [],
                "faiss_hits": [],
            }

        if len(buffered_texts) < min_chunks:
            return {
                "status": "NORMAL",
                "loss": loss,
                "details": None,
                "risk_score": 0.0,
                "keywords": detected_kw,
                "faiss_hits": faiss_hits,
            }

        # ---- 상세 분석(koBERT): AE가 의심이거나, 키워드가 일정 이상이면 분석 ----
        run_bert = ae_suspicious or (kw_count >= warn_threshold)

        details = None
        bert_risk = 0.0
        status = "NORMAL"  # 기본은 NORMAL로 시작

        if run_bert:
            details = [self.bert_analyze(t) for t in buffered_texts]
            window = details[-min_chunks:]
            dangers = [d for d in window if d.get("risk_label", d.get("result")) == RISK_DANGER]
            warnings = [d for d in window if d.get("risk_label", d.get("result")) == RISK_WARNING]

            # bert risk_score (0~1)
            max_prob = 0.0
            for d in details:
                max_prob = max(max_prob, float(d.get("prob", 0.0)))
            bert_risk = max_prob / 100.0

            # 상태 결정(최근 N개 기준)
            if len(dangers) >= 1:
                status = "CRITICAL"
            elif len(warnings) >= 2:
                status = "WARNING"
            else:
                status = "NORMAL"

        # 최종 risk_score = max(bert_risk, keyword_risk) (클램프)
        risk_score = float(min(1.0, max(bert_risk, keyword_risk)))
        print(f"status={status} risk_score={risk_score} keywords={detected_kw} faiss_hits ={faiss_hits} ")
        return {
            "status": status,
            "loss": loss,
            "details": details,
            "risk_score": risk_score,
            "keywords": detected_kw, # 탐지키워드 나오죵 이거 전달하면 됨
            "faiss_hits": faiss_hits, # 추가로 문장마다 추정해야하는데 
        }
