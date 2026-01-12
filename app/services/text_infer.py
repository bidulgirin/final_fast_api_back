# app/services/text_infer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.feature_extraction.text  # noqa: F401
from transformers import BertForSequenceClassification, BertTokenizer, AutoTokenizer


# torch load 시 vectorizer(TfidfVectorizer) 안전 글로벌 등록
torch.serialization.add_safe_globals([sklearn.feature_extraction.text.TfidfVectorizer])


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


@dataclass
class TextInferConfig:
    device: str = "mps"  # "mps"
    ae_path: str = "assets/models/final_ae.pth"
    kobert_path: str = "assets/models/kobert"  # 로컬 디렉토리 or HF id
    threshold: float = 5500.0
    buffer_size: int = 3

    # koBERT 후처리 설정
    temp: float = 5.0
    danger_low: float = 29.5
    danger_high: float = 31.5

    safe_words: Optional[List[str]] = None


class TextInfer:
    """
    1) AE loss로 이상 여부 판단
    2) 이상이면 call_id 기준 최근 N개 텍스트를 koBERT로 상세 분석
    3) status / loss / details / risk_score(0~1) 반환
    """
    def __init__(self, cfg: TextInferConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if (cfg.device == "mps" and torch.mps.is_available()) else "cpu")

        # ---- AE + Vectorizer 로드 ----
        ckpt = torch.load(cfg.ae_path, map_location=self.device, weights_only=False)
        self.input_dim: int = int(ckpt.get("input_dim", 8000))
        self.vectorizer = ckpt.get("vec")
        if self.vectorizer is None:
            raise RuntimeError("final_ae.pt checkpoint에 'vec'(TfidfVectorizer)가 없습니다.")

        self.ae = Autoencoder(self.input_dim).to(self.device)
        state = ckpt.get("state", None)
        if state is None:
            # 혹시 state_dict만 저장된 케이스
            state = ckpt
        self.ae.load_state_dict(state, strict=False)
        self.ae.eval()

        # ---- koBERT 로드 ----
        # 토크나이저는 monologg/kobert를 쓰는 패턴이 많아서 기본은 HF id,
        # 로컬에 저장된 경우 kobert_path가 디렉토리여도 동작합니다.
        # self.tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.kobert_path, use_fast=False)
        self.bert = BertForSequenceClassification.from_pretrained(cfg.kobert_path).to(self.device)
        self.bert.eval()

        self.safe_words = cfg.safe_words or ["점심", "저녁", "먹자", "카페", "친구", "고생", "사랑해", "반가워"]

    def ae_loss(self, text: str) -> float:
        vec = self.vectorizer.transform([text]).toarray()
        x = torch.FloatTensor(vec).to(self.device)
        with torch.no_grad():
            recon = self.ae(x)
            loss = torch.mean((recon - x) ** 2).item()
        return float(loss)

    def bert_analyze(self, text: str) -> Dict[str, Any]:
        # 일상 대화 필터
        if any(w in text for w in self.safe_words):
            return {"result": "✅ 안전", "prob": 0.0, "msg": "일상 대화 필터링"}

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
            return {"result": "🚨 위험", "prob": prob, "msg": "피싱 패턴 감지"}
        elif 28.0 <= prob < self.cfg.danger_low:
            return {"result": "🟠 경고", "prob": prob, "msg": "의심 정황 포착"}
        return {"result": "✅ 안전", "prob": prob, "msg": "정상 문맥"}

    def predict(self, buffered_texts: List[str]) -> Dict[str, Any]:
        """
        buffered_texts: call_id 기준 최근 N개(예: 3개) 텍스트
        """
        if not buffered_texts:
            return {"status": "SAFE", "loss": 0.0, "details": None, "risk_score": 0.0}

        # AE는 "가장 최근 chunk" 기준으로 이상 여부 판단 (원하시면 합쳐서도 가능)
        latest = buffered_texts[-1]
        loss = self.ae_loss(latest)

        if loss <= self.cfg.threshold:
            return {"status": "SAFE", "loss": loss, "details": None, "risk_score": 0.0}

        # threshold 넘으면 최근 N개를 koBERT로 분석
        details = [self.bert_analyze(t) for t in buffered_texts]
        dangers = [d for d in details if d["result"] == "🚨 위험"]
        warnings = [d for d in details if d["result"] == "🟠 경고"]

        if len(dangers) >= 1 or len(warnings) >= 2:
            status = "🚨 CRITICAL"
        else:
            status = "✅ NORMAL"

        # risk_score (0~1) 만들기: details 중 최고 prob 사용(0~100 -> 0~1)
        max_prob = 0.0
        for d in details:
            max_prob = max(max_prob, float(d.get("prob", 0.0)))
        risk_score = max_prob / 100.0

        return {"status": status, "loss": loss, "details": details, "risk_score": risk_score}
