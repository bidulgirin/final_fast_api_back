# app/db/models/phising_sign.py

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn


# 1) 모델 구조 정의 (ckpt에 저장된 가중치와 동일해야 함)
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


# 2) 전처리 함수
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


class PhishingDetectorAE:
    """
    - ckpt 포맷 가정:
      {
        'vec': TfidfVectorizer,
        'input_dim': int,
        'state': model_state_dict
      }
    - keywords 는 외부(Faiss)에서 뽑아와서 주입하는 방식 권장
    """

    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # torch 버전에 따라 weights_only 인자가 없을 수 있어 방어
        try:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(self.model_path, map_location=self.device)

        self.vec = ckpt["vec"]
        self.input_dim = int(ckpt["input_dim"])
        self.model = PhishingFilterAE(self.input_dim).to(self.device)
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()

    def _score(self, sentence: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        cleaned = advanced_deidentify(sentence)
        x_vec = self.vec.transform([cleaned]).toarray()
        x_tensor = torch.FloatTensor(x_vec).to(self.device)

        with torch.no_grad():
            pred = self.model(x_tensor)
            base_loss = torch.abs(pred - x_tensor).sum().item()

        # keywords (Faiss 등으로 외부에서 주입)
        detected_kw = keywords or  [
        # [기관 및 직함]
        "수사관", "검찰", "검사", "지검", "중앙지검", "금감원", "금융감독원", "경찰", "형사", "법원",
        
        # [범죄 상황 유도]
        "명의도용", "대포통장", "사건번호", "연루", "범죄단체", "불법자금", "자산보호", "협조",
        
        # [금전 및 금융 요구]
        "송금", "이체", "예치", "현금", "계좌", "비밀번호", "인증번호", "카드번호",
        
        # [금융 사기 - 대출/정부지원]
        "저금리", "대환", "정부지원", "상환", "한도", "채무", "특별공급",
        
        # [메신저 피싱 - 지인 사칭]
        "액정", "수리비", "급해", "도와줘", "문화상품권", "기프티콘", "구글플레이", "기프트카드"
    ]
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
            "cleaned": cleaned,  # 디버깅용 (원치 않으면 제거)
        }

    def predict(self, sentence: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        if not sentence or not isinstance(sentence, str):
            return {"result": "✅ 정상", "score": 0.0, "keywords": []}
        return self._score(sentence, keywords=keywords)


# 싱글톤 로드 (서버 뜰 때 1번만)
# - 경로는 프로젝트 루트 기준 "assets/models/final_ae.pth" 라고 하셨으니 그대로 둠
DEFAULT_AE_PATH = Path("assets/models/final_ae.pth")
ae_detector = PhishingDetectorAE(DEFAULT_AE_PATH)
