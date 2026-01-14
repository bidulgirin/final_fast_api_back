# app/services/mel_best_infer.py

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import librosa


@dataclass
class MelInferConfig:
    device: str = "cpu"

    # ✅ 문서 기준 (inference는 sr=22050로 고정해서 처리)
    target_sample_rate: int = 22050
    duration_sec: int = 3

    # ✅ 문서 기준
    n_mels: int = 224
    hop_length: int = 512

    # ✅ 문서 기준 threshold 예시
    threshold: float = 0.6

    # ✅ 문서 기준 (224, 224)
    img_size: int = 224

    # 서버 입력이 16k PCM일 때만 필요 (문서에는 없음. 하지만 서버에선 보통 필요)
    input_sample_rate: Optional[int] = None  # None이면 리샘플 안 함


class MelBestInfer:
    """
    문서 predict_back_to_basics 전처리와 동일하게 맞춘 infer 클래스.

    ✅ 동일 포인트
    - librosa.feature.melspectrogram(y, sr=22050, n_mels=224, hop_length=512) (기타 파라미터 미지정)
    - S_dB = librosa.power_to_db(S, ref=np.max)
    - torch.from_numpy(S_dB).float().unsqueeze(0).repeat(3, 1, 1)
    - transforms.Resize((224, 224))
    - softmax 후 deepvoice 확률
    - 이미지(PIL) 사용 X
    - 0~1 정규화 X
    """

    def __init__(self, model_path: str, cfg: MelInferConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        self.model.eval()

        self.resize = transforms.Resize((cfg.img_size, cfg.img_size))
        self.classes = {0: "normal", 1: "deepvoice"}

    # ---------- 입력 처리 유틸 ----------

    def _pcm_i16_to_float32(self, audio_i16: np.ndarray) -> np.ndarray:
        """int16 PCM -> float32 파형"""
        if audio_i16.dtype != np.int16:
            audio_i16 = audio_i16.astype(np.int16)
        y = audio_i16.astype(np.float32) / 32768.0
        return y

    def _fix_length(self, y: np.ndarray) -> np.ndarray:
        """문서 기준: 3초 길이로 자르거나 padding"""
        fixed_length = self.cfg.target_sample_rate * self.cfg.duration_sec
        if len(y) > fixed_length:
            return y[:fixed_length]
        if len(y) < fixed_length:
            return np.pad(y, (0, fixed_length - len(y)))
        return y

    def _resample_if_needed(self, y: np.ndarray) -> np.ndarray:
        """서버 입력 SR이 따로 있을 때만 22050으로 변환"""
        if not self.cfg.input_sample_rate:
            return y
        if self.cfg.input_sample_rate == self.cfg.target_sample_rate:
            return y

        return librosa.resample(
            y,
            orig_sr=self.cfg.input_sample_rate,
            target_sr=self.cfg.target_sample_rate,
        )

    # ---------- 문서와 동일한 전처리 ----------

    def _to_input_tensor_doc_style(self, y: np.ndarray) -> torch.Tensor:
        """
        문서 Step3~4 동일:
          S = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=224, hop_length=512)
          S_dB = librosa.power_to_db(S, ref=np.max)
          tensor_data = torch.from_numpy(S_dB).float().unsqueeze(0).repeat(3, 1, 1)
          input_tensor = transforms.Resize((224,224))(tensor_data).unsqueeze(0)
        """
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.cfg.target_sample_rate,
            n_mels=self.cfg.n_mels,
            hop_length=self.cfg.hop_length,
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        tensor_data = torch.from_numpy(S_dB).float().unsqueeze(0).repeat(3, 1, 1)  # (3, n_mels, time)
        input_tensor = self.resize(tensor_data).unsqueeze(0).to(self.device)      # (1, 3, 224, 224)
        return input_tensor

    # ---------- 외부에서 호출하는 API ----------

    def predict_from_pcm_i16(self, audio_i16: np.ndarray, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        서버에서 PCM(int16) numpy로 받는 경우
        """
        if audio_i16 is None or audio_i16.size == 0:
            raise ValueError("Empty PCM array")

        y = self._pcm_i16_to_float32(audio_i16)
        y = self._resample_if_needed(y)
        y = self._fix_length(y)

        x = self._to_input_tensor_doc_style(y)

        with torch.no_grad():
            outputs = self.model(x)
            probs = F.softmax(outputs, dim=1)  # (1,2)

        normal_prob = float(probs[0][0].item())
        deep_prob = float(probs[0][1].item())

        th = self.cfg.threshold if threshold is None else float(threshold)
        final_label = "deepvoice" if deep_prob >= th else "normal"

        return {
            "threshold_class": final_label,
            "pred_class": self.classes[int(torch.argmax(probs, dim=1).item())],
            "phishing_score": deep_prob,
            "probs": {"normal": normal_prob, "deepvoice": deep_prob},
            "meta": {
                "target_sr": self.cfg.target_sample_rate,
                "duration_sec": self.cfg.duration_sec,
                "n_mels": self.cfg.n_mels,
                "hop_length": self.cfg.hop_length,
                "threshold": th,
            },
        }

    def predict_from_pcm_bytes(self, pcm_bytes: bytes, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        서버에서 보통 bytes로 들어오는 경우 (안드로이드 PCM)
        - 16-bit little-endian mono PCM 가정
        """
        if pcm_bytes is None or len(pcm_bytes) == 0:
            raise ValueError("Empty PCM bytes")

        # little-endian int16
        audio_i16 = np.frombuffer(pcm_bytes, dtype="<i2")
        return self.predict_from_pcm_i16(audio_i16, threshold=threshold)
