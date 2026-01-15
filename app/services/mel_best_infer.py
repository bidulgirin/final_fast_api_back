# app/services/mel_best_infer.py

from dataclasses import dataclass
from typing import Dict, Any, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import librosa

# state_dict 로드하려면 모델을 다시 만들어야 하므로 timm 필요
import timm


@dataclass
class MelInferConfig:
    device: str = "cpu"
    target_sample_rate: int = 22050
    duration_sec: int = 5

    n_mels: int = 224
    hop_length: int = 512
    threshold: float = 0.6
    img_size: int = 224

    # 모델 구조 정보 (timm create_model용)
    model_name: str = "res2net50_26w_4s"
    num_classes: int = 2

    # 서버 입력 PCM 샘플레이트
    input_sample_rate: Optional[int] = None  # None이면 리샘플 안 함


class MelBestInfer:
    """
    노트북 predict_back_to_basics 전처리와 동일하게 맞춘 infer 클래스.

    추가로:
    - torch.save(model) 로 저장된 파일(nn.Module)과
    - torch.save(model.state_dict()) 로 저장된 파일(OrderedDict)
      둘 다 로드 가능하게 처리
    """

    def __init__(self, model_path: str, cfg: MelInferConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        self.model = self._build_and_load_model(ckpt)
        self.model.to(self.device)
        self.model.eval()

        self.resize = transforms.Resize((cfg.img_size, cfg.img_size))
        self.classes = {0: "normal", 1: "deepvoice"}

    # ---------- 모델 로드(핵심 수정) ----------

    def _build_and_load_model(self, ckpt_obj) -> nn.Module:
        # 1) 전체 모델로 저장된 경우
        if isinstance(ckpt_obj, nn.Module):
            return ckpt_obj

        # 2) state_dict / 체크포인트(dict)로 저장된 경우
        if isinstance(ckpt_obj, (dict, OrderedDict)):
            state_dict = None

            # (a) ckpt_obj 자체가 state_dict인 경우: 키에 '.'가 많은 편
            if all(isinstance(k, str) for k in ckpt_obj.keys()) and any("." in k for k in ckpt_obj.keys()):
                state_dict = ckpt_obj
            else:
                # (b) 체크포인트 안에서 흔한 키들 탐색
                for key in ("state_dict", "model_state_dict", "model", "net"):
                    if key in ckpt_obj and isinstance(ckpt_obj[key], (dict, OrderedDict)):
                        state_dict = ckpt_obj[key]
                        break

            if state_dict is None:
                raise ValueError(
                    "Checkpoint dict에서 state_dict를 찾지 못했습니다. "
                    "키 목록/저장 방식을 확인하세요."
                )

            # DDP 학습 등으로 'module.' prefix가 붙어있으면 제거
            cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}

            # timm 모델을 동일 구조로 생성 후 로드
            model = timm.create_model(
                self.cfg.model_name,
                pretrained=False,          # 서버에서 다운로드 방지
                num_classes=self.cfg.num_classes,
            )

            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            # strict=False로 두고 로그로만 확인 (필요하면 strict=True로 바꿀 수 있음)
            if missing or unexpected:
                print(f"[MelBestInfer] load_state_dict warnings:")
                if missing:
                    print(f"  - missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
                if unexpected:
                    print(f"  - unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

            return model

        raise TypeError(f"지원하지 않는 체크포인트 타입: {type(ckpt_obj)}")

    # ---------- 입력 처리 유틸 ----------

    def _pcm_i16_to_float32(self, audio_i16: np.ndarray) -> np.ndarray:
        if audio_i16.dtype != np.int16:
            audio_i16 = audio_i16.astype(np.int16)
        return audio_i16.astype(np.float32) / 32768.0

    def _fix_length(self, y: np.ndarray) -> np.ndarray:
        fixed_length = self.cfg.target_sample_rate * self.cfg.duration_sec
        if len(y) > fixed_length:
            return y[:fixed_length]
        if len(y) < fixed_length:
            return np.pad(y, (0, fixed_length - len(y)))
        return y

    def _resample_if_needed(self, y: np.ndarray) -> np.ndarray:
        if not self.cfg.input_sample_rate:
            return y
        if self.cfg.input_sample_rate == self.cfg.target_sample_rate:
            return y
        return librosa.resample(y, orig_sr=self.cfg.input_sample_rate, target_sr=self.cfg.target_sample_rate)

    # ---------- 노트북과 동일 전처리 ----------

    def _to_input_tensor_doc_style(self, y: np.ndarray) -> torch.Tensor:
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

    # ---------- 외부 API ----------

    def predict_from_pcm_i16(self, audio_i16: np.ndarray, threshold: Optional[float] = None) -> Dict[str, Any]:
        if audio_i16 is None or audio_i16.size == 0:
            raise ValueError("Empty PCM array")

        y = self._pcm_i16_to_float32(audio_i16)
        y = self._resample_if_needed(y)
        y = self._fix_length(y)

        x = self._to_input_tensor_doc_style(y)

        with torch.no_grad():
            outputs = self.model(x)
            probs = F.softmax(outputs, dim=1)

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
                "model_name": self.cfg.model_name,
            },
        }

    def predict_from_pcm_bytes(self, pcm_bytes: bytes, threshold: Optional[float] = None) -> Dict[str, Any]:
        if pcm_bytes is None or len(pcm_bytes) == 0:
            raise ValueError("Empty PCM bytes")

        audio_i16 = np.frombuffer(pcm_bytes, dtype="<i2")  # little-endian int16
        return self.predict_from_pcm_i16(audio_i16, threshold=threshold)
