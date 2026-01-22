# app/services/mfcc_infer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from app.models.mfcc_model import Res2Net50SE


@dataclass
class MFCCInferConfig:
    sr: int = 16000
    seconds: int = 5

    n_mfcc: int = 40
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160

    # Res2Net 샘플 코드 기준이면 False (5초 -> T=498)
    # CNNBinary 학습 파이프라인이 center=True였다면 그건 CNNBinary에만 맞던 설정일 가능성 큼
    center: bool = False

    # 프레임을 강제로 맞출지 옵션으로 둠
    # - None: MFCC 나온 그대로 사용 (보통 498)
    # - 500: 예전처럼 pad/trunc로 500 고정
    target_frames: Optional[int] = 498

    device: str = "cpu"


class MFCCInfer:
    def __init__(self, model_path: str, cfg: MFCCInferConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.max_samples = cfg.sr * cfg.seconds  # 80000

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=cfg.sr,
            n_mfcc=cfg.n_mfcc,
            melkwargs={
                "n_fft": cfg.n_fft,
                "hop_length": cfg.hop_length,
                "n_mels": cfg.n_mels,
                "center": cfg.center,
            },
        ).to(self.device)

        self.model = self._load_model(model_path).to(self.device).eval()

    def _unwrap_state_dict(self, ckpt: Any) -> Dict[str, torch.Tensor]:
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            ckpt = ckpt["state_dict"]

        if not isinstance(ckpt, dict):
            raise ValueError(f"Expected state_dict(dict), got {type(ckpt)}")

        if any(k.startswith("module.") for k in ckpt.keys()):
            ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

        return ckpt

    def _load_model(self, model_path: str) -> nn.Module:
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        sd = self._unwrap_state_dict(ckpt)

        model = Res2Net50SE()
        model.load_state_dict(sd, strict=True)
        return model

    @torch.inference_mode()
    def predict_from_pcm_i16(self, audio_i16: np.ndarray) -> Dict[str, Any]:
        x, raw_T = self._pcm_to_model_input(audio_i16)  # x: (1, 40, T)

        out = self.model(x)  # 보통 (1,2)

        if out.dim() == 2 and out.size(1) == 2:
            probs = F.softmax(out, dim=1)
            spoof_prob = probs[0, 1].item()
            logit = (out[0, 1] - out[0, 0]).item()
        else:
            # 혹시 단일 logit 모델이면 fallback
            logit = float(out[0].item())
            spoof_prob = float(torch.sigmoid(out)[0].item())

         # threshold 적용: 넘으면 1, 아니면 0
        threshold = getattr(self.cfg, "threshold", 0.2893)
        phishing_score = 1 if spoof_prob >= threshold else 0

        return {
            #"phishing_score": int(phishing_score),   # 0 or 1
            "phishing_score": spoof_prob,   # 예측값
            "spoof_prob": float(spoof_prob),         # (원하면 디버깅/로깅용으로 유지)
            "threshold": float(threshold),
            "logits": float(logit),
            "raw_T": int(raw_T),
            "input_shape": tuple(x.shape),
            "center": bool(self.cfg.center),
            "target_frames": self.cfg.target_frames,
        }

    def _pcm_to_model_input(self, audio_i16: np.ndarray) -> Tuple[torch.Tensor, int]:
        if audio_i16.dtype != np.int16:
            audio_i16 = audio_i16.astype(np.int16, copy=False)
        audio_i16 = audio_i16.reshape(-1)

        # PCM16 -> float waveform (-1~1)
        wav = audio_i16.astype(np.float32) / 32768.0

        # 2. 음량 정규화 추가 (이게 소리가 작아서 발생하는 오탐을 잡아줍니다)
        max_val = np.abs(wav).max()
        if max_val > 1e-6:
            wav = wav / max_val * 0.9

        # 5초(80000) pad/trunc
        if wav.shape[0] > self.max_samples:
            wav = wav[: self.max_samples]
        elif wav.shape[0] < self.max_samples:
            wav = np.pad(wav, (0, self.max_samples - wav.shape[0]), mode="constant")

        wav_t = torch.from_numpy(wav).to(self.device).unsqueeze(0)  # (1, N)

        # MFCC: (1, 40, T)
        mfcc = self.mfcc_transform(wav_t)  # (1, 40, T)
        mfcc = mfcc.squeeze(0)             # (40, T)
        raw_T = mfcc.shape[1]

        # 필요하면 T 고정
        target = self.cfg.target_frames
        if target is not None:
            if raw_T > target:
                mfcc = mfcc[:, :target]
            elif raw_T < target:
                mfcc = F.pad(mfcc, (0, target - raw_T))

        x = mfcc.unsqueeze(0).float()      # (1, 40, T)
        return x, raw_T