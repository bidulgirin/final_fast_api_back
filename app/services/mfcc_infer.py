# app/services/mfcc_infer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class CNNBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)   # (B, 1, 40, 500)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(1)
    
    
@dataclass
class MFCCInferConfig:
    sr: int = 16000
    seconds: int = 5

    n_mfcc: int = 40
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    center: bool = True

    target_frames: int = 500  # 학습에서 MAX_LEN=500으로 잘랐음
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

    def _load_model(self, model_path: str) -> nn.Module:
        sd = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
        if not isinstance(sd, dict):
            raise ValueError(f"Expected state_dict(dict/OrderedDict), got {type(sd)}")

        # DataParallel 저장이면 "module." 제거
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

        model = CNNBinary()
        model.load_state_dict(sd, strict=True)
        return model

    @torch.inference_mode()
    def predict_from_pcm_i16(self, audio_i16: np.ndarray) -> Dict[str, Any]:
        x, raw_T = self._pcm_to_model_input(audio_i16)  # x: (1, 40, 500)

        logits = self.model(x)                 # (1,)
        prob = torch.sigmoid(logits)[0].item() # 0~1

        return {
            "phishing_score": float(prob),
            "logits": float(logits[0].item()),
            "raw_T": int(raw_T),               # MFCC 원래 프레임 수(보통 501)
            "input_shape": tuple(x.shape),
        }

    def _pcm_to_model_input(self, audio_i16: np.ndarray) -> Tuple[torch.Tensor, int]:
        if audio_i16.dtype != np.int16:
            audio_i16 = audio_i16.astype(np.int16, copy=False)
        audio_i16 = audio_i16.reshape(-1)

        # PCM16 -> float waveform (-1~1)
        wav = audio_i16.astype(np.float32) / 32768.0

        # 학습과 동일하게 5초(80000)로 pad/trunc
        if wav.shape[0] > self.max_samples:
            wav = wav[: self.max_samples]
        elif wav.shape[0] < self.max_samples:
            wav = np.pad(wav, (0, self.max_samples - wav.shape[0]), mode="constant")

        wav = torch.from_numpy(wav).to(self.device).unsqueeze(0)  # (1, N)

        # MFCC: (1, 40, T)
        mfcc = self.mfcc_transform(wav)  # (1, 40, T)
        mfcc = mfcc.squeeze(0)           # (40, T)
        raw_T = mfcc.shape[1]

        # 학습과 동일하게 T를 500으로 고정 (501 -> 500 잘림)
        target = self.cfg.target_frames
        if raw_T > target:
            mfcc = mfcc[:, :target]
        elif raw_T < target:
            mfcc = F.pad(mfcc, (0, target - raw_T))

        x = mfcc.unsqueeze(0).float()    # (1, 40, 500)
        return x, raw_T
