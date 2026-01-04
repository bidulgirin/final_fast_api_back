# app/services/mfcc_infer.py
# app/services/mfcc_infer.py (state_dict 로딩 + MFCC 전처리 + 추론)
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from app.models.mfcc_model import MFCCBestModel

@dataclass
class MFCCInferConfig:
    sr: int = 16000
    n_mfcc: int = 40
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    center: bool = True
    target_len: int = 501
    device: str = "cpu"

class MFCCInfer:
    def __init__(self, model_path: str, cfg: MFCCInferConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # ✅ torchaudio MFCC transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=cfg.sr,
            n_mfcc=cfg.n_mfcc,
            melkwargs={
                "n_fft": cfg.n_fft,
                "hop_length": cfg.hop_length,
                "n_mels": cfg.n_mels,
                "center": cfg.center,
            }
        ).to(self.device)

        self.model = self._load_model(model_path).to(self.device).eval()

    def _load_model(self, model_path: str) -> nn.Module:
        """
        ✅ mfcc_best_model.pt가 OrderedDict(state_dict) 형태인 상황을 처리.
        """
        sd = torch.load(model_path, map_location="cpu")
        if not isinstance(sd, dict):
            raise ValueError(f"Expected state_dict dict/OrderedDict, got {type(sd)}")

        # DataParallel로 저장된 경우 module. prefix 제거
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

        # checkpoint에서 conv / fc weight로 구조 파라미터 추출
        conv1_w = sd["cnn.0.weight"]
        conv2_w = sd["cnn.4.weight"]

        fc_w = sd["fc.weight"]  # shape: (out, in)
        fc_out, fc_in = fc_w.shape

        model = MFCCBestModel(conv1_w=conv1_w, conv2_w=conv2_w, fc_in=fc_in, fc_out=fc_out)
        model.load_state_dict(sd, strict=True)  # ✅ 구조가 맞으면 여기서 통과
        return model

    @torch.inference_mode()
    def predict_from_pcm_i16(self, audio_i16: np.ndarray) -> Dict[str, Any]:
        """
        audio_i16: np.int16 mono, 16kHz, 5초 권장(80000 samples)
        return: phishing_score 등
        """
        x, lengths, raw_T = self._pcm_to_model_input(audio_i16)  # x: (1, target_len, 40)

        logits = self.model(x, lengths)          # (1,)
        prob = torch.sigmoid(logits)[0].item()   # 0~1

        return {
            "phishing_score": float(prob),
            "logits": float(logits[0].item()),
            "raw_T": int(raw_T),
            "used_len": int(lengths[0].item()),
            "input_shape": tuple(x.shape),
        }

    def _pcm_to_model_input(self, audio_i16: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if audio_i16.dtype != np.int16:
            audio_i16 = audio_i16.astype(np.int16, copy=False)
        audio_i16 = audio_i16.reshape(-1)

        # PCM16 -> float waveform (-1~1)
        wav = (audio_i16.astype(np.float32) / 32768.0)
        wav = torch.from_numpy(wav).to(self.device).unsqueeze(0)  # (1, N)

        # MFCC: (1, 40, T)
        mfcc = self.mfcc_transform(wav)
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # (T, 40)
        raw_T = mfcc.shape[0]

        # 길이 고정
        target_len = self.cfg.target_len
        length = min(raw_T, target_len)

        if raw_T >= target_len:
            mfcc = mfcc[:target_len]
        else:
            mfcc = F.pad(mfcc, (0, 0, 0, target_len - raw_T))

        x = mfcc.unsqueeze(0)  # (1, target_len, 40)
        lengths = torch.tensor([length], device=self.device, dtype=torch.long)
        return x, lengths, raw_T
