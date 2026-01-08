# app/services/stt_infer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError as e:
    raise ImportError(
        "faster-whisper가 필요합니다. `pip install faster-whisper`로 설치하세요."
    ) from e


@dataclass
class STTInferConfig:
    model_size: str = "tiny"         # tiny / base / small / medium / large-v3 등
    device: str = "cpu"               # "cpu" or "cuda"
    compute_type: str = "int8"        # cpu면 int8 권장, cuda면 float16 가능
    language: str = "ko"              # 한국어 고정이면 "ko"
    vad_filter: bool = True           # 무음 구간 제거로 품질/속도 도움
    beam_size: int = 1                # 속도 우선이면 1~2
    best_of: int = 1


class STTInfer:
    def __init__(self, cfg: STTInferConfig):
        self.cfg = cfg
        self.model = WhisperModel(
            cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )

    def transcribe_from_pcm_i16(self, audio_i16: np.ndarray, sample_rate: int = 16000) -> str:
        """
        audio_i16: np.int16 1D PCM (mono)
        return: transcription text
        """
        if audio_i16 is None or audio_i16.size == 0:
            return ""

        # int16 -> float32 [-1, 1]
        audio_f32 = (audio_i16.astype(np.float32) / 32768.0)

        segments, _info = self.model.transcribe(
            audio_f32,
            language=self.cfg.language,
            vad_filter=self.cfg.vad_filter,
            beam_size=self.cfg.beam_size,
            best_of=self.cfg.best_of,
        )

        texts = []
        for seg in segments:
            if seg.text:
                texts.append(seg.text.strip())

        return " ".join([t for t in texts if t]).strip()
