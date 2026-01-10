# app/services/stt_infer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os
import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError as e:
    raise ImportError("faster-whisper가 필요합니다. `pip install faster-whisper`로 설치하세요.") from e


@dataclass
class STTInferConfig:
    # 1060 3GB면 medium도 빡셀 수 있음. 운영은 small 권장, 최대치로 medium.
    model_size: str = "small"         # tiny / base / small / medium
    device: str = "cuda"              # "cpu" or "cuda"
    compute_type: str = "int8"        # 1060 3GB: cuda에서도 int8이 가장 안전

    language: str = "ko"

    # 5초 청크면 VAD는 켜면 이득도 있지만, 짧은 청크에서는 오히려 결과/지연이 흔들릴 때가 있어 기본 False 권장
    vad_filter: bool = False

    # 속도 우선
    beam_size: int = 1
    best_of: int = 1
    temperature: float = 0.0

    # 5초 chunk 독립 처리
    condition_on_previous_text: bool = False
    word_timestamps: bool = False

    # CPU 전용 튜닝(필요 시)
    cpu_threads: int = max(1, (os.cpu_count() or 4) - 1)
    num_workers: int = 1              # 1 권장(메모리/안정성)


class STTInfer:
    def __init__(self, cfg: STTInferConfig):
        self.cfg = cfg

        # ⚠️ WhisperModel 생성자에는 temperature/vad_filter 등이 들어가면 안 됨
        self.model = WhisperModel(
            cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
            cpu_threads=cfg.cpu_threads if cfg.device == "cpu" else 0,
            num_workers=cfg.num_workers,
        )

    def transcribe_from_pcm_i16(self, audio_i16: np.ndarray, sample_rate: int = 16000) -> str:
        """
        audio_i16: np.int16 1D PCM (mono), sample_rate=16000 가정
        return: transcription text
        """
        if audio_i16 is None:
            return ""

        audio_i16 = np.asarray(audio_i16)

        if audio_i16.size == 0:
            return ""

        # shape 정리: (N,)로
        if audio_i16.ndim != 1:
            audio_i16 = audio_i16.reshape(-1)

        # dtype 보정
        if audio_i16.dtype != np.int16:
            audio_i16 = audio_i16.astype(np.int16, copy=False)

        # 너무 짧으면 스킵(예: 0.2초 미만)
        if sample_rate == 16000 and audio_i16.size < 3200:
            return ""

        # int16 -> float32 [-1, 1], contiguous 보장
        audio_f32 = np.ascontiguousarray(audio_i16, dtype=np.float32) / 32768.0

        segments, _info = self.model.transcribe(
            audio_f32,
            language=self.cfg.language,
            vad_filter=self.cfg.vad_filter,

            beam_size=self.cfg.beam_size,
            best_of=self.cfg.best_of,
            temperature=self.cfg.temperature,

            condition_on_previous_text=self.cfg.condition_on_previous_text,
            word_timestamps=self.cfg.word_timestamps,
        )

        # faster-whisper는 segments가 generator라 바로 join 하는 게 가볍고 빠름
        texts = []
        for seg in segments:
            t = (seg.text or "").strip()
            if t:
                texts.append(t)

        return " ".join(texts).strip()
