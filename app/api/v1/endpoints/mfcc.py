# 오디오 음성을 mfcc 모델 돌리기 위함
# mfcc로 변환해야함 
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np
import tempfile
import soundfile as sf

from app.utils.crypto import decrypt_aes

# faster-whisper
from faster_whisper import WhisperModel  # :contentReference[oaicite:2]{index=2}

router = APIRouter()

# 서버 시작 시 1회 로딩 (매 요청마다 로딩하면 너무 느림)
# model_size: "tiny", "base", "small", "medium", "large-v3" 등
# CPU면 tiny/base/small 추천
MODEL_SIZE = "small"
stt_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")  # :contentReference[oaicite:3]{index=3}


@router.post("/mfcc")
async def mfcc_endpoint(
    iv: str = Form(...),
    audio: UploadFile = File(...)
):
    try:
        # 1) 암호화된 오디오 읽기
        encrypted_bytes = await audio.read()
        if not encrypted_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # 2) AES 복호화 -> PCM bytes
        pcm_bytes = decrypt_aes(iv, encrypted_bytes)

        # 3) PCM bytes -> np.int16
        audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        if audio_i16.size == 0:
            raise HTTPException(status_code=400, detail="Decoded PCM is empty")

        # 4) wav 파일로 임시 저장 (가장 쉬운 방식)
        #    (soundfile은 float/short 모두 저장 가능)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_i16, samplerate=16000)
            wav_path = f.name

        # 5) STT 실행 (한국어 고정)
        # segments는 generator 비슷하게 나오므로 text를 이어붙임
        segments, info = stt_model.transcribe(
            wav_path,
            language="ko",
            task="transcribe"
        )  # :contentReference[oaicite:4]{index=4}

        text_parts = []
        for seg in segments:
            # seg.text 안에 각 구간 텍스트
            text_parts.append(seg.text)

        text = "".join(text_parts).strip()
        # mfcc_best_model 로 5초 분량의 음성데이터를 넣고 돌림
        return {"text": text, "language": getattr(info, "language", "ko")}

    except HTTPException:
        raise
    except Exception as e:
        # 디버깅용 에러 메시지(운영에서는 숨기는 게 좋음)
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")
