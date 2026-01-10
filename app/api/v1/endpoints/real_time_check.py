# app/api/routes/mfcc_mel_fusion.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np

from starlette.concurrency import run_in_threadpool

from app.utils.crypto import decrypt_aes
from app.services.vp_store import VoicePhishingStore

from app.services.mfcc_infer import MFCCInfer, MFCCInferConfig
from app.services.mel_infer import MelInfer, MelInferConfig

from app.services.stt_store import STTBufferStore
from app.services.text_infer import TextInfer, TextInferConfig

from app.services.stt_infer import STTInfer, STTInferConfig
import time
import logging
import asyncio
import logging
import logging
# logger = logging.getLogger("mel")

router = APIRouter()

# logger = logging.getLogger("vp")
# logger.setLevel(logging.INFO)

mfcc_infer: MFCCInfer | None = None
mel_infer: MelInfer | None = None
text_infer: TextInfer | None = None
stt_infer: STTInfer | None = None

vp_store = VoicePhishingStore(ttl_sec=60 * 60)
stt_store = STTBufferStore(ttl_sec=60 * 60, max_keep=50)


def fuse_scores(mfcc_score: float, mel_score: float, w_mfcc: float = 0.5, w_mel: float = 0.5) -> float:
    denom = (w_mfcc + w_mel)
    if denom <= 0:
        return float((mfcc_score + mel_score) / 2.0)
    fused = (mfcc_score * w_mfcc + mel_score * w_mel) / denom
    return float(min(1.0, max(0.0, fused)))


def fuse_three(audio_score: float, text_score: float, w_audio: float = 0.8, w_text: float = 0.2) -> float:
    denom = w_audio + w_text
    if denom <= 0:
        return float((audio_score + text_score) / 2.0)
    v = (audio_score * w_audio + text_score * w_text) / denom
    return float(min(1.0, max(0.0, v)))


@router.on_event("startup")
def startup_load_models():
    global mfcc_infer, mel_infer, text_infer, stt_infer

    mfcc_infer = MFCCInfer(
        model_path="assets/models/binary_cnn_mfcc.pt",
        cfg=MFCCInferConfig(device="cuda", target_frames=500),
    )

    mel_infer = MelInfer(
        model_path="assets/models/mel_spectrogram_model.pt",
        cfg=MelInferConfig(
            device="cuda",
            sample_rate=16000,
            segment_sec=5.0,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
            fmin=20,
            fmax=8000,
            img_size=224,
        ),
    )

    text_infer = TextInfer(
        TextInferConfig(
            device="cuda",
            ae_path="assets/models/final_ae.pth",
            kobert_path="assets/models/kobert",
            threshold=5500.0,
            buffer_size=3,
        )
    )

    # 서버 STT(Whisper) 로드
    stt_infer = STTInfer(
        STTInferConfig(
            model_size="small",
            device="cuda",
            compute_type="int8",
            language="ko",
            vad_filter=False,
            beam_size=1,
            best_of=1,
        )
    )


@router.post("")
async def mfcc_mel_fusion_endpoint(
    call_id: str = Form(...),
    iv: str = Form(...),
    audio: UploadFile = File(...),
):
    t0 = time.perf_counter()
    
    if mfcc_infer is None or mel_infer is None or text_infer is None or stt_infer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    encrypted_bytes = await audio.read()
    if not encrypted_bytes:
        raise HTTPException(status_code=400, detail="Empty audio")

    try:
        pcm_bytes = decrypt_aes(iv, encrypted_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Decrypt failed")

    audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    if audio_i16.size == 0:
        raise HTTPException(status_code=400, detail="Decoded PCM is empty")

    # ----- 오디오 모델 추론 -----
    try:
        mfcc_result = mfcc_infer.predict_from_pcm_i16(audio_i16)
        mfcc_score = float(mfcc_result["phishing_score"])
    except Exception:
        raise HTTPException(status_code=500, detail="MFCC inference failed")

    try:
        mel_result = mel_infer.predict_from_pcm_i16(audio_i16)
        mel_score = float(mel_result["phishing_score"])
    except Exception:
        # logger.exception("MEL inference failed")  
        raise HTTPException(status_code=500, detail="MEL inference failed")

    audio_fused = fuse_scores(mfcc_score, mel_score, w_mfcc=0.5, w_mel=0.5)

    # ----- 서버 STT -> 누적 -> 텍스트 추론 -----
    text_payload = None
    text_risk = 0.0
    should_alert = False
    stt_text = ""

    # STT는 시간이 걸리므로 threadpool에서 실행
    try:
        stt_text = await asyncio.wait_for(
            run_in_threadpool(stt_infer.transcribe_from_pcm_i16, audio_i16, 16000),
            timeout=3.0,
        )
        print("STT text:", repr(stt_text))
    except asyncio.TimeoutError as e:
        print("STT timeout:", e)
        stt_text = ""
    except Exception as e:
        print("STT error:", repr(e))
        stt_text = ""

    if stt_text.strip():
        await stt_store.add_text(call_id, stt_text.strip())
        buffered = await stt_store.get_last_texts(call_id, n=text_infer.cfg.buffer_size)

        text_payload = text_infer.predict(buffered)
        text_risk = float(text_payload.get("risk_score", 0.0))

        if text_payload.get("status") == "🚨 CRITICAL":
            should_alert = True

    # ----- 최종 fused_score -----
    final_fused = audio_fused if text_payload is None else fuse_three(audio_fused, text_risk, w_audio=0.8, w_text=0.2)

    await vp_store.add_score(call_id, final_fused)

    if final_fused >= 0.85:
        should_alert = True
        
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print("VP_LOG", call_id, audio_fused, final_fused, should_alert)
    
    return {
        "call_id": call_id,
        "phishing_score": final_fused,
        "should_alert": should_alert,

        "stt": {
            "text": stt_text,
            "buffered_n": (len(await stt_store.get_last_texts(call_id, n=text_infer.cfg.buffer_size)) if stt_text.strip() else 0),
        },

        "audio": {
            "phishing_score": audio_fused,
            "mfcc_score": mfcc_score,
            "mel_score": mel_score,
        },

        "text": text_payload,
        "mfcc": {"raw": mfcc_result},
        "mel": {"raw": mel_result},
    }
