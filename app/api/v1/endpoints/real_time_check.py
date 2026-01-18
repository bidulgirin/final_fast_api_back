from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np
import os
os.environ["CT2_CUDA_ALLOCATOR"] = "cuda_malloc_async"  # Python import 전에

from starlette.concurrency import run_in_threadpool

from app.utils.crypto import decrypt_aes
from app.services.vp_store import VoicePhishingStore

from app.services.mfcc_infer import MFCCInfer, MFCCInferConfig
from app.services.mel_best_infer import MelBestInfer, MelInferConfig

from app.services.stt_store import STTBufferStore
from app.services.text_infer import TextInfer, TextInferConfig

from app.services.stt_infer import STTInfer, STTInferConfig
import time
import asyncio
import logging

router = APIRouter(
    prefix="/real_time",
    tags=["real_time"],
)

mfcc_infer: MFCCInfer | None = None
mel_infer: MelBestInfer | None = None
text_infer: TextInfer | None = None
stt_infer: STTInfer | None = None

vp_store = VoicePhishingStore(ttl_sec=60 * 60)
stt_store = STTBufferStore(ttl_sec=60 * 60, max_keep=50)

# 중복호출을 막기 위한 lock
# Prevent concurrent model loads on startup.
_load_lock = asyncio.Lock()


def fuse_scores(mfcc_score: float, mel_score: float, w_mfcc: float = 0.5, w_mel: float = 0.5) -> float:
    denom = (w_mfcc + w_mel)
    if denom <= 0:
        return float((mfcc_score + mel_score) / 2.0)
    fused = (mfcc_score * w_mfcc + mel_score * w_mel) / denom
    return float(min(1.0, max(0.0, fused)))

# 이거 안쓸것임ㅋㅋㅋㅋㅋ
def fuse_three(audio_score: float, text_score: float, w_audio: float = 0.8, w_text: float = 0.2) -> float:
    denom = w_audio + w_text
    if denom <= 0:
        return float((audio_score + text_score) / 2.0)
    v = (audio_score * w_audio + text_score * w_text) / denom
    return float(min(1.0, max(0.0, v)))


async def startup_load_models():
    global mfcc_infer, mel_infer, text_infer, stt_infer

    async with _load_lock:
        # stt_infer ::: 이게 젤 무겁고 후반에 로드되어서 이건만 체크~~
        print("시작!!!")
        if stt_infer is not None:
            print("이미 stt_infer 로드됨")
            return

        print("모델 로드 중...")
    mfcc_infer = MFCCInfer(
        model_path="assets/models/best_res2net50_se.pth",
        cfg=MFCCInferConfig(device="cpu", center=False, target_frames=498),
    )
   
    mel_infer = MelBestInfer(
        model_path="assets/models/best_model_tuning.pth",
        cfg=MelInferConfig(
            device="cpu",
            input_sample_rate=16000,
            target_sample_rate=22050,
            duration_sec=5, 
            n_mels=224,
            hop_length=512,
            img_size=224,
            threshold=0.6,
            model_name="res2net50_26w_4s",
            num_classes=2,
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
    # Load heavy STT model once to avoid per-request overhead.
    stt_infer = STTInfer(
        STTInferConfig(
            model_size="large-v3",
            device="cuda",
            compute_type="float16",
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

    # Audio-model inference (MFCC + MEL) for fast risk scoring.

    # ----- 오디오 모델 추론 -----
    try:
        mfcc_result = mfcc_infer.predict_from_pcm_i16(audio_i16)
        mfcc_score = float(mfcc_result["phishing_score"])
        print("mfcc_result", mfcc_result )
        print("mfcc_score", mfcc_score )
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

    # STT is CPU/GPU heavy, so keep the event loop responsive.

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
    # Final score favors audio, but allows text risk to lift the result.
    # fused_score 가 아닌 따로 알림을 울려야한다
    # 1. w_audio 가 0.9 가 넘으면 알림
    # 2. w_text 의 result 를 5초마다 알림(3개 누적한 알림보고)

    # 3. mel + mfcc 는 맞음
    final_fused = audio_fused if text_payload is None else fuse_three(audio_fused, text_risk, w_audio=0.8, w_text=0.2)
    
    # mel + mfcc 점수 표기 
    deepvoice_score = audio_fused # 0.5 * mfcc + 0.5 * mel

    await vp_store.add_score(call_id, final_fused)

    if final_fused >= 0.80:
        should_alert = True
        
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print("VP_LOG", call_id, audio_fused, final_fused, should_alert)
    
    return {
        "call_id": call_id,
        "phishing_score": final_fused, # 최종 피싱 점수 xxxx 안씀
        "deepvoice_score": deepvoice_score, # mel + mfcc 점수
        "should_alert": should_alert, 

        "stt": {
            "text": stt_text, # 5초 음성에 대한 STT 결과(stt 만)
            "buffered_n": (len(await stt_store.get_last_texts(call_id, n=text_infer.cfg.buffer_size)) if stt_text.strip() else 0),
        },
        "audio": {
            "phishing_score": audio_fused,
            "mfcc_score": mfcc_score,
            "mel_score": mel_score,
        },

        "text": text_payload, # kobert + ae 결과
        "mfcc": {"raw": mfcc_result},
        "mel": {"raw": mel_result},
    }
