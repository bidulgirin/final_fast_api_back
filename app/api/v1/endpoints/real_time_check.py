from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np
import os
# 모델 로딩 전에 환경변수 설정.
os.environ["CT2_CUDA_ALLOCATOR"] = "cuda_malloc_async"  # Python import 전에

from starlette.concurrency import run_in_threadpool

from app.utils.crypto import decrypt_aes
from app.services.vp_store import VoicePhishingStore

from app.services.mfcc_infer import MFCCInfer, MFCCInferConfig
from app.services.mel_best_infer import MelBestInfer, MelInferConfig

from app.services.stt_store import STTBufferStore
from app.services.text_infer import TextInfer, TextInferConfig

from app.services.stt_infer import STTInfer, STTInferConfig
import asyncio

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
# 기본 파라미터/임계값 모음.

PCM_SAMPLE_RATE = 16000
STT_TIMEOUT_SEC = 3.0
AUDIO_FUSE_W_MFCC = 0.6
AUDIO_FUSE_W_MEL = 0.4
TEXT_FUSE_W_AUDIO = 1.0
TEXT_FUSE_W_TEXT = 0.0
TEXT_ALERT_MIN_RISK = 0.6
ALERT_THRESHOLD = 0.80

# 중복호출을 막기 위한 lock
# 모델 중복 로드를 막기 위한 락.
_load_lock = asyncio.Lock()


def fuse_scores(mfcc_score: float, mel_score: float, w_mfcc: float = 0.6, w_mel: float = 0.4) -> float:
    denom = (w_mfcc + w_mel)
    if denom <= 0:
        return float((mfcc_score + mel_score) / 2.0)
    fused = (mfcc_score * w_mfcc + mel_score * w_mel) / denom
    return float(min(1.0, max(0.0, fused)))

# def fuse_three(audio_score: float, text_score: float, w_audio: float = 0.8, w_text: float = 0.2) -> float:
#     denom = w_audio + w_text
#     if denom <= 0:
#         return float((audio_score + text_score) / 2.0)
#     v = (audio_score * w_audio + text_score * w_text) / denom
#     return float(min(1.0, max(0.0, v)))


def _require_models_loaded() -> None:
    if mfcc_infer is None or mel_infer is None or text_infer is None or stt_infer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")


async def _read_pcm_i16(iv: str, audio: UploadFile) -> np.ndarray:
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
    return audio_i16


def _infer_audio_scores(audio_i16: np.ndarray) -> float:
    try:
        mfcc_result = mfcc_infer.predict_from_pcm_i16(audio_i16)
        mfcc_score = float(mfcc_result["phishing_score"])
    except Exception:
        raise HTTPException(status_code=500, detail="MFCC inference failed")

    try:
        mel_result = mel_infer.predict_from_pcm_i16(audio_i16)
        mel_score = float(mel_result["phishing_score"])
    except Exception:
        raise HTTPException(status_code=500, detail="MEL inference failed")

    return fuse_scores(mfcc_score, mel_score, w_mfcc=AUDIO_FUSE_W_MFCC, w_mel=AUDIO_FUSE_W_MEL)


async def _run_stt(audio_i16: np.ndarray) -> str:
    try:
        stt_text = await asyncio.wait_for(
            run_in_threadpool(stt_infer.transcribe_from_pcm_i16, audio_i16, PCM_SAMPLE_RATE),
            timeout=STT_TIMEOUT_SEC,
        )
        print("STT text:", repr(stt_text))
        return stt_text or ""
    except asyncio.TimeoutError as e:
        print("STT timeout:", e)
    except Exception as e:
        print("STT error:", repr(e))
    return ""


async def _infer_text_risk(call_id: str, stt_text: str) -> tuple[dict | None, float, bool]:
    if not stt_text.strip():
        return None, 0.0, False

    cleaned = stt_text.strip()
    await stt_store.add_text(call_id, cleaned)
    buffered = await stt_store.get_last_texts(call_id, n=text_infer.cfg.buffer_size)

    text_payload = text_infer.predict(buffered)
    if not isinstance(text_payload, dict):
        return None, 0.0, False
    if not isinstance(text_payload.get("keywords"), list):
        text_payload["keywords"] = []

    text_risk = float(text_payload.get("risk_score", 0.0))
    text_status = text_payload.get("status")
    should_alert = text_status != "NORMAL" and text_risk >= TEXT_ALERT_MIN_RISK
    return text_payload, text_risk, should_alert


# def _combine_scores(audio_fused: float, text_payload: dict | None, text_risk: float) -> float:
#     if text_payload is None:
#         return audio_fused
#     return fuse_three(
#         audio_fused,
#         text_risk,
#         w_audio=TEXT_FUSE_W_AUDIO,
#         w_text=TEXT_FUSE_W_TEXT,
#     )


async def startup_load_models():
    global mfcc_infer, mel_infer, text_infer, stt_infer

    # 최초 1회만 로딩되도록 보호.
    async with _load_lock:
        # stt_infer ::: 이게 젤 무겁고 후반에 로드되어서 이건만 체크~~
        print("시작!!!")
        if stt_infer is not None:
            print("이미 stt_infer 로드됨")
            return

        print("모델 로드 중...")

    # 딥보이스 탐지 모델 
    # mfcc 모델 로드
    # 음성 모델(MFCC/MEL) 로드.
    mfcc_infer = MFCCInfer(
        model_path="assets/models/best_res2net50_se.pth",
        cfg=MFCCInferConfig(device="cpu", center=False, target_frames=498),
    )
    # mel 모델 로드
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
    
    # mfcc_infer, mel_infer 모델 예측값에 대한 추가 모델 로드 
    # 
 
    # 맥락탐지 모델 로드
    # 텍스트 위험도 모델 로드.
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
    # STT 모델은 무거우므로 1회만 로딩.
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
    _require_models_loaded()
    audio_i16 = await _read_pcm_i16(iv, audio)

    # MFCC/MEL 결과를 소프트 보팅해 음성 위험도 계산.

    # ----- 오디오 모델 추론(현재여기적용) -----
    audio_fused = _infer_audio_scores(audio_i16)

    # ----- 서버 STT -> 누적 -> 텍스트 추론 -----
    # STT는 무거우므로 threadpool에서 실행.

    # STT는 시간이 걸리므로 threadpool에서 실행
    stt_text = await _run_stt(audio_i16)

    text_payload, text_risk, should_alert = await _infer_text_risk(call_id, stt_text)

    # ----- 최종 fused_score -----
    # 음성/텍스트 점수 결합(현재 텍스트 비중 0).
    # fused_score 가 아닌 따로 알림을 울려야한다
    # 1. w_audio 가 0.9 가 넘으면 알림
    # 2. w_text 의 result 를 5초마다 알림(3개 누적한 알림보고)

    # 3. mel + mfcc 는 맞음
    # final_fused = _combine_scores(audio_fused, text_payload, text_risk)
    
    # mel + mfcc 점수 표기 
    deepvoice_score = audio_fused  # MFCC+MEL 소프트 보팅 결과. 

    # await vp_store.add_score(call_id, final_fused)
    await vp_store.add_score(call_id, audio_fused)

    if audio_fused >= ALERT_THRESHOLD:
        should_alert = True
        
    print("DEEPVOICE_SCORE", deepvoice_score)
    print("KOBERTSCORE", text_payload)
    
    return {
        "call_id": call_id,
        "deepvoiceScore": deepvoice_score, # mel + mfcc 점수
        "should_alert": should_alert,
        "text_risk" : text_risk, # 텍스트 위험도 점수(지금은...안쓰긴함...)
        "koberScore": text_payload, # kobert + ae 결과
        "keywords": (text_payload.get("keywords", []) if isinstance(text_payload, dict) else []),
        "stt": {
            "text": stt_text, # 5초 음성에 대한 STT 결과
        },
    }

#  return {
#         "call_id": call_id,
#         "deepvoiceScore": deepvoice_score, # mel + mfcc 점수
#         "should_alert": should_alert, 
#         "koberScore": text_payload, # kobert + ae 결과
#         "stt": {
#             "text": stt_text, # 5초 음성에 대한 STT 결과(stt 만)
#             "buffered_n": (len(await stt_store.get_last_texts(call_id, n=text_infer.cfg.buffer_size)) if stt_text.strip() else 0),
#         },
#         "audio": {
#             "deepvoiceScore": audio_fused,
#             "mfcc_score": mfcc_score,
#             "mel_score": mel_score,
#         },

        
#         "mfcc": {"raw": mfcc_result},
#         "mel": {"raw": mel_result},
#     }
