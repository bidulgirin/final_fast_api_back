# app/api/routes/mfcc_mel_fusion.py
# 5초단위의 mfcc + mel 모델을 "하나의 엔드포인트"에서 함께 추론하고
# 최종 phishing_score(퓨전 점수)를 반환하는 예시

# 5초 단위로 함 

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np

from app.utils.crypto import decrypt_aes
from app.services.vp_store import VoicePhishingStore

from app.services.mfcc_infer import MFCCInfer, MFCCInferConfig
from app.services.mel_infer import MelInfer, MelInferConfig


router = APIRouter()

mfcc_infer: MFCCInfer | None = None
mel_infer: MelInfer | None = None

# 전역 store (로컬 single-process에서 OK)
vp_store = VoicePhishingStore(ttl_sec=60 * 60)


def fuse_scores(
    mfcc_score: float,
    mel_score: float,
    w_mfcc: float = 0.5,
    w_mel: float = 0.5,
) -> float:
    """
    두 모델 점수를 가중 평균으로 퓨전한다.
    점수는 0~1 범위 확률이라고 가정한다.

    운영에서 더 나은 퓨전이 필요하면:
    - (mfcc_score, mel_score)를 입력으로 하는 logistic regression/MLP를 따로 학습
    - max(mfcc_score, mel_score) 같은 보수적 규칙
    - call_id 기준 누적 평균/중앙값 등
    으로 교체 가능하다.
    """
    # 가중치 합이 0이면 방어
    denom = (w_mfcc + w_mel)
    if denom <= 0:
        return float((mfcc_score + mel_score) / 2.0)

    fused = (mfcc_score * w_mfcc + mel_score * w_mel) / denom

    # 혹시 범위 튀면 클램프
    if fused < 0.0:
        fused = 0.0
    if fused > 1.0:
        fused = 1.0
    return float(fused)


@router.on_event("startup")
def startup_load_models():
    """
    서버 시작 시 두 모델을 모두 메모리에 로드한다.
    """
    global mfcc_infer, mel_infer

    # MFCC 모델 로드
    mfcc_infer = MFCCInfer(
        # model_path="assets/models/mfcc_best_model.pt",
        model_path="assets/models/binary_cnn_mfcc.pt",
        cfg=MFCCInferConfig(device="cpu", target_frames=500),
    )

    # MEL(MobileNetV2) 모델 로드
    # sample_rate는 "실제 PCM이 생성되는 SR"과 반드시 동일해야 한다.
    mel_infer = MelInfer(
        model_path="assets/models/mel_spectrogram_model.pt",
        cfg=MelInferConfig(
            device="cpu",
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


@router.post("")
async def mfcc_mel_fusion_endpoint(
    call_id: str = Form(...),   # 통화 식별자 (CallLog id 등)
    iv: str = Form(...),
    audio: UploadFile = File(...),
):
    """
    1) 업로드된 암호화 오디오 bytes를 읽는다.
    2) decrypt_aes로 복호화해서 raw PCM bytes를 얻는다.
    3) PCM bytes를 np.int16으로 변환한다.
    4) MFCC 모델 추론, MEL 모델 추론을 각각 수행한다.
    5) 두 점수를 퓨전해서 최종 phishing_score를 만든다.
    6) 5초 점수로 store에 저장한다.
    """
    if mfcc_infer is None or mel_infer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    encrypted_bytes = await audio.read()
    if not encrypted_bytes:
        raise HTTPException(status_code=400, detail="Empty audio")

    # 복호화: encrypted_bytes -> pcm_bytes
    try:
        pcm_bytes = decrypt_aes(iv, encrypted_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Decrypt failed")

    # PCM bytes -> int16 array
    audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    if audio_i16.size == 0:
        raise HTTPException(status_code=400, detail="Decoded PCM is empty")

    # MFCC 추론
    try:
        mfcc_result = mfcc_infer.predict_from_pcm_i16(audio_i16)
        mfcc_score = float(mfcc_result["phishing_score"])
    except Exception:
        raise HTTPException(status_code=500, detail="MFCC inference failed")

    # MEL 추론
    try:
        mel_result = mel_infer.predict_from_pcm_i16(audio_i16)
        mel_score = float(mel_result["phishing_score"])
    except Exception:
        raise HTTPException(status_code=500, detail="MEL inference failed")

    # 점수 퓨전 (가중치는 운영에서 조절)
    fused_score = fuse_scores(
        mfcc_score=mfcc_score,
        mel_score=mel_score,
        w_mfcc=0.5,
        w_mel=0.5,
    )

    # 5초 점수 저장 (퓨전 점수를 저장)
    await vp_store.add_score(call_id, fused_score)

    # 원하면 mfcc/mel 개별 점수도 같이 반환해서 디버깅 가능
    return {
        "call_id": call_id,
        "phishing_score": fused_score,
        "mfcc": {
            "phishing_score": mfcc_score,
            "raw": mfcc_result,
        },
        "mel": {
            "phishing_score": mel_score,
            "raw": mel_result,
        },
    }
