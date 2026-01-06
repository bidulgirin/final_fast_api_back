from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import tempfile, os, subprocess
import imageio_ffmpeg

from app.utils.crypto import decrypt_aes
from faster_whisper import WhisperModel
from app.utils.llm import postprocess_stt
from app.api.v1.endpoints.emotion import load_emotion_model, infer_emotion_probs

# mfcc store import (같은 인스턴스를 써야 함)
from app.api.v1.endpoints.mfcc import vp_store

router = APIRouter()
# large v3 + gpu + int 16 으로 변환해야함 (그래야 정확도 올라감)
MODEL_SIZE = "small"
# MODEL_SIZE = "large-v3"
stt_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
emotion_model = load_emotion_model("assets/models/emotion_model_android.pt")


FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

def convert_m4a_to_wav(m4a_path: str, wav_path: str) -> None:
    cmd = [FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
           "-i", m4a_path, "-ac", "1", "-ar", "16000", wav_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {result.stderr}")
# 들어온 데이터를 stt 로 변환한거 + 감성분류
@router.post("/stt")  
async def stt_endpoint(
    iv: str = Form(...),
    audio: UploadFile = File(...),
    llm: bool = Form(True),
):
    m4a_path = None
    wav_path = None

    try:
        encrypted_bytes = await audio.read()
        if not encrypted_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        m4a_bytes = decrypt_aes(iv, encrypted_bytes)

        if b"ftyp" not in m4a_bytes[:64]:
            raise HTTPException(status_code=400, detail="Decrypted bytes are not m4a (ftyp not found)")

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f_m4a:
            f_m4a.write(m4a_bytes)
            m4a_path = f_m4a.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_wav:
            wav_path = f_wav.name

        convert_m4a_to_wav(m4a_path, wav_path)

        segments, info = stt_model.transcribe(
            wav_path, 
            language="ko", 
            task="transcribe",
            beam_size=5,        # 기본이 5인 경우가 많지만 명시해도 좋음(필수)
            vad_filter=True,    # 무음 구간 제거로 품질/안정성에 도움될 때 많음
            )
        

        



        text = "".join(seg.text for seg in segments).strip()

        if not text:
            return {"text": "", "llm": None}

        # 통화 종료 시점: mfcc 점수들 최종 집계
        call_id = vp_store._last_call_id
        voicephishing_flag, voicephishing_score, vp_debug = await vp_store.finalize(call_id)

        llm_result = None
        if llm:
            llm_result = postprocess_stt(
                text=text,
                is_voicephishing=voicephishing_flag,
                voicephishing_score=voicephishing_score if voicephishing_score is not None else 0.0,
            )
            print("wav_path", wav_path)
            result = infer_emotion_probs(emotion_model, wav_path)
            print("감성감성~~~", result)

        # 디버그 확인용
        print("VP_DEBUG:", vp_debug)
        print("결과확인", llm_result)

        return {
            "text": text,
            "llm": llm_result,
            "voicephishing": {
                "flag": voicephishing_flag,
                "score": voicephishing_score,
                "debug": vp_debug,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")
    finally:
        try:
            if m4a_path and os.path.exists(m4a_path):
                os.remove(m4a_path)
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except:
            pass
