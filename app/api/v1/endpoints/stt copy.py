from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import tempfile, os, subprocess
import imageio_ffmpeg

from starlette.concurrency import run_in_threadpool

from app.utils.crypto import decrypt_aes
from faster_whisper import WhisperModel
from app.utils.llm import postprocess_stt
from app.api.v1.endpoints.emotion import load_emotion_model, infer_emotion_probs
from app.api.v1.endpoints.mfcc import vp_store

# AE detector (내부에서 FAISS 검색/키워드 추출까지 수행)
from app.db.models.phising_sign import ae_detector

router = APIRouter()

MODEL_SIZE = "small"
stt_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
emotion_model = load_emotion_model("assets/models/emotion_model_android.pt")

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()


def convert_m4a_to_wav(m4a_path: str, wav_path: str) -> None:
    cmd = [
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", m4a_path, "-ac", "1", "-ar", "16000", wav_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {result.stderr}")


@router.post("")
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

        # (1) 감정분류
        emotion_probs = None
        emotion_top = None
        try:
            emotion_probs = await run_in_threadpool(infer_emotion_probs, emotion_model, wav_path)
            emotion_top = max(emotion_probs.items(), key=lambda x: x[1])[0] if emotion_probs else None
        except Exception as e:
            print("Emotion inference failed:", e)

        # STT
        segments, info = stt_model.transcribe(
            wav_path,
            language="ko",
            task="transcribe",
            beam_size=5,
            vad_filter=True,
        )

        text = "".join(seg.text for seg in segments).strip()

        if not text:
            return {
                "text": "",
                "llm": None,
                "voicephishing": None,
                "emotion": {"top": emotion_top, "probs": emotion_probs},
                "phising_sign": None,
            }

        # VP score finalize
        call_id = vp_store._last_call_id
        voicephishing_flag, voicephishing_score, vp_debug = await vp_store.finalize(call_id)

        # LLM (옵션)
        llm_result = None
        if llm:
            llm_result = postprocess_stt(
                text=text,
                is_voicephishing=voicephishing_flag,
                voicephishing_score=voicephishing_score if voicephishing_score is not None else 0.0,
            )

        # =========================
        # AE (내부 FAISS 검색 포함)
        # - llm_result 있으면 우선, 없으면 stt text
        # - 의심/주의일 때만 돌리고 싶으면 is_suspicious 조건 사용
        # =========================
        ae_input = llm_result if (llm_result and isinstance(llm_result, str)) else text

        ae_result = None
        try:
            # 의심 판정 기준(예시!!!!!!!!) - 프로젝트에 맞게 조정
            is_suspicious = bool(voicephishing_flag) or ((voicephishing_score or 0.0) >= 0.5)

            if is_suspicious:
                # 키워드 파라미터 없이 호출: 내부에서 FAISS 검색 수행
                ae_result = await run_in_threadpool(ae_detector.predict, ae_input)
            else:
                ae_result = None

        except Exception as e:
            print("AE phishing_sign inference failed:", e)
            ae_result = None
        
        return {
            "text": text,
            "llm": llm_result,
            "voicephishing": {
                "flag": voicephishing_flag,
                "score": voicephishing_score,
                "debug": vp_debug,
            },
            "emotion": {
                "top": emotion_top,
                "probs": emotion_probs,
            },
            "phising_sign": ae_result,
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
        except Exception:
            pass
