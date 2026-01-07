from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import tempfile, os, subprocess
import imageio_ffmpeg

from starlette.concurrency import run_in_threadpool  
from app.db.models.phising_sign import ae_detector

from app.utils.crypto import decrypt_aes
from faster_whisper import WhisperModel
from app.utils.llm import postprocess_stt
from app.api.v1.endpoints.emotion import load_emotion_model, infer_emotion_probs

from app.api.v1.endpoints.mfcc import vp_store

router = APIRouter()
MODEL_SIZE = "small"
stt_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
emotion_model = load_emotion_model("assets/models/emotion_model_android.pt")

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

def convert_m4a_to_wav(m4a_path: str, wav_path: str) -> None:
    cmd = [FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
           "-i", m4a_path, "-ac", "1", "-ar", "16000", wav_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {result.stderr}")
    

def get_keywords_from_faiss(sentence: str) -> list[str]:
    """
    TODO:
    - 여기에 Faiss 검색/추출 로직을 연결하세요.
    - 요구사항: 보이스피싱 의심~주의일 경우 키워드 뽑아서 적재/반영
    """
    return []


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

        # (1) 감정분류: wav 만들어진 직후(삭제되기 전) 수행
        emotion_probs = None
        emotion_top = None
        try:
            emotion_probs = await run_in_threadpool(infer_emotion_probs, emotion_model, wav_path)
            emotion_top = max(emotion_probs.items(), key=lambda x: x[1])[0] if emotion_probs else None
        except Exception as e:
            # 감정모델 실패가 STT 전체를 죽이지 않게 (원하면 raise로 바꿔도 됨)
            print("Emotion inference failed:", e)

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
                "emotion": {
                    "top": emotion_top,
                    "probs": emotion_probs,
                },
            }

        call_id = vp_store._last_call_id
        voicephishing_flag, voicephishing_score, vp_debug = await vp_store.finalize(call_id)

        llm_result = None
        if llm:
            llm_result = postprocess_stt(
                text=text,
                is_voicephishing=voicephishing_flag,
                voicephishing_score=voicephishing_score if voicephishing_score is not None else 0.0,
            )



        # =========================
        # (추가) AE 보이스피싱 문장모델 추론
        # - llm_result가 있으면 우선 사용, 없으면 stt text 사용
        # - keywords는 faiss에서 뽑아 주입 (stub 자리)
        # =========================
        ae_input = llm_result if (llm_result and isinstance(llm_result, str)) else text

        ae_keywords = []
        try:
            # "보이스피싱 의심~주의일경우 키워드 뽑아서 적재" 요구사항 반영 지점
            # 아래 조건은 예시입니다. 프로젝트 기준으로 조건/임계값을 조정하세요.
            is_suspicious = bool(voicephishing_flag) or ((voicephishing_score or 0.0) >= 0.5)

            if is_suspicious:
                ae_keywords = await run_in_threadpool(get_keywords_from_faiss, ae_input)

            ae_result = await run_in_threadpool(ae_detector.predict, ae_input, ae_keywords)
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
            # (2) 응답에 감정 결과 포함
            "emotion": {
                "top": emotion_top,
                "probs": emotion_probs,
            },
            # 여기로 출력
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
        except:
            pass
