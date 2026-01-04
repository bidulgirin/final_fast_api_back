from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import tempfile, os, subprocess
import imageio_ffmpeg

from app.utils.crypto import decrypt_aes
from faster_whisper import WhisperModel
from app.utils.llm import postprocess_stt 

router = APIRouter()

MODEL_SIZE = "small"
stt_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

def convert_m4a_to_wav(m4a_path: str, wav_path: str) -> None:
    cmd = [FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
           "-i", m4a_path, "-ac", "1", "-ar", "16000", wav_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {result.stderr}")

@router.post("/stt")
async def stt_endpoint(
    iv: str = Form(...),
    audio: UploadFile = File(...),
    llm: bool = Form(True),  # 옵션: LLM 후처리 on/off
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
            task="transcribe"
        )

        text = "".join(seg.text for seg in segments).strip()

        # STT가 빈 문자열이면 LLM 돌릴 필요 없음
        if not text:
            return {"text": "", "llm": None}

        # LLM 후처리 옵션
        llm_result = None
        # 보이스피싱 판별 여부 (딥보이스 + mel spectrogram 분석 결과 활용 예정)
        voicephishing_flag = True
        voicephishing_score = 0.95 # 점수도 주셈~
        if llm:
            # OpenAI 호출은 동기 함수라서(현재 유틸) blocking 될 수 있음
            # 간단하게는 그대로 써도 되지만, 트래픽 있으면 to_thread 권장(아래 참고)
            llm_result = postprocess_stt(
                    text=text,
                    is_voicephishing=voicephishing_flag,
                    voicephishing_score=voicephishing_score,
                )
        print("결과확인", llm_result)
        return {"text": text, "llm": llm_result}

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
