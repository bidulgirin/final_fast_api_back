## 이건 통화 끝나고 발동됨
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import tempfile, os
from starlette.concurrency import run_in_threadpool

from app.utils.crypto import decrypt_aes
from faster_whisper import WhisperModel
from app.utils.llm import postprocess_stt
from app.api.v1.endpoints.emotion import load_emotion_model, infer_emotion_probs
from app.api.v1.endpoints.real_time_check import vp_store
from app.db.models.phising_sign import ae_detector
import imageio_ffmpeg
import tempfile, os, subprocess

router = APIRouter()

MODEL_SIZE = "large-v3"
stt_model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8")
emotion_model = load_emotion_model("assets/models/emotion_model_android.pt")


def _normalize(v: str, default: str) -> str:
    return (v or default).strip().lower()


def _is_wav_bytes(b: bytes) -> bool:
    # WAV: "RIFF" .... "WAVE"
    return len(b) >= 12 and b[0:4] == b"RIFF" and b[8:12] == b"WAVE"


async def _decrypt_encrypted_wav_to_temp(iv: str, audio: UploadFile) -> str:
    enc = await audio.read()
    if not enc:
        raise HTTPException(status_code=400, detail="Empty encrypted audio")

    try:
        wav_bytes = decrypt_aes(iv, enc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"decrypt failed: {e}")

    if not _is_wav_bytes(wav_bytes):
        raise HTTPException(status_code=400, detail="Decrypted bytes are not wav (RIFF/WAVE not found)")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        return f.name


async def _run_stt_only(wav_path: str) -> str:
    def _sync() -> str:
        segments, _info = stt_model.transcribe(
            wav_path,
            language="ko",
            task="transcribe",
            beam_size=5,
            vad_filter=True,
        )
        return "".join(seg.text for seg in segments).strip()

    return await run_in_threadpool(_sync)


async def _run_pipeline_wav(
    wav_path: str,
    llm: bool,
    run_emotion: bool,
    run_vp: bool,
    run_ae: bool,
    mfcc_call_id: str | None,
) -> dict:
    emotion_probs = None
    emotion_top = None
    if run_emotion:
        try:
            emotion_probs = await run_in_threadpool(infer_emotion_probs, emotion_model, wav_path)
            emotion_top = max(emotion_probs.items(), key=lambda x: x[1])[0] if emotion_probs else None
        except Exception:
            emotion_probs = None
            emotion_top = None

    text = await _run_stt_only(wav_path)

    if not text:
        return {
            "text": "",
            "llm": None,
            "voicephishing": None,
            "emotion": {"top": emotion_top, "probs": emotion_probs},
            "phising_sign": None,
        }

    voicephishing_flag = None
    voicephishing_score = None
    vp_debug = None
    if run_vp:
        try:
            vp_id = mfcc_call_id if mfcc_call_id else vp_store._last_call_id
            voicephishing_flag, voicephishing_score, vp_debug = await vp_store.finalize(vp_id)
        except Exception:
            voicephishing_flag, voicephishing_score, vp_debug = None, None, None

    llm_result = None
    if llm:
        try:
            llm_result = postprocess_stt(
                text=text,
                is_voicephishing=bool(voicephishing_flag) if voicephishing_flag is not None else False,
                voicephishing_score=voicephishing_score if voicephishing_score is not None else 0.0,
            )
        except Exception:
            llm_result = None

    ae_result = None
    if run_ae:
        try:
            ae_input = llm_result if (llm_result and isinstance(llm_result, str)) else text
            is_suspicious = bool(voicephishing_flag) or ((voicephishing_score or 0.0) >= 0.5)

            if is_suspicious:
                ae_result = await run_in_threadpool(ae_detector.predict, ae_input)
            else:
                ae_result = None
        except Exception:
            ae_result = None

    return {
        "text": text,
        "llm": llm_result,
        "voicephishing": None if not run_vp else {
            "flag": voicephishing_flag,
            "score": voicephishing_score,
            "debug": vp_debug,
        },
        "emotion": {"top": emotion_top, "probs": emotion_probs},
        "phising_sign": ae_result,
    }


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
        print("voicephishing_flag", voicephishing_flag)

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
            # is_suspicious = bool(voicephishing_flag) or ((voicephishing_score or 0.0) >= 0.5)

            # if is_suspicious:
            #     # 키워드 파라미터 없이 호출: 내부에서 FAISS 검색 수행
            #     ae_result = await run_in_threadpool(ae_detector.predict, ae_input)
            # else:
            #     ae_result = None

            ae_result = await run_in_threadpool(ae_detector.predict, ae_input)

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

    

@router.post("/dual_wav")
async def stt_dual_wav_endpoint(
    call_id: str = Form(None),
    mfcc_call_id: str = Form(None),

    iv_uplink: str = Form(...),
    audio_uplink: UploadFile = File(...),

    iv_downlink: str = Form(...),
    audio_downlink: UploadFile = File(...),

    llm: bool = Form(True),

    stt_mode: str = Form("both"),            # both | uplink | downlink
    analysis_target: str = Form("downlink"), # downlink | uplink | both | none
    return_mode: str = Form("compat"),       # compat | full
):
    up_wav_path = None
    dn_wav_path = None

    try:
        stt_mode = _normalize(stt_mode, "both")
        analysis_target = _normalize(analysis_target, "downlink")
        return_mode = _normalize(return_mode, "compat")

        if stt_mode not in ("both", "uplink", "downlink"):
            raise HTTPException(status_code=400, detail="stt_mode must be both|uplink|downlink")
        if analysis_target not in ("downlink", "uplink", "both", "none"):
            raise HTTPException(status_code=400, detail="analysis_target must be downlink|uplink|both|none")
        if return_mode not in ("compat", "full"):
            raise HTTPException(status_code=400, detail="return_mode must be compat|full")

        up_wav_path = await _decrypt_encrypted_wav_to_temp(iv_uplink, audio_uplink)
        dn_wav_path = await _decrypt_encrypted_wav_to_temp(iv_downlink, audio_downlink)

        do_stt_uplink = stt_mode in ("both", "uplink")
        do_stt_downlink = stt_mode in ("both", "downlink")

        if analysis_target == "none":
            analyze_uplink = analyze_downlink = False
        elif analysis_target == "both":
            analyze_uplink = analyze_downlink = True
        elif analysis_target == "uplink":
            analyze_uplink, analyze_downlink = True, False
        else:
            analyze_uplink, analyze_downlink = False, True

        empty_result = {
            "text": "",
            "llm": None,
            "voicephishing": None,
            "emotion": {"top": None, "probs": None},
            "phising_sign": None,
        }

        results = {"uplink": empty_result, "downlink": empty_result}

        if do_stt_uplink:
            results["uplink"] = await _run_pipeline_wav(
                wav_path=up_wav_path,
                llm=llm and analyze_uplink,
                run_emotion=analyze_uplink,
                run_vp=analyze_uplink,
                run_ae=analyze_uplink,
                mfcc_call_id=mfcc_call_id,
            )

        if do_stt_downlink:
            results["downlink"] = await _run_pipeline_wav(
                wav_path=dn_wav_path,
                llm=llm and analyze_downlink,
                run_emotion=analyze_downlink,
                run_vp=analyze_downlink,
                run_ae=analyze_downlink,
                mfcc_call_id=mfcc_call_id,
            )

        if return_mode == "full":
            return {
                "call_id": call_id,
                "mfcc_call_id": mfcc_call_id,
                "results": results,
            }

        primary = results["downlink"]
        other = results["uplink"]

        return {
            **primary,
            "call_id": call_id,
            "mfcc_call_id": mfcc_call_id,
            "other": other,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT dual_wav failed: {e}")
    finally:
        for p in (up_wav_path, dn_wav_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
