# # 5초단위의 
# # mfcc + mel 모델 임


# from fastapi import APIRouter, UploadFile, File, Form, HTTPException
# import numpy as np
# from app.utils.crypto import decrypt_aes
# from app.services.mfcc_infer import MFCCInfer, MFCCInferConfig
# from app.services.vp_store import VoicePhishingStore

# router = APIRouter()
# mfcc_infer: MFCCInfer | None = None

# # 전역 store (로컬 single-process에서 OK)
# vp_store = VoicePhishingStore(ttl_sec=60 * 60)

# @router.on_event("startup")
# def startup_load_model():
#     global mfcc_infer
#     mfcc_infer = MFCCInfer(
#         model_path="assets/models/mfcc_best_model.pt",
#         cfg=MFCCInferConfig(device="cpu", target_len=501),
#     )

# @router.post("") 
# async def mfcc_endpoint(
#     call_id: str = Form(...),      # 통화 식별자 (CallLog id 등) 1로 고정 시킴
#     iv: str = Form(...),
#     audio: UploadFile = File(...)
# ):
#     if mfcc_infer is None:
#         raise HTTPException(status_code=503, detail="MFCC model not loaded")

#     encrypted_bytes = await audio.read()
#     if not encrypted_bytes:
#         raise HTTPException(status_code=400, detail="Empty audio")

#     pcm_bytes = decrypt_aes(iv, encrypted_bytes)
#     audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
#     if audio_i16.size == 0:
#         raise HTTPException(status_code=400, detail="Decoded PCM is empty")

#     result = mfcc_infer.predict_from_pcm_i16(audio_i16)
#     score = float(result["phishing_score"])

#     # 5초 점수 저장
#     await vp_store.add_score(call_id, score)

#     # 원하면 debug로 통계도 같이 반환 가능
#     return result
