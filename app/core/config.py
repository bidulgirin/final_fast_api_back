from pydantic import BaseModel

class Settings(BaseModel):
    # Android(특히 Termux)는 GPU/CUDA 거의 못 쓴다고 보면 됨
    DEVICE: str = "cpu"

    STT_MODEL_PATH: str = "assets/models/fast-whisper.pt"
    CLS_MODEL_PATH: str = "assets/models/emotion_model_android.pt"

settings = Settings()
