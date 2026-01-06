from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Android(특히 Termux)는 GPU/CUDA 거의 못 쓴다고 보면 됨
    DEVICE: str = "cpu"

    STT_MODEL_PATH: str = "assets/models/fast-whisper.pt"
    CLS_MODEL_PATH: str = "assets/models/emotion_model_android.pt"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str  # 예: postgresql+psycopg2://user:pass@localhost:5432/mydb
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-5.2"   # 필요 시 프로젝트에 맞게 변경
    OPENAI_MAX_OUTPUT_TOKENS: int = 700
    

settings = Settings()
