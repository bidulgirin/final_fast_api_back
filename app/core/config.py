from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    # Android(특히 Termux)는 GPU/CUDA 거의 못 쓴다고 보면 됨
    DEVICE: str = "cpu"

    STT_MODEL_PATH: str = "assets/models/fast-whisper.pt"
    CLS_MODEL_PATH: str = "assets/models/emotion_model_android.pt"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str  # 예: postgresql+psycopg2://user:pass@localhost:5432/mydb
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-3.5-turbo"   # 예: gpt-4, gpt-3.5-turbo
    OPENAI_MAX_OUTPUT_TOKENS: int = 700

    WEB_CLIENT_ID: str = os.getenv("WEB_CLIENT_ID", "")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "dev-secret-change-me")
    JWT_ALG: str = os.getenv("JWT_ALG", "HS256")
    ACCESS_TOKEN_MINUTES: int = int(os.getenv("ACCESS_TOKEN_MINUTES", "1440"))
    

settings = Settings()
