# google_verify.py
import os
from google.oauth2 import id_token
from google.auth.transport import requests
from app.core.config import settings


def verify_google_id_token(token: str) -> dict:
    """
    Google ID Token 검증.
    - audience 는 WEB_CLIENT_ID 로 지정 (공식 가이드 패턴) :contentReference[oaicite:16]{index=16}
    """
    if not settings.WEB_CLIENT_ID:
        raise ValueError("WEB_CLIENT_ID 환경변수가 필요합니다.")

    # 토큰 검증 + payload 반환
    payload = id_token.verify_oauth2_token(token, requests.Request(), settings.WEB_CLIENT_ID)
    return payload
