from datetime import datetime, timedelta, timezone
from jose import jwt
from app.core.config import settings

def create_access_token(user_id: str) -> str:
    """
    우리 서비스용 JWT 발급.
    sub에 user_id를 넣고 exp로 만료시간 설정.
    """
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=settings.ACCESS_TOKEN_MINUTES)

    payload = {
        "sub": user_id,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALG)
