from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
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


def decode(token: str, secret: str | None = None, algorithms: list | None = None) -> dict:
    """Decode and verify a JWT.

    - If `secret` or `algorithms` are not provided, use defaults from `settings`.
    - Raises `jose.JWTError` on failure so callers can handle authentication errors.
    """
    if secret is None:
        secret = settings.JWT_SECRET
    if algorithms is None:
        algorithms = [settings.JWT_ALG]

    try:
        payload = jwt.decode(token, secret, algorithms=algorithms)
        return payload
    except JWTError:
        # propagate so callers (e.g., deps) can catch and return 401
        raise
