from pydantic import BaseModel
from typing import Optional

class GoogleLoginRequest(BaseModel):
    idToken: str  # Android가 보내는 구글 idToken
    nickname: Optional[str] = None

class UserOut(BaseModel):
    id: str
    email: str
    name: str | None = None
    picture: str | None = None
    nickname: str | None = None

class GoogleLoginResponse(BaseModel):
    accessToken: str
    isNewUser: bool
    user: UserOut


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str | None = None
    isAgree: bool


class LoginRequest(BaseModel):
    email: str
    password: str
