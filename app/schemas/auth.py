from pydantic import BaseModel

class GoogleLoginRequest(BaseModel):
    idToken: str  # Android가 보내는 구글 idToken

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
