from pydantic import BaseModel

class UpdateProfileRequest(BaseModel):
    nickname: str
