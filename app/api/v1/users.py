from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.session import get_db, get_current_user_id
from app.schemas.users import UpdateProfileRequest
from app.schemas.auth import UserOut
from app.db.models import User

router = APIRouter()

@router.put("/me", response_model=UserOut)
def update_me(
    body: UpdateProfileRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Authorization: Bearer <우리JWT>
    닉네임 등 프로필 업데이트
    """
    user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")


    user.nickname = body.nickname

    db.commit()
    db.refresh(user)

    return UserOut(
        id=str(user.id),
        email=user.email,
        name=user.name,
        picture=user.picture,
        nickname=user.nickname,
    )
