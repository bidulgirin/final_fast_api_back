from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.api.v1.deps import get_db
from app.core.google_verify import verify_google_id_token
from app.core.jwt_utils import create_access_token
from app.schemas.auth import GoogleLoginRequest, GoogleLoginResponse, UserOut

from app.db.models import User

router = APIRouter()

@router.post("/google", response_model=GoogleLoginResponse)
def google_login(body: GoogleLoginRequest, db: Session = Depends(get_db)):
    """
    Android → idToken 전달
    서버 → Google ID Token 검증 → users 저장(없으면 생성) → 우리 JWT 발급
    """
    # 1) 구글 토큰 검증
    try:
        payload = verify_google_id_token(body.idToken)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {e}")

    google_sub = payload.get("sub")
    email = payload.get("email")
    name = payload.get("name")
    picture = payload.get("picture")

    if not google_sub or not email:
        raise HTTPException(status_code=400, detail="Google payload missing sub/email")

    # 2) 유저 조회 (google_sub 기준)
    user = db.execute(select(User).where(User.google_sub == google_sub)).scalar_one_or_none()

    is_new = False
    if user is None:
        # 회원가입 처리
        is_new = True
        user = User(
            google_sub=google_sub,
            email=email,
            name=name,
            picture=picture,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        # (선택) 구글 프로필 변경 반영
        user.email = email
        user.name = name
        user.picture = picture
        db.commit()
        db.refresh(user)

    # 3) 우리 서비스 JWT 발급
    access_token = create_access_token(str(user.id))

    return GoogleLoginResponse(
        accessToken=access_token,
        isNewUser=is_new,
        user=UserOut(
            id=str(user.id),
            email=user.email,
            name=user.name,
            picture=user.picture,
            nickname=user.nickname,
        )
    )
