FROM python:3.10-slim

WORKDIR /app

# (선택) 시스템 패키지 필요하면 추가
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip

# 1) torch 먼저 설치 (Linux CPU wheel)
# 버전은 프로젝트에 맞게 고정 추천
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch

# 2) 나머지 requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사
COPY . .

EXPOSE 8000

# 엔트리포인트 (프로젝트에 맞게 main:app 경로 수정)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
