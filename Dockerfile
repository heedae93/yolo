# Base
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# (선택) ffmpeg가 필요하면 주석 해제
# RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Install deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App
COPY . .

# (옵션) 비루트 사용자
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5510
# Flask 엔트리포인트: app.py 내 Flask 객체 이름이 app라고 가정
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
