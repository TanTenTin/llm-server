# llm-server 컨테이너 이미지.
# Oracle ARM(aarch64) 인스턴스에서 CPU 추론 전제. python:3.12-slim은 멀티아치(arm64) 지원.
FROM python:3.12-slim

WORKDIR /app

# 의존성 먼저 설치해 레이어 캐시 활용 (소스만 바뀌면 재설치 안 함)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스
COPY app ./app

EXPOSE 8000

# 0.0.0.0 바인드 — compose가 호스트 8000으로 퍼블리시, Caddy(Lightsail)가 공인 IP로 접근
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
