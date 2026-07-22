# llm-server 컨테이너 이미지.
# Oracle ARM(aarch64) 인스턴스에서 CPU 추론 전제. python:3.12-slim은 멀티아치(arm64) 지원.
FROM python:3.12-slim

WORKDIR /app

# 로컬 Realtime 음성(model=local-live) 의존성 설치 여부.
# 기본 false — Oracle ARM 게이트웨이는 gemini-live만 중계하므로 무거운 스택(torch·MeloTTS·
# faster-whisper)을 넣지 않는다(이미지 비대·ARM 빌드 지연 방지). 로컬 파이프라인을 실제로
# 돌릴 호스트(GPU/CPU 여유가 있는 로컬-LLM 서버)에서만 true로 빌드한다:
#     docker compose build --build-arg INSTALL_REALTIME=true
# 미설치 상태로 local-live 세션이 열리면 게이트웨이가 'RealtimeMediaUnavailable' 에러
# 이벤트로 정직하게 알리고 닫는다(그 외 경로는 무영향).
ARG INSTALL_REALTIME=false

# 의존성 먼저 설치해 레이어 캐시 활용 (소스만 바뀌면 재설치 안 함)
COPY requirements.txt requirements-realtime.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_REALTIME" = "true" ]; then \
         pip install --no-cache-dir -r requirements-realtime.txt; \
       fi

# 애플리케이션 소스
COPY app ./app

EXPOSE 8000

# 0.0.0.0 바인드 — compose가 호스트 8000으로 퍼블리시, Caddy(Lightsail)가 공인 IP로 접근
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
