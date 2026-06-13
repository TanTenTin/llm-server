from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = ""
    google_ai_api_key: str = ""
    # Ollama가 뜨는 주소. 온프레미스 서버 주소로 변경 가능
    ollama_base_url: str = "http://localhost:11434"
    # 게이트웨이 공유 인증 토큰. 설정 시 /v1/* 요청에 'Authorization: Bearer <키>' 필요.
    # 비우면 인증 없이 개방 → 외부(llm.tan-kim.com) 노출 시 반드시 설정할 것.
    gateway_api_key: str = ""


settings = Settings()
