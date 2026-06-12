from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = ""
    # Ollama가 뜨는 주소. 온프레미스 서버 주소로 변경 가능
    ollama_base_url: str = "http://localhost:11434"


settings = Settings()
