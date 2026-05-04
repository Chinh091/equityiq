from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "equityiq"
    postgres_user: str = "equityiq"
    postgres_password: str = "equityiq_dev"

    reranker_base_url: str = "http://localhost:8081"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_timeout_s: float = 30.0

    candidate_pool_size: int = 50
    final_top_k: int = 10
    rrf_k: int = 60

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
