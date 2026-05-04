from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # EDGAR is generous, but caps at ~10 req/s per IP. Stay polite.
    sec_edgar_user_agent: str = "EquityIQ research-portfolio you@example.com"
    sec_edgar_base_url: str = "https://www.sec.gov"
    sec_edgar_data_url: str = "https://data.sec.gov"
    sec_edgar_max_rps: float = 8.0
    sec_edgar_timeout_s: float = 60.0

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "equityiq"
    postgres_user: str = "equityiq"
    postgres_password: str = "equityiq_dev"

    # Chunker
    chunk_target_tokens: int = 320
    chunk_overlap_tokens: int = 48
    chunk_max_tokens: int = 480

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
