from enum import StrEnum

from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelTier(StrEnum):
    PRIMARY = "primary"
    FALLBACK = "fallback"
    JUDGE = "judge"
    EMBED = "embed"


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_primary_model: str = "openai/gpt-oss-120b:free"
    openrouter_fallback_model: str = "openai/gpt-oss-20b:free"
    openrouter_judge_model: str = "openai/gpt-oss-20b:free"
    openrouter_embed_model: str = "nomic-ai/nomic-embed-text-v1.5"
    openrouter_request_timeout_s: float = 120.0

    def model_for(self, tier: ModelTier) -> str:
        match tier:
            case ModelTier.PRIMARY:
                return self.openrouter_primary_model
            case ModelTier.FALLBACK:
                return self.openrouter_fallback_model
            case ModelTier.JUDGE:
                return self.openrouter_judge_model
            case ModelTier.EMBED:
                return self.openrouter_embed_model
