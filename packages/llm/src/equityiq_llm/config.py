from enum import StrEnum

from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelTier(StrEnum):
    PRIMARY = "primary"
    FALLBACK = "fallback"
    JUDGE = "judge"
    EMBED = "embed"


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_primary_model: str = "llama3.3:70b-instruct-q4_K_M"
    ollama_fallback_model: str = "qwen2.5:32b-instruct"
    ollama_judge_model: str = "qwen2.5:32b-instruct"
    ollama_embed_model: str = "nomic-embed-text:v1.5"
    ollama_keep_alive: str = "30m"
    ollama_request_timeout_s: float = 300.0

    def model_for(self, tier: ModelTier) -> str:
        match tier:
            case ModelTier.PRIMARY:
                return self.ollama_primary_model
            case ModelTier.FALLBACK:
                return self.ollama_fallback_model
            case ModelTier.JUDGE:
                return self.ollama_judge_model
            case ModelTier.EMBED:
                return self.ollama_embed_model
