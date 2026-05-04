from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    agent_max_subqueries: int = 4
    agent_retrieve_top_k: int = 8
    agent_critic_min_faithfulness: float = 0.6
    agent_max_revision_rounds: int = 1
