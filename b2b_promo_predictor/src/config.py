"""Centralised configuration – loads settings from .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve the project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


class Settings:
    """Application-wide settings resolved from environment variables."""

    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    db_path: Path = PROJECT_ROOT / os.getenv("DB_PATH", "data/promos.db")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "paraphrase-MiniLM-L6-v2")

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)


settings = Settings()
