"""Centralised configuration – loads settings from .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


class Settings:
    """Application-wide settings resolved from environment variables."""

    # Supabase
    supabase_url: str | None = os.getenv("SUPABASE_URL")
    supabase_anon_key: str | None = os.getenv("SUPABASE_ANON_KEY")

    # Gemini
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")

    # Local
    db_path: Path = PROJECT_ROOT / os.getenv("DB_PATH", "data/promos.db")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "paraphrase-MiniLM-L6-v2")

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def has_supabase(self) -> bool:
        return bool(self.supabase_url and self.supabase_anon_key)


settings = Settings()
