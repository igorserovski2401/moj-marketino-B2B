"""Centralised configuration – loads settings from .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


class Settings:
    """Application-wide settings resolved from environment variables."""

    # Supabase
    supabase_url: str | None = os.getenv("SUPABASE_URL", "https://uddnrppybsooxsasvkev.supabase.co")
    supabase_anon_key: str | None = os.getenv(
        "SUPABASE_ANON_KEY",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVkZG5ycHB5YnNvb3hzYXN2a2V2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMxOTA4MDUsImV4cCI6MjA3ODc2NjgwNX0.RQHaA1m7pNm0cJPQQRPSP4KpDvsQGMAOO4MDuQbZz5E",
    )

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
