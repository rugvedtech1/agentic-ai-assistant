from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "Agentic AI Research Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # API Keys - these come from .env file
    OPENAI_API_KEY: str = ""
    TAVILY_API_KEY: str = ""

    # Model settings
    DEFAULT_MODEL: str = "gpt-4o-mini"
    TEMPERATURE: float = 0.7

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"

    model_config = {"env_file": ".env", "extra": "allow"}

# Single instance used across entire app
settings = Settings()