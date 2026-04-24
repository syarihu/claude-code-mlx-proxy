import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration for the Claude Code MLX Proxy"""

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8888"))

    # Backend mlx-lm server URL (must include /v1)
    MLX_SERVER_URL: str = os.getenv("MLX_SERVER_URL", "http://localhost:8080/v1")

    # API-exposed model name (what Claude Code sees — must match the model
    # name configured in Claude Code's /model settings)
    API_MODEL_NAME: str = os.getenv("API_MODEL_NAME", "claude-4-sonnet-20250514")

    # Tool handling — reduce prompt size by stripping detailed JSON schemas
    # from tool definitions.  "full" keeps everything, "slim" sends only
    # name + description + required param names, "none" omits tools entirely.
    TOOL_MODE: str = os.getenv("TOOL_MODE", "slim")

    # Logging
    VERBOSE: bool = os.getenv("VERBOSE", "false").lower() == "true"


config = Config()
