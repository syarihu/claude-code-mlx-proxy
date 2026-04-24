import os
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()


class Config:
    """Configuration for the Claude Code MLX Proxy"""

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8888"))

    # Model settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "mlx-community/GLM-4.5-Air-3bit")
    TRUST_REMOTE_CODE: bool = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"
    EOS_TOKEN: Optional[str] = os.getenv("EOS_TOKEN")

    # Generation settings
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "1.0"))
    DEFAULT_TOP_P: float = float(os.getenv("DEFAULT_TOP_P", "1.0"))

    # API settings
    API_MODEL_NAME: str = os.getenv("API_MODEL_NAME", "claude-4-sonnet-20250514")

    # Inject a system-prompt instruction asking the model to use <think>
    # tags.  Set to false for models that natively emit <think> (e.g. Qwen3).
    INJECT_THINKING_PROMPT: bool = (
        os.getenv("INJECT_THINKING_PROMPT", "true").lower() == "true"
    )

    # Tool handling — reduce prompt size by stripping detailed JSON schemas
    # from tool definitions.  "full" keeps everything, "slim" sends only
    # name + description + required param names, "none" omits tools entirely.
    TOOL_MODE: str = os.getenv("TOOL_MODE", "slim")

    # Extra stop sequences (comma-separated).  Generation stops when any of
    # these appear in the raw model output.  Useful when the model's EOS
    # token doesn't fire (e.g. GLM-4's <|end|> isn't in eos_token_ids).
    STOP_SEQUENCES: List[str] = [
        s
        for s in os.getenv("STOP_SEQUENCES", "<|user|>,<|end|>,<|im_start|>").split(",")
        if s
    ]

    # Logging
    VERBOSE: bool = os.getenv("VERBOSE", "false").lower() == "true"


config = Config()
