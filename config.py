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

    # Max chars for tool descriptions in slim mode (0 = no limit).
    TOOL_DESC_LIMIT: int = int(os.getenv("TOOL_DESC_LIMIT", "200"))

    # Comma-separated tool names whose descriptions are never truncated
    # in slim mode (e.g. "Agent,Bash").
    TOOL_FULL_DESC_NAMES: list = [
        n.strip()
        for n in os.getenv("TOOL_FULL_DESC_NAMES", "").split(",")
        if n.strip()
    ]

    # Response language — when set, injects a language instruction into
    # the system prompt (e.g. "ja", "en", "zh").  Unset = no injection.
    RESPONSE_LANGUAGE: str = os.getenv("RESPONSE_LANGUAGE", "")

    # Logging
    VERBOSE: bool = os.getenv("VERBOSE", "false").lower() == "true"

    # Force model name — when set, overrides the backend's reported model name.
    # Useful when the MLX backend loads from a local path but reports an HF
    # repo ID (e.g. "unsloth/...").  Setting this to the local path ensures
    # the proxy sends requests that the backend can resolve correctly.
    MODEL_NAME: str = os.getenv("MODEL_NAME", "")

    # Web search via Tavily — when set, the proxy intercepts the WebSearch
    # tool call from the local LLM, executes a real search through the Tavily
    # API, and feeds the results back to the model in an internal loop.
    # Without a key, WebSearch calls fall through unchanged (which typically
    # yields empty results because local backends cannot search the web).
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    WEB_SEARCH_MAX_RESULTS: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
    WEB_SEARCH_MAX_ITERATIONS: int = int(os.getenv("WEB_SEARCH_MAX_ITERATIONS", "5"))


config = Config()
