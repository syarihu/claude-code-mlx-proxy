import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Literal
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import PreTrainedTokenizerFast
from mlx_lm import load, generate, stream_generate
from mlx_lm.utils import load_model, hf_repo_to_path, snapshot_download
from mlx_lm.tokenizer_utils import TokenizerWrapper
from config import config

# Global variables for model and tokenizer
model = None
tokenizer = None

# Thinking tag pairs used by different models / system prompts.
# Each entry is (open_tag, close_tag).
THINKING_TAG_PAIRS: List[tuple[str, str]] = [
    ("<antThinking>", "</antThinking>"),
    ("<think>", "</think>"),
]

# Build a combined regex for one-shot filtering (non-streaming path).
_THINKING_TAG_PATTERN = re.compile(
    "|".join(
        re.escape(open_t) + ".*?" + re.escape(close_t)
        for open_t, close_t in THINKING_TAG_PAIRS
    ),
    re.DOTALL,
)


class ThinkingFilter:
    """Stateful filter that strips thinking blocks from a stream of text
    chunks.  Supports multiple tag formats (e.g. <think>..</think>,
    <antThinking>..</antThinking>).  Tracks whether we are inside a
    thinking block so partial tokens at chunk boundaries are handled
    correctly."""

    def __init__(self):
        self._inside_thinking = False
        self._open_tag = ""
        self._close_tag = ""
        self._buffer = ""

    def filter_chunk(self, text: str) -> str:
        """Return the portion of *text* that is NOT inside a thinking block."""
        self._buffer += text

        if self._inside_thinking:
            end = self._buffer.find(self._close_tag)
            if end == -1:
                return ""
            self._buffer = self._buffer[end + len(self._close_tag):]
            self._inside_thinking = False
            self._open_tag = ""
            self._close_tag = ""
            # After exiting, continue to drain any remaining content.
        else:
            return self._drain_non_thinking()

        return self._drain_non_thinking()

    def _drain_non_thinking(self) -> str:
        """Extract all non-thinking content from the buffer."""
        parts: List[str] = []
        buf = self._buffer

        while True:
            # Find the earliest opening tag among all known pairs.
            earliest_pos = len(buf)
            earliest_open = ""
            earliest_close = ""
            for open_t, close_t in THINKING_TAG_PAIRS:
                pos = buf.find(open_t)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    earliest_open = open_t
                    earliest_close = close_t

            if earliest_pos == len(buf):
                # No complete opening tag found.  Hold back a trailing
                # partial prefix (e.g. "<antThink" without "ing>") so we
                # don't emit it before the next chunk completes the tag.
                # Only check the END of the buffer — a prefix appearing in
                # the middle is clearly not a partial tag.
                safe_end = len(buf)
                for open_t, _ in THINKING_TAG_PAIRS:
                    for plen in range(1, len(open_t)):
                        prefix = open_t[:plen]
                        if buf.endswith(prefix):
                            safe_end = min(safe_end, len(buf) - plen)
                            break
                if safe_end < len(buf):
                    parts.append(buf[:safe_end])
                    self._buffer = buf[safe_end:]
                else:
                    parts.append(buf)
                    self._buffer = ""
                break

            parts.append(buf[:earliest_pos])
            buf = buf[earliest_pos + len(earliest_open):]

            end = buf.find(earliest_close)
            if end == -1:
                # Unclosed tag — enter thinking mode
                self._inside_thinking = True
                self._open_tag = earliest_open
                self._close_tag = earliest_close
                self._buffer = ""
                break

            buf = buf[end + len(earliest_close):]

        return "".join(parts)

    def filter_remaining(self) -> str:
        """Drain any remaining buffered text (called at end of stream)."""
        if self._inside_thinking:
            # If we're still inside a thinking block, discard it.
            self._inside_thinking = False
            self._open_tag = ""
            self._close_tag = ""
        output = self._buffer
        self._buffer = ""
        return output


def filter_thinking_text(text: str) -> str:
    """One-shot filter: remove all thinking blocks."""
    return _THINKING_TAG_PATTERN.sub("", text)


# Content block models
class ContentBlockText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"] = "image"
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class SystemContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingConfig(BaseModel):
    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: Optional[int] = None


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
            ]
        ],
    ]


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class MessageResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[ContentBlockText]
    model: str
    stop_reason: str = "end_turn"
    stop_sequence: Optional[str] = None
    usage: Usage


class MessageStreamResponse(BaseModel):
    type: str
    index: Optional[int] = None
    delta: Optional[Dict[str, Any]] = None
    usage: Optional[Usage] = None


def _load_model_with_fallback(model_name: str, tokenizer_config: dict):
    """Load model and tokenizer, falling back to PreTrainedTokenizerFast when
    the model's tokenizer uses the Transformers v5 TokenizersBackend class which
    is not available in older Transformers installations.
    """
    try:
        return load(model_name, tokenizer_config=tokenizer_config)
    except ValueError as e:
        if "TokenizersBackend" not in str(e):
            raise
        print(
            "Warning: Failed to load tokenizer via AutoTokenizer (TokenizersBackend not "
            "found). This typically means the model was saved with Transformers v5 but an "
            "older version is installed. Upgrading to 'transformers>=5.0.0' is recommended. "
            "Attempting fallback using PreTrainedTokenizerFast..."
        )

    # Use cached model files (downloaded by the failed load() call above, or already
    # present from a previous run).
    model_path = Path(hf_repo_to_path(model_name))
    if not model_path.exists():
        model_path = Path(snapshot_download(model_name))
    mlx_model, mlx_config = load_model(model_path)

    # PreTrainedTokenizerFast.from_pretrained does not do the tokenizer-class
    # name lookup that AutoTokenizer performs, so it avoids the TokenizersBackend
    # error while still reading tokenizer.json correctly.
    hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        str(model_path), **tokenizer_config
    )
    eos_token_id = mlx_config.get("eos_token_id")
    tokenizer = TokenizerWrapper(hf_tokenizer, eos_token_ids=eos_token_id)

    return mlx_model, tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model, tokenizer
    print(f"Loading MLX model: {config.MODEL_NAME}")

    # Prepare tokenizer config
    tokenizer_config = {}
    if config.TRUST_REMOTE_CODE:
        tokenizer_config["trust_remote_code"] = True
    if config.EOS_TOKEN:
        tokenizer_config["eos_token"] = config.EOS_TOKEN

    model, tokenizer = _load_model_with_fallback(config.MODEL_NAME, tokenizer_config)
    print("Model loaded successfully!")
    yield
    # Cleanup on shutdown
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


def extract_text_from_content(
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
            ]
        ],
    ],
) -> str:
    """Extract text content from Claude-style content blocks"""
    if isinstance(content, str):
        return content

    text_parts = []
    for block in content:
        if hasattr(block, "type") and block.type == "text":
            text_parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))

    return " ".join(text_parts)


def extract_system_text(
    system: Optional[Union[str, List[SystemContent]]],
) -> Optional[str]:
    """Extract system text from system parameter"""
    if isinstance(system, str):
        return system
    elif isinstance(system, list):
        return " ".join([content.text for content in system])
    return None


def format_messages_for_llama(
    messages: List[Message], system: Optional[Union[str, List[SystemContent]]] = None
) -> str:
    """Convert Claude-style messages to Llama format"""
    formatted_messages = []

    # Add system message if provided
    system_text = extract_system_text(system)
    if system_text:
        formatted_messages.append({"role": "system", "content": system_text})

    # Add user messages
    for message in messages:
        content_text = extract_text_from_content(message.content)
        formatted_messages.append({"role": message.role, "content": content_text})

    # Apply chat template if available
    if tokenizer.chat_template is not None:
        try:
            result = tokenizer.apply_chat_template(
                formatted_messages, add_generation_prompt=True, tokenize=False
            )
            # Ensure we return a string, not tokens
            if isinstance(result, str):
                return result
        except Exception:
            # Fall through to manual formatting if template fails
            pass

    # Fallback formatting (used if no template or template fails)
    prompt = ""
    for msg in formatted_messages:
        if msg["role"] == "system":
            prompt += f"<|system|>\n{msg['content']}\n<|end|>\n"
        elif msg["role"] == "user":
            prompt += f"<|user|>\n{msg['content']}\n<|end|>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|>\n{msg['content']}\n<|end|>\n"
    prompt += "<|assistant|>\n"
    return prompt


def count_tokens(text: str) -> int:
    """Count tokens in text"""
    try:
        # MLX tokenizers often expect the text to be handled through their specific methods
        # First try the standard approach with proper string handling
        if isinstance(text, str) and text.strip():
            # For MLX, we may need to use a different approach
            # Try to get tokens using the tokenizer's __call__ method or encode
            try:
                # Some MLX tokenizers work better with this approach
                result = tokenizer(text, return_tensors=False, add_special_tokens=False)
                if isinstance(result, dict) and "input_ids" in result:
                    return len(result["input_ids"])
                elif hasattr(result, "__len__"):
                    return len(result)
            except (AttributeError, TypeError, ValueError):
                pass

            # Try direct encode without parameters
            try:
                encoded = tokenizer.encode(text)
                return (
                    len(encoded) if hasattr(encoded, "__len__") else len(list(encoded))
                )
            except (AttributeError, TypeError, ValueError):
                pass

            # Try with explicit string conversion and basic parameters
            try:
                tokens = tokenizer.encode(str(text), add_special_tokens=False)
                return len(tokens)
            except (AttributeError, TypeError, ValueError):
                pass

        # Final fallback: character-based estimation
        return max(1, len(str(text)) // 4)  # At least 1 token, ~4 chars per token

    except Exception as e:
        print(f"Token counting failed with error: {e}")
        return max(1, len(str(text)) // 4)  # Fallback estimation


@app.post("/v1/messages")
async def create_message(request: MessagesRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Format messages for Llama
        prompt = format_messages_for_llama(request.messages, request.system)

        # Count input tokens
        input_tokens = count_tokens(prompt)

        if request.stream:
            return StreamingResponse(
                stream_generate_response(request, prompt, input_tokens),
                media_type="text/event-stream",
            )
        else:
            return await generate_response(request, prompt, input_tokens)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(request: TokenCountRequest):
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Format messages for token counting
        prompt = format_messages_for_llama(request.messages, request.system)

        # Count tokens
        token_count = count_tokens(prompt)

        return {"input_tokens": token_count}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_response(request: MessagesRequest, prompt: str, input_tokens: int):
    """Generate non-streaming response"""
    raw_response_text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=request.max_tokens,
        verbose=config.VERBOSE,
    )

    if config.VERBOSE:
        print(f"DEBUG: Raw response (before filter): {repr(raw_response_text[:500])}")

    raw_output_tokens = count_tokens(raw_response_text)
    response_text = filter_thinking_text(raw_response_text).strip()

    if config.VERBOSE:
        print(f"DEBUG: Filtered response: {repr(response_text[:500])}")

    output_tokens = count_tokens(response_text)
    stop_reason = "max_tokens" if raw_output_tokens >= request.max_tokens else "end_turn"

    response = MessageResponse(
        id="msg_" + str(abs(hash(prompt)))[:8],
        content=[ContentBlockText(text=response_text)],
        model=request.model,
        stop_reason=stop_reason,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )

    return response


async def stream_generate_response(
    request: MessagesRequest, prompt: str, input_tokens: int
):
    """Generate streaming response"""
    response_id = "msg_" + str(abs(hash(prompt)))[:8]
    full_text = ""

    thinking_filter = ThinkingFilter()

    # Send message start event
    message_start = {
        "type": "message_start",
        "message": {
            "id": response_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": request.model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    # Send content block start
    content_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    yield f"event: content_block_start\ndata: {json.dumps(content_start)}\n\n"

    # Stream generation
    generated_token_count = 0
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=request.max_tokens,
    ):
        generated_token_count += 1
        filtered_text = thinking_filter.filter_chunk(response.text)

        if filtered_text:
            full_text += filtered_text

            # Send content block delta
            content_delta = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": filtered_text},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(content_delta)}\n\n"

    # Drain any remaining buffered text
    remaining = thinking_filter.filter_remaining()
    if remaining:
        full_text += remaining
        content_delta = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": remaining},
        }
        yield f"event: content_block_delta\ndata: {json.dumps(content_delta)}\n\n"

    # Count output tokens from the visible (filtered) text for usage reporting.
    output_tokens = count_tokens(full_text)

    # Determine stop reason from the raw token count (before filtering).
    # If the model generated as many tokens as max_tokens, the response was
    # truncated. Claude Code uses this to decide whether to send a follow-up.
    stop_reason = "max_tokens" if generated_token_count >= request.max_tokens else "end_turn"

    # Send content block stop
    content_stop = {"type": "content_block_stop", "index": 0}
    yield f"event: content_block_stop\ndata: {json.dumps(content_stop)}\n\n"

    # Send message delta with usage
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

    # Send message stop
    message_stop = {"type": "message_stop"}
    yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"


@app.get("/v1/models")
async def list_models():
    """Return list of available models (Anthropic API compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": config.API_MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "mlx",
            }
        ],
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/")
async def root():
    return {
        "message": "Claude Code MLX Proxy",
        "status": "running",
        "model_loaded": model is not None,
    }


if __name__ == "__main__":
    print(f"Starting Claude Code MLX Proxy on {config.HOST}:{config.PORT}")
    uvicorn.run(app, host=config.HOST, port=config.PORT)
