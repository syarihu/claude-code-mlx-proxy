import json
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Literal
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import PreTrainedTokenizerFast
from mlx_lm import load, stream_generate
from mlx_lm.utils import load_model, hf_repo_to_path, snapshot_download
from mlx_lm.tokenizer_utils import TokenizerWrapper
from config import config

# ---------------------------------------------------------------------------
# Global state (set during model loading)
# ---------------------------------------------------------------------------
model = None
tokenizer = None
tool_call_start_tag: Optional[str] = None
tool_call_end_tag: Optional[str] = None
tool_parser_fn = None

# ---------------------------------------------------------------------------
# Thinking tag definitions
# ---------------------------------------------------------------------------
THINKING_TAG_PAIRS: List[tuple[str, str]] = [
    ("<antThinking>", "</antThinking>"),
    ("<think>", "</think>"),
]

_THINKING_PATTERN = re.compile(
    "|".join(
        re.escape(o) + ".*?" + re.escape(c) for o, c in THINKING_TAG_PAIRS
    ),
    re.DOTALL,
)

THINKING_SYSTEM_INSTRUCTION = (
    "\n\nIMPORTANT: When you think or reason internally before responding, "
    "you MUST wrap ALL of your internal thinking inside <think>...</think> tags. "
    "Only content outside <think> tags will be visible to the user. "
    "Never output your internal reasoning without wrapping it in <think> tags."
)

# ---------------------------------------------------------------------------
# Streaming output parser — filters thinking, extracts tool calls
# ---------------------------------------------------------------------------


class OutputParser:
    """Stateful streaming parser.

    Processes token-by-token chunks and emits events:
      ("text", str)       – visible text to stream to the client
      ("tool_call", dict) – a complete parsed tool call
    """

    def __init__(self, start_in_thinking: bool = False):
        if start_in_thinking:
            self._state = "thinking"
            self._close_tag = "</think>"
        else:
            self._state = "normal"
            self._close_tag = ""
        self._buffer = ""
        self._tool_buffer = ""

    # -- public API --

    def feed(self, chunk: str) -> List[tuple]:
        events: List[tuple] = []
        self._buffer += chunk
        self._drain(events)
        return events

    def finish(self) -> List[tuple]:
        events: List[tuple] = []
        if self._state == "tool_call":
            parsed = _try_parse_tool(self._tool_buffer + self._buffer)
            if parsed:
                for tc in (parsed if isinstance(parsed, list) else [parsed]):
                    events.append(("tool_call", tc))
        elif self._state == "normal" and self._buffer:
            events.append(("text", self._buffer))
        self._buffer = ""
        self._tool_buffer = ""
        self._state = "normal"
        return events

    # -- internals --

    def _drain(self, events: List[tuple]):
        while True:
            if self._state == "thinking":
                idx = self._buffer.find(self._close_tag)
                if idx == -1:
                    self._buffer = ""
                    return
                self._buffer = self._buffer[idx + len(self._close_tag) :]
                self._state = "normal"
                self._close_tag = ""
                continue

            if self._state == "tool_call":
                if not tool_call_end_tag:
                    self._tool_buffer += self._buffer
                    self._buffer = ""
                    return
                idx = self._buffer.find(tool_call_end_tag)
                if idx == -1:
                    self._tool_buffer += self._buffer
                    self._buffer = ""
                    return
                self._tool_buffer += self._buffer[:idx]
                self._buffer = self._buffer[idx + len(tool_call_end_tag) :]
                self._state = "normal"
                parsed = _try_parse_tool(self._tool_buffer)
                if parsed:
                    for tc in (parsed if isinstance(parsed, list) else [parsed]):
                        events.append(("tool_call", tc))
                self._tool_buffer = ""
                continue

            # --- normal state ---
            earliest_pos = len(self._buffer)
            earliest_open = ""
            earliest_close = ""
            earliest_type = ""

            for open_t, close_t in THINKING_TAG_PAIRS:
                pos = self._buffer.find(open_t)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    earliest_open = open_t
                    earliest_close = close_t
                    earliest_type = "thinking"

            if tool_call_start_tag:
                pos = self._buffer.find(tool_call_start_tag)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    earliest_open = tool_call_start_tag
                    earliest_close = tool_call_end_tag
                    earliest_type = "tool_call"

            if earliest_pos == len(self._buffer):
                safe = self._safe_end()
                if safe > 0:
                    events.append(("text", self._buffer[:safe]))
                self._buffer = self._buffer[safe:]
                return

            if earliest_pos > 0:
                events.append(("text", self._buffer[:earliest_pos]))

            self._buffer = self._buffer[earliest_pos + len(earliest_open) :]

            if earliest_type == "thinking":
                self._state = "thinking"
                self._close_tag = earliest_close
            elif earliest_type == "tool_call":
                self._state = "tool_call"
                self._tool_buffer = ""

    def _safe_end(self) -> int:
        buf = self._buffer
        safe = len(buf)
        all_tags = [t for t, _ in THINKING_TAG_PAIRS]
        if tool_call_start_tag:
            all_tags.append(tool_call_start_tag)
        for tag in all_tags:
            for plen in range(1, len(tag)):
                if buf.endswith(tag[:plen]):
                    safe = min(safe, len(buf) - plen)
                    break
        return safe


def _try_parse_tool(text: str):
    if tool_parser_fn is None:
        return None
    try:
        return tool_parser_fn(text.strip(), None)
    except Exception as e:
        if config.VERBOSE:
            print(f"Warning: tool call parse failed: {e}")
        return None


def parse_complete_output(raw_text: str) -> tuple[str, list[dict]]:
    """One-shot parse for non-streaming: returns (visible_text, tool_calls)."""
    text = _THINKING_PATTERN.sub("", raw_text)

    tool_calls: list[dict] = []
    if tool_call_start_tag and tool_call_end_tag and tool_parser_fn:
        tc_pattern = re.compile(
            re.escape(tool_call_start_tag)
            + "(.*?)"
            + re.escape(tool_call_end_tag),
            re.DOTALL,
        )
        for m in tc_pattern.finditer(text):
            try:
                parsed = tool_parser_fn(m.group(1), None)
                if parsed:
                    if isinstance(parsed, list):
                        tool_calls.extend(parsed)
                    else:
                        tool_calls.append(parsed)
            except Exception:
                pass
        text = tc_pattern.sub("", text)

    return text.strip(), tool_calls


# ---------------------------------------------------------------------------
# Pydantic models (Anthropic Messages API)
# ---------------------------------------------------------------------------


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
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    model: str
    stop_reason: str = "end_turn"
    stop_sequence: Optional[str] = None
    usage: Usage


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model_with_fallback(model_name: str, tokenizer_config: dict):
    try:
        return load(model_name, tokenizer_config=tokenizer_config)
    except ValueError as e:
        if "TokenizersBackend" not in str(e):
            raise
        print(
            "Warning: Failed to load tokenizer via AutoTokenizer "
            "(TokenizersBackend not found). Attempting fallback using "
            "PreTrainedTokenizerFast..."
        )

    model_path = Path(hf_repo_to_path(model_name))
    if not model_path.exists():
        model_path = Path(snapshot_download(model_name))
    mlx_model, mlx_config = load_model(model_path)

    hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        str(model_path), **tokenizer_config
    )
    eos_token_id = mlx_config.get("eos_token_id")
    wrapped = TokenizerWrapper(hf_tokenizer, eos_token_ids=eos_token_id)
    return mlx_model, wrapped


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, tool_call_start_tag, tool_call_end_tag, tool_parser_fn
    print(f"Loading MLX model: {config.MODEL_NAME}")

    tokenizer_config: dict = {}
    if config.TRUST_REMOTE_CODE:
        tokenizer_config["trust_remote_code"] = True
    if config.EOS_TOKEN:
        tokenizer_config["eos_token"] = config.EOS_TOKEN

    model, tokenizer = _load_model_with_fallback(config.MODEL_NAME, tokenizer_config)

    if getattr(tokenizer, "has_tool_calling", False):
        tool_call_start_tag = getattr(tokenizer, "_tool_call_start", None)
        tool_call_end_tag = getattr(tokenizer, "_tool_call_end", None)
        tool_parser_fn = getattr(tokenizer, "tool_parser", None)
        print(
            f"Tool calling enabled — "
            f"start={tool_call_start_tag!r}  end={tool_call_end_tag!r}"
        )
    else:
        print("Tool calling not detected for this model")

    # Register stop sequences as EOS tokens so mlx-lm stops generation
    # BEFORE the token is yielded.  This prevents partial stop-sequence
    # text from leaking into the output.
    original_eos = set(tokenizer.eos_token_ids or [])
    added_eos: list[str] = []
    for seq in config.STOP_SEQUENCES:
        try:
            ids = tokenizer.encode(seq, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in original_eos:
                original_eos.add(ids[0])
                added_eos.append(f"{seq!r}→{ids[0]}")
        except Exception:
            pass
    if added_eos:
        tokenizer.eos_token_ids = list(original_eos)
        print(f"Added stop-sequence EOS tokens: {', '.join(added_eos)}")
    print(f"EOS token IDs: {tokenizer.eos_token_ids}")

    print("Model loaded successfully!")
    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Format conversion: Anthropic ↔ OpenAI / chat-template
# ---------------------------------------------------------------------------


def extract_system_text(
    system: Optional[Union[str, List[SystemContent]]],
) -> Optional[str]:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return " ".join(c.text for c in system)
    return None


def anthropic_tools_to_openai(tools: List[Tool]) -> List[dict]:
    result = []
    for t in tools:
        if config.TOOL_MODE == "slim":
            props = t.input_schema.get("properties", {})
            required = t.input_schema.get("required", [])
            slim_params: Dict[str, Any] = {
                "type": "object",
                "properties": {
                    k: {"type": v.get("type", "string")}
                    for k, v in props.items()
                },
            }
            if required:
                slim_params["required"] = required
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": (t.description or "")[:200],
                        "parameters": slim_params,
                    },
                }
            )
        else:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.input_schema,
                    },
                }
            )
    return result


def _tool_result_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict) and content.get("type") == "text":
        return content.get("text", "")
    return str(content) if content else ""


def anthropic_messages_to_chat(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]] = None,
) -> List[dict]:
    """Convert Anthropic-format messages to OpenAI-style chat messages."""
    result: List[dict] = []

    system_text = extract_system_text(system)
    if config.INJECT_THINKING_PROMPT:
        if system_text:
            system_text += THINKING_SYSTEM_INSTRUCTION
        else:
            system_text = THINKING_SYSTEM_INSTRUCTION.strip()
    if system_text:
        result.append({"role": "system", "content": system_text})

    for msg in messages:
        if isinstance(msg.content, str):
            result.append({"role": msg.role, "content": msg.content})
            continue

        if msg.role == "assistant":
            text_parts: List[str] = []
            tc_list: List[dict] = []
            for block in msg.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    text_parts.append(block.text)
                elif btype == "tool_use":
                    tc_list.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": (
                                    block.input
                                    if isinstance(block.input, dict)
                                    else {}
                                ),
                            },
                        }
                    )
            entry: dict = {"role": "assistant"}
            text = "\n".join(text_parts) if text_parts else None
            if text:
                entry["content"] = text
            if tc_list:
                entry["tool_calls"] = tc_list
            if not text and not tc_list:
                entry["content"] = ""
            result.append(entry)

        elif msg.role == "user":
            text_parts = []
            tool_results: List[dict] = []
            for block in msg.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    text_parts.append(block.text)
                elif btype == "tool_result":
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": _tool_result_to_str(block.content),
                        }
                    )
            for tr in tool_results:
                result.append(tr)
            if text_parts:
                result.append({"role": "user", "content": "\n".join(text_parts)})

    return result


def _strip_tool_fields(chat_messages: List[dict]) -> List[dict]:
    """Remove tool_calls / role:'tool' so the template works without tools=."""
    cleaned: List[dict] = []
    for msg in chat_messages:
        if msg["role"] == "tool":
            name_hint = msg.get("name", "")
            content = msg.get("content", "")
            label = f"[Tool result{': ' + name_hint if name_hint else ''}] " if name_hint else "[Tool result] "
            cleaned.append({"role": "user", "content": label + content})
            continue
        if "tool_calls" in msg:
            new_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            parts: List[str] = []
            if new_msg.get("content"):
                parts.append(new_msg["content"])
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                parts.append(f"[Calling tool: {fn.get('name', '?')}({fn.get('arguments', '')})]")
            new_msg["content"] = "\n".join(parts) if parts else ""
            cleaned.append(new_msg)
            continue
        cleaned.append(msg)
    return cleaned


def format_prompt(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]] = None,
    tools: Optional[List[Tool]] = None,
) -> str:
    """Build prompt string using the model's chat template."""
    chat_messages = anthropic_messages_to_chat(messages, system)
    if config.TOOL_MODE == "none":
        openai_tools = None
    else:
        openai_tools = anthropic_tools_to_openai(tools) if tools else None

    if tokenizer.chat_template is not None:
        # Try with tools first
        if openai_tools:
            try:
                result = tokenizer.apply_chat_template(
                    chat_messages,
                    tools=openai_tools,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                if isinstance(result, str):
                    if config.VERBOSE:
                        print(f"DEBUG prompt (last 500): ...{result[-500:]}")
                    return result
            except Exception as e:
                print(f"Warning: chat template failed WITH tools: {e}")
                print("Retrying without tools...")

        # Try without tools — strip tool_calls/role:tool so template won't choke
        plain_messages = _strip_tool_fields(chat_messages)
        try:
            result = tokenizer.apply_chat_template(
                plain_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            if isinstance(result, str):
                if openai_tools:
                    print(
                        f"Warning: tools excluded from prompt "
                        f"({len(openai_tools)} tools dropped)"
                    )
                if config.VERBOSE:
                    print(f"DEBUG prompt (last 500): ...{result[-500:]}")
                return result
        except Exception as e:
            print(f"ERROR: chat template failed completely: {e}")

    # Last-resort fallback — should rarely be reached
    print("Warning: using generic fallback prompt format")
    prompt = ""
    for msg in chat_messages:
        role = msg["role"]
        content = msg.get("content", "")
        prompt += f"<|{role}|>\n{content}\n<|end|>\n"
    prompt += "<|assistant|>\n"
    return prompt


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    try:
        if isinstance(text, str) and text.strip():
            try:
                result = tokenizer(
                    text, return_tensors=False, add_special_tokens=False
                )
                if isinstance(result, dict) and "input_ids" in result:
                    return len(result["input_ids"])
                if hasattr(result, "__len__"):
                    return len(result)
            except (AttributeError, TypeError, ValueError):
                pass
            try:
                encoded = tokenizer.encode(text)
                return len(encoded) if hasattr(encoded, "__len__") else len(list(encoded))
            except (AttributeError, TypeError, ValueError):
                pass
        return max(1, len(str(text)) // 4)
    except Exception:
        return max(1, len(str(text)) // 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prompt_starts_in_thinking(prompt: str) -> bool:
    """Check if the prompt's generation starts inside a <think> block."""
    return prompt.rstrip().endswith("<think>")


def _make_tool_use_id() -> str:
    return "toolu_" + uuid.uuid4().hex[:24]


def _make_msg_id(prompt: str) -> str:
    return "msg_" + uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/messages")
async def create_message(request: MessagesRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        prompt = format_prompt(request.messages, request.system, request.tools)
        input_tokens = count_tokens(prompt)
        starts_thinking = _prompt_starts_in_thinking(prompt)
        n_tools = len(request.tools) if request.tools else 0
        print(
            f"[/v1/messages] input_tokens={input_tokens}, tools={n_tools}, "
            f"tool_mode={config.TOOL_MODE}, stream={request.stream}, "
            f"starts_thinking={starts_thinking}"
        )

        if request.stream:
            return StreamingResponse(
                stream_generate_response(
                    request, prompt, input_tokens, starts_thinking
                ),
                media_type="text/event-stream",
            )
        return await generate_response(
            request, prompt, input_tokens, starts_thinking
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(request: TokenCountRequest):
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        prompt = format_prompt(request.messages, request.system, request.tools)
        return {"input_tokens": count_tokens(prompt)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------


async def generate_response(
    request: MessagesRequest,
    prompt: str,
    input_tokens: int,
    starts_thinking: bool = False,
):
    raw_text = ""
    finish_reason = None
    for resp in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=request.max_tokens
    ):
        raw_text += resp.text
        finish_reason = resp.finish_reason
        hit_stop = False
        for seq in config.STOP_SEQUENCES:
            pos = raw_text.find(seq)
            if pos != -1:
                raw_text = raw_text[:pos]
                hit_stop = True
                break
        if hit_stop:
            break

    if starts_thinking:
        raw_text = "<think>" + raw_text

    if config.VERBOSE:
        print(f"DEBUG raw (first 500): {repr(raw_text[:500])}")

    visible_text, tool_calls = parse_complete_output(raw_text)

    if config.VERBOSE:
        print(f"DEBUG visible: {repr(visible_text[:300])}")
        print(f"DEBUG tool_calls: {tool_calls}")
        print(f"DEBUG finish_reason={finish_reason}")

    content: List[Union[ContentBlockText, ContentBlockToolUse]] = []
    if visible_text:
        content.append(ContentBlockText(text=visible_text))
    for tc in tool_calls:
        content.append(
            ContentBlockToolUse(
                id=_make_tool_use_id(),
                name=tc["name"],
                input=tc.get("arguments", {}),
            )
        )
    if not content:
        content.append(ContentBlockText(text=""))

    if tool_calls:
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    output_tokens = count_tokens(raw_text)

    return MessageResponse(
        id=_make_msg_id(prompt),
        content=content,
        model=request.model,
        stop_reason=stop_reason,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# ---------------------------------------------------------------------------
# Streaming response
# ---------------------------------------------------------------------------


async def stream_generate_response(
    request: MessagesRequest,
    prompt: str,
    input_tokens: int,
    starts_thinking: bool = False,
):
    response_id = _make_msg_id(prompt)
    parser = OutputParser(start_in_thinking=starts_thinking)
    full_text = ""
    has_tool_calls = False

    # --- message_start ---
    yield _sse(
        "message_start",
        {
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
        },
    )

    block_index = 0
    text_block_open = False

    def _ensure_text_block():
        nonlocal block_index, text_block_open
        if not text_block_open:
            text_block_open = True
            return _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        return None

    def _close_text_block():
        nonlocal block_index, text_block_open
        if text_block_open:
            text_block_open = False
            msg = _sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": block_index},
            )
            block_index += 1
            return msg
        return None

    # --- stream tokens ---
    finish_reason = None
    raw_tail = ""  # rolling window for stop-sequence detection
    generation_stopped = False
    for resp in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=request.max_tokens
    ):
        finish_reason = resp.finish_reason
        if config.VERBOSE:
            print(f"DEBUG chunk: {repr(resp.text)}", flush=True)

        # Stop-sequence detection on the raw (unfiltered) output.
        raw_tail += resp.text
        if len(raw_tail) > 200:
            raw_tail = raw_tail[-200:]
        for seq in config.STOP_SEQUENCES:
            if seq in raw_tail:
                if config.VERBOSE:
                    print(f"DEBUG: stop sequence {seq!r} detected, stopping")
                generation_stopped = True
                break
        if generation_stopped:
            break

        for event_type, event_data in parser.feed(resp.text):
            if event_type == "text":
                full_text += event_data
                msg = _ensure_text_block()
                if msg:
                    yield msg
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": event_data},
                    },
                )

            elif event_type == "tool_call":
                has_tool_calls = True
                msg = _close_text_block()
                if msg:
                    yield msg
                for msg in _emit_tool_use_block(block_index, event_data):
                    yield msg
                block_index += 1

    # --- flush remaining ---
    for event_type, event_data in parser.finish():
        if event_type == "text":
            full_text += event_data
            msg = _ensure_text_block()
            if msg:
                yield msg
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": event_data},
                },
            )
        elif event_type == "tool_call":
            has_tool_calls = True
            msg = _close_text_block()
            if msg:
                yield msg
            for msg in _emit_tool_use_block(block_index, event_data):
                yield msg
            block_index += 1

    # --- close last open block ---
    msg = _close_text_block()
    if msg:
        yield msg

    # --- stop reason ---
    if has_tool_calls:
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    output_tokens = count_tokens(full_text)
    if config.VERBOSE:
        print(
            f"DEBUG: finish_reason={finish_reason} → stop_reason={stop_reason}, "
            f"output_tokens={output_tokens}",
            flush=True,
        )

    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _emit_tool_use_block(index: int, tc: dict):
    tool_id = _make_tool_use_id()
    name = tc.get("name", "unknown")
    arguments = tc.get("arguments", {})

    yield _sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": {},
            },
        },
    )
    yield _sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps(arguments),
            },
        },
    )
    yield _sse(
        "content_block_stop",
        {"type": "content_block_stop", "index": index},
    )


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models():
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
