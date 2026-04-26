import json
import uuid
from typing import List, Dict, Any, Optional, Union, Literal
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from config import config


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
# App lifecycle
# ---------------------------------------------------------------------------

http_client: Optional[httpx.AsyncClient] = None
external_http_client: Optional[httpx.AsyncClient] = None
_backend_model_name: Optional[str] = None


async def _get_backend_model_name() -> str:
    global _backend_model_name
    if _backend_model_name is not None:
        return _backend_model_name
    # Use forced model name if set (e.g. local path for MLX backends)
    if config.MODEL_NAME:
        _backend_model_name = config.MODEL_NAME
        print(f"Using forced model name: {_backend_model_name}")
        return _backend_model_name
    if http_client is None:
        return "default"
    try:
        resp = await http_client.get("models", timeout=10.0)
        if resp.status_code == 200:
            payload = resp.json()
            models = payload.get("data", []) if isinstance(payload, dict) else []
            if models and isinstance(models[0], dict):
                model_id = models[0].get("id")
                if model_id:
                    _backend_model_name = model_id
                    print(f"Backend model: {_backend_model_name}")
                    return _backend_model_name
    except Exception as e:
        print(f"Warning: could not query backend models: {e}")
    return "default"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, external_http_client
    base_url = (
        config.MLX_SERVER_URL
        if config.MLX_SERVER_URL.endswith("/")
        else f"{config.MLX_SERVER_URL}/"
    )
    http_client = httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0),
    )
    external_http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
    )
    await _get_backend_model_name()
    print(f"Proxy started — forwarding to {base_url}")
    yield
    await http_client.aclose()
    await external_http_client.aclose()
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Format conversion: Anthropic → OpenAI
# ---------------------------------------------------------------------------


def _extract_system_text(
    system: Optional[Union[str, List[SystemContent]]],
) -> Optional[str]:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return " ".join(c.text for c in system)
    return None


SLIM_TOOL_GUIDE = """\
# Tool usage guide

## Agent
- subagent_type options: "Explore" (codebase search, find files/keywords), "Plan" (architecture design), "claude-code-guide" (Claude Code questions), "general-purpose" (multi-step tasks), "statusline-setup"
- For broad codebase exploration (>3 queries), use Agent with subagent_type="Explore"
- If target is already known, use Read or Bash(grep) directly instead
- Always include description and prompt params; prompt should be self-contained

## Bash
- Use Read/Edit/Write tools instead of cat/sed/echo
- Git: always create NEW commits (never amend unless asked), never force push, never --no-verify
- Git commit format: use HEREDOC — git commit -m "$(cat <<'EOF'\\nmessage\\nEOF\\n)"
- PR: use gh pr create with --title and --body (HEREDOC)
- Use run_in_background for long-running commands

## Edit
- Must Read the file first before editing
- old_string must be unique in the file; add surrounding context if not
- Use replace_all=true to rename across file

## Read
- Use absolute paths; supports images, PDFs (use pages param for large ones), notebooks

## Write
- Must Read existing files first; prefer Edit for modifications
- Only use for new files or complete rewrites

## EnterPlanMode
- Use proactively for non-trivial implementation tasks before writing code
- Skip for simple fixes, typos, single-line changes

## Skill
- When user types "/<name>", invoke via Skill tool with that name
- Only use skills listed in system-reminder messages
"""


def _anthropic_tools_to_openai(tools: List[Tool]) -> List[dict]:
    result = []
    for t in tools:
        if config.TOOL_MODE == "slim":
            props = t.input_schema.get("properties", {})
            required = t.input_schema.get("required", [])
            slim_props: Dict[str, Any] = {}
            for k, v in props.items():
                sp: Dict[str, Any] = {"type": v.get("type", "string")}
                if "enum" in v:
                    sp["enum"] = v["enum"]
                desc = v.get("description", "")
                if desc:
                    sp["description"] = desc[:100]
                if v.get("type") == "array" and "items" in v:
                    items = v["items"]
                    slim_items: Dict[str, Any] = {"type": items.get("type", "string")}
                    if "enum" in items:
                        slim_items["enum"] = items["enum"]
                    sp["items"] = slim_items
                slim_props[k] = sp
            slim_params: Dict[str, Any] = {
                "type": "object",
                "properties": slim_props,
            }
            if required:
                slim_params["required"] = required
            desc = t.description or ""
            if t.name not in config.TOOL_FULL_DESC_NAMES and config.TOOL_DESC_LIMIT > 0:
                desc = desc[:config.TOOL_DESC_LIMIT]
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": desc,
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


def _anthropic_messages_to_openai(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]] = None,
) -> List[dict]:
    """Convert Anthropic-format messages to OpenAI chat messages."""
    result: List[dict] = []

    system_text = _extract_system_text(system)
    if config.TOOL_MODE == "slim":
        system_text = f"{system_text}\n\n{SLIM_TOOL_GUIDE}" if system_text else SLIM_TOOL_GUIDE
    if config.RESPONSE_LANGUAGE:
        lang_instruction = f"You must always respond in {config.RESPONSE_LANGUAGE}."
        system_text = f"{system_text}\n\n{lang_instruction}" if system_text else lang_instruction
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
                                    json.dumps(block.input)
                                    if isinstance(block.input, dict)
                                    else "{}"
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
            text_parts: List[str] = []

            def flush_user_text() -> None:
                if text_parts:
                    result.append({"role": "user", "content": "\n".join(text_parts)})
                    text_parts.clear()

            for block in msg.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    text_parts.append(block.text)
                elif btype == "tool_result":
                    flush_user_text()
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": _tool_result_to_str(block.content),
                        }
                    )

            flush_user_text()

    return result


# ---------------------------------------------------------------------------
# Build OpenAI request body
# ---------------------------------------------------------------------------


async def _build_openai_request(request: MessagesRequest) -> dict:
    chat_messages = _anthropic_messages_to_openai(request.messages, request.system)
    model_name = await _get_backend_model_name()

    body: dict = {
        "model": model_name,
        "messages": chat_messages,
        "max_tokens": request.max_tokens,
        "stream": request.stream or False,
    }

    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.stop_sequences:
        body["stop"] = request.stop_sequences

    if config.TOOL_MODE != "none" and request.tools:
        body["tools"] = _anthropic_tools_to_openai(request.tools)
        if request.tool_choice:
            tc_type = request.tool_choice.get("type")
            if tc_type == "auto":
                body["tool_choice"] = "auto"
            elif tc_type == "any":
                body["tool_choice"] = "required"
            elif tc_type == "tool":
                tool_name = request.tool_choice.get("name")
                if not isinstance(tool_name, str) or not tool_name.strip():
                    raise HTTPException(
                        status_code=400,
                        detail="tool_choice.name is required when tool_choice.type is 'tool'.",
                    )
                body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

    if request.stream:
        body["stream_options"] = {"include_usage": True}

    return body


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_use_id() -> str:
    return "toolu_" + uuid.uuid4().hex[:24]


def _make_msg_id() -> str:
    return "msg_" + uuid.uuid4().hex[:8]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _map_finish_reason(fr: Optional[str]) -> str:
    if fr == "tool_calls":
        return "tool_use"
    if fr == "length":
        return "max_tokens"
    return "end_turn"


# ---------------------------------------------------------------------------
# Non-streaming response converter
# ---------------------------------------------------------------------------


def _convert_response(openai_data: dict, request: MessagesRequest) -> MessageResponse:
    choice = openai_data.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = openai_data.get("usage", {})

    content: List[Union[ContentBlockText, ContentBlockToolUse]] = []

    text = message.get("content")
    if text:
        content.append(ContentBlockText(text=text))

    for tc in message.get("tool_calls", []):
        fn = tc.get("function", {})
        try:
            arguments = json.loads(fn.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        content.append(
            ContentBlockToolUse(
                id=tc.get("id") or _make_tool_use_id(),
                name=fn.get("name", "unknown"),
                input=arguments,
            )
        )

    if not content:
        content.append(ContentBlockText(text=""))

    return MessageResponse(
        id=_make_msg_id(),
        content=content,
        model=request.model,
        stop_reason=_map_finish_reason(choice.get("finish_reason")),
        usage=Usage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        ),
    )


# ---------------------------------------------------------------------------
# Streaming response converter
# ---------------------------------------------------------------------------


async def _stream_response(
    response: httpx.Response, request: MessagesRequest
):
    msg_id = _make_msg_id()

    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": request.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    block_index = 0
    text_block_open = False
    finish_reason = None
    input_tokens = 0
    output_tokens = 0

    def _open_text_block():
        nonlocal text_block_open
        if text_block_open:
            return None
        text_block_open = True
        return _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": block_index,
                "content_block": {"type": "text", "text": ""},
            },
        )

    def _close_text_block():
        nonlocal block_index, text_block_open
        if not text_block_open:
            return None
        text_block_open = False
        msg = _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": block_index},
        )
        block_index += 1
        return msg

    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            break

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if "usage" in data:
            u = data["usage"]
            input_tokens = u.get("prompt_tokens", input_tokens)
            output_tokens = u.get("completion_tokens", output_tokens)

        choice = (data.get("choices") or [{}])[0]
        delta = choice.get("delta", {})
        fr = choice.get("finish_reason")
        if fr:
            finish_reason = fr

        if config.VERBOSE:
            keys = [k for k in delta if k != "role"]
            if keys:
                preview = {k: (delta[k][:80] if isinstance(delta[k], str) else delta[k]) for k in keys}
                print(f"  delta: {preview}", flush=True)

        # Text content
        text = delta.get("content")
        if text:
            msg = _open_text_block()
            if msg:
                yield msg
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )

        # Tool calls (sent as complete objects by mlx-lm)
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            msg = _close_text_block()
            if msg:
                yield msg
            for tc in tool_calls:
                fn = tc.get("function", {})
                try:
                    arguments = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.get("id") or _make_tool_use_id(),
                            "name": fn.get("name", "unknown"),
                            "input": {},
                        },
                    },
                )
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(arguments),
                        },
                    },
                )
                yield _sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": block_index},
                )
                block_index += 1

    # Close any remaining text block
    msg = _close_text_block()
    if msg:
        yield msg

    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": _map_finish_reason(finish_reason),
                "stop_sequence": None,
            },
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Web search (Tavily) interception
# ---------------------------------------------------------------------------

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
WEB_SEARCH_TOOL_NAME = "WebSearch"


def _has_web_search_tool(tools: Optional[List[Tool]]) -> bool:
    if not tools:
        return False
    return any(t.name == WEB_SEARCH_TOOL_NAME for t in tools)


def _format_tavily_results(data: dict) -> str:
    parts: List[str] = []
    answer = data.get("answer")
    if answer:
        parts.append(f"Answer: {answer}")
    results = data.get("results") or []
    for i, r in enumerate(results, 1):
        title = r.get("title") or "Untitled"
        url = r.get("url") or ""
        content = (r.get("content") or "").strip()
        parts.append(f"[{i}] {title}\nURL: {url}\n{content}")
    return "\n\n".join(parts) if parts else "No results found."


async def _tavily_search(
    query: str,
    max_results: int,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> str:
    if not config.TAVILY_API_KEY:
        return "Web search is not configured on this proxy."
    if not query or not query.strip():
        return "Web search query was empty."
    if external_http_client is None:
        return "Web search is temporarily unavailable."

    body: Dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "include_answer": True,
    }
    if include_domains:
        body["include_domains"] = include_domains
    if exclude_domains:
        body["exclude_domains"] = exclude_domains

    headers = {
        "Authorization": f"Bearer {config.TAVILY_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = await external_http_client.post(
            TAVILY_SEARCH_URL, json=body, headers=headers
        )
    except Exception as e:
        if config.VERBOSE:
            print(f"[web_search] Tavily request raised: {e}")
        return "Web search failed due to a network error."

    if resp.status_code != 200:
        if config.VERBOSE:
            print(
                f"[web_search] Tavily returned HTTP {resp.status_code}: "
                f"{resp.text[:200]}"
            )
        return "Web search failed due to an upstream service error."

    try:
        return _format_tavily_results(resp.json())
    except Exception as e:
        if config.VERBOSE:
            print(f"[web_search] Failed to parse Tavily response: {e}")
        return "Web search returned an unexpected response."


async def _run_web_search_loop(openai_body: dict) -> dict:
    """Iteratively resolve WebSearch tool calls inside the proxy.

    The model is invoked non-streaming; whenever every tool call in the
    response is WebSearch, the proxy executes them via Tavily and feeds the
    results back into the conversation. The loop returns the first response
    that contains no WebSearch tool calls (or contains a mix of WebSearch
    with other tool calls — those are forwarded to the client untouched so
    Claude Code can execute the non-WebSearch ones).
    """
    current_body = dict(openai_body)
    current_body["stream"] = False
    current_body.pop("stream_options", None)
    current_body["messages"] = list(openai_body.get("messages") or [])

    for _ in range(max(1, config.WEB_SEARCH_MAX_ITERATIONS)):
        resp = await http_client.post("chat/completions", json=current_body)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        tool_calls = message.get("tool_calls") or []

        web_search_calls = [
            tc for tc in tool_calls
            if (tc.get("function") or {}).get("name") == WEB_SEARCH_TOOL_NAME
        ]
        if not web_search_calls:
            return data

        other_calls = [tc for tc in tool_calls if tc not in web_search_calls]
        if other_calls:
            # Mixed tool calls: bail out and let the client handle the rest.
            # (WebSearch results would otherwise be lost without a way to
            # re-inject them alongside the non-WebSearch tool calls.)
            return data

        # All tool calls are WebSearch — execute them and continue the loop.
        assistant_msg: Dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
        if message.get("content"):
            assistant_msg["content"] = message["content"]
        current_body["messages"].append(assistant_msg)

        for tc in web_search_calls:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            result_text = await _tavily_search(
                query=args.get("query", ""),
                max_results=config.WEB_SEARCH_MAX_RESULTS,
                include_domains=args.get("allowed_domains"),
                exclude_domains=args.get("blocked_domains"),
            )
            if config.VERBOSE:
                print(
                    f"[web_search] query={args.get('query', '')!r} "
                    f"chars={len(result_text)}"
                )
            current_body["messages"].append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "content": result_text,
            })

    raise HTTPException(
        status_code=500,
        detail="WebSearch loop exceeded WEB_SEARCH_MAX_ITERATIONS.",
    )


async def _stream_with_web_search(openai_body: dict, request: MessagesRequest):
    """Stream the backend response, intercepting WebSearch tool calls inline.

    Text deltas are forwarded to the client as they arrive, so a request that
    happens to never call WebSearch keeps native streaming TTFB. WebSearch
    tool calls are accumulated, executed internally via Tavily, and the
    follow-up iteration's tokens continue streaming as more text deltas in
    the same Anthropic message — the WebSearch round-trip is invisible to
    the client.

    If the model emits a mix of WebSearch and other tool calls in a single
    response, the proxy bails out and forwards every tool call to the client
    (mirroring the non-streaming fall-back path).
    """
    msg_id = _make_msg_id()

    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": request.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    block_index = 0
    text_block_open = False
    total_input_tokens = 0
    total_output_tokens = 0
    final_finish_reason: Optional[str] = None
    completed = False

    current_body = dict(openai_body)
    current_body["stream"] = True
    current_body["stream_options"] = {"include_usage": True}
    current_body["messages"] = list(openai_body.get("messages") or [])

    def emit_tool_use(tc: dict, idx: int):
        fn = tc.get("function") or {}
        try:
            arguments = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        events = [
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc.get("id") or _make_tool_use_id(),
                        "name": fn.get("name", "unknown"),
                        "input": {},
                    },
                },
            ),
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(arguments),
                    },
                },
            ),
            _sse("content_block_stop", {"type": "content_block_stop", "index": idx}),
        ]
        return events

    for _ in range(max(1, config.WEB_SEARCH_MAX_ITERATIONS)):
        req = http_client.build_request(
            "POST", "chat/completions", json=current_body
        )
        resp = await http_client.send(req, stream=True)
        if resp.status_code != 200:
            body = await resp.aread()
            await resp.aclose()
            raise HTTPException(status_code=resp.status_code, detail=body.decode())

        accumulated_text_parts: List[str] = []
        accumulated_tool_calls: List[dict] = []
        finish_reason: Optional[str] = None
        input_tokens = 0
        output_tokens = 0

        try:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if data.get("usage"):
                    u = data["usage"]
                    input_tokens = u.get("prompt_tokens", input_tokens)
                    output_tokens = u.get("completion_tokens", output_tokens)

                choice = (data.get("choices") or [{}])[0]
                delta = choice.get("delta", {}) or {}
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr

                text = delta.get("content")
                if text:
                    accumulated_text_parts.append(text)
                    if not text_block_open:
                        yield _sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                        text_block_open = True
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )

                tcs = delta.get("tool_calls")
                if tcs:
                    accumulated_tool_calls.extend(tcs)
        finally:
            await resp.aclose()

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        final_finish_reason = finish_reason

        web_search_calls = [
            tc for tc in accumulated_tool_calls
            if (tc.get("function") or {}).get("name") == WEB_SEARCH_TOOL_NAME
        ]
        non_ws_calls = [
            tc for tc in accumulated_tool_calls if tc not in web_search_calls
        ]

        # Decide what to emit and whether to continue the loop.
        if not web_search_calls:
            tcs_to_emit = non_ws_calls
        elif non_ws_calls:
            # Mixed — pass everything through so Claude Code handles it.
            tcs_to_emit = accumulated_tool_calls
        else:
            tcs_to_emit = None  # internal-only iteration

        if tcs_to_emit is not None:
            if text_block_open:
                yield _sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": block_index},
                )
                block_index += 1
                text_block_open = False
            for tc in tcs_to_emit:
                for event in emit_tool_use(tc, block_index):
                    yield event
                block_index += 1
            completed = True
            break

        # All accumulated tool calls are WebSearch — execute internally and loop.
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "tool_calls": accumulated_tool_calls,
        }
        if accumulated_text_parts:
            assistant_msg["content"] = "".join(accumulated_text_parts)
        current_body["messages"].append(assistant_msg)

        for tc in web_search_calls:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            result_text = await _tavily_search(
                query=args.get("query", ""),
                max_results=config.WEB_SEARCH_MAX_RESULTS,
                include_domains=args.get("allowed_domains"),
                exclude_domains=args.get("blocked_domains"),
            )
            if config.VERBOSE:
                print(
                    f"[web_search] query={args.get('query', '')!r} "
                    f"chars={len(result_text)}"
                )
            current_body["messages"].append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "content": result_text,
            })

    if not completed and config.VERBOSE:
        print("[web_search] WEB_SEARCH_MAX_ITERATIONS exceeded; closing stream.")

    if text_block_open:
        yield _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": block_index},
        )
        block_index += 1
        text_block_open = False

    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": _map_finish_reason(final_finish_reason),
                "stop_sequence": None,
            },
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
            },
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/messages")
async def create_message(request: MessagesRequest):
    if http_client is None:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    openai_body = await _build_openai_request(request)
    n_tools = len(request.tools) if request.tools else 0
    use_web_search_loop = bool(
        config.TAVILY_API_KEY and _has_web_search_tool(request.tools)
    )

    if config.VERBOSE:
        print(
            f"[/v1/messages] tools={n_tools}, tool_mode={config.TOOL_MODE}, "
            f"stream={request.stream}, web_search_loop={use_web_search_loop}"
        )
        if openai_body.get("tools"):
            print("[tools] " + json.dumps(openai_body["tools"], ensure_ascii=False, indent=2))

    try:
        if use_web_search_loop:
            if request.stream:
                return StreamingResponse(
                    _stream_with_web_search(openai_body, request),
                    media_type="text/event-stream",
                )
            data = await _run_web_search_loop(openai_body)
            return _convert_response(data, request)

        if request.stream:
            req = http_client.build_request(
                "POST", "chat/completions", json=openai_body
            )
            resp = await http_client.send(req, stream=True)
            if resp.status_code != 200:
                body = await resp.aread()
                await resp.aclose()
                raise HTTPException(
                    status_code=resp.status_code, detail=body.decode()
                )

            async def stream_wrapper():
                try:
                    async for chunk in _stream_response(resp, request):
                        yield chunk
                finally:
                    await resp.aclose()

            return StreamingResponse(
                stream_wrapper(), media_type="text/event-stream"
            )

        # Non-streaming
        resp = await http_client.post("chat/completions", json=openai_body)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code, detail=resp.text
            )
        return _convert_response(resp.json(), request)

    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to mlx-lm server at {config.MLX_SERVER_URL}",
        )


@app.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(request: TokenCountRequest):
    text = json.dumps(
        _anthropic_messages_to_openai(request.messages, request.system)
    )
    return {"input_tokens": max(1, len(text) // 4)}


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
    backend_ok = False
    if http_client:
        try:
            resp = await http_client.get("models", timeout=5.0)
            backend_ok = resp.status_code == 200
        except Exception:
            pass
    return {"status": "healthy" if backend_ok else "degraded", "backend": backend_ok}


@app.get("/")
async def root():
    return {
        "message": "Claude Code MLX Proxy",
        "status": "running",
        "backend": config.MLX_SERVER_URL,
    }


if __name__ == "__main__":
    print(f"Starting Claude Code MLX Proxy on {config.HOST}:{config.PORT}")
    uvicorn.run(app, host=config.HOST, port=config.PORT)
