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
_backend_model_name: Optional[str] = None


async def _get_backend_model_name() -> str:
    global _backend_model_name
    if _backend_model_name is not None:
        return _backend_model_name
    if http_client is None:
        return "default"
    try:
        resp = await http_client.get("/models", timeout=10.0)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            if models:
                _backend_model_name = models[0]["id"]
                print(f"Backend model: {_backend_model_name}")
                return _backend_model_name
    except Exception as e:
        print(f"Warning: could not query backend models: {e}")
    return "default"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        base_url=config.MLX_SERVER_URL,
        timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0),
    )
    await _get_backend_model_name()
    print(f"Proxy started — forwarding to {config.MLX_SERVER_URL}")
    yield
    await http_client.aclose()
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


def _anthropic_tools_to_openai(tools: List[Tool]) -> List[dict]:
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


def _anthropic_messages_to_openai(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]] = None,
) -> List[dict]:
    """Convert Anthropic-format messages to OpenAI chat messages."""
    result: List[dict] = []

    system_text = _extract_system_text(system)
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

    if config.TOOL_MODE != "none" and request.tools:
        body["tools"] = _anthropic_tools_to_openai(request.tools)

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
                id=_make_tool_use_id(),
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
                            "id": _make_tool_use_id(),
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
# API endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/messages")
async def create_message(request: MessagesRequest):
    if http_client is None:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    openai_body = await _build_openai_request(request)
    n_tools = len(request.tools) if request.tools else 0

    if config.VERBOSE:
        print(
            f"[/v1/messages] tools={n_tools}, tool_mode={config.TOOL_MODE}, "
            f"stream={request.stream}"
        )

    try:
        if request.stream:
            req = http_client.build_request(
                "POST", "/chat/completions", json=openai_body
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
        resp = await http_client.post("/chat/completions", json=openai_body)
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
            resp = await http_client.get("/models", timeout=5.0)
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
