# Local MLX Backend for Claude Code

This is a fork of [chand1012/claude-code-mlx-proxy](https://github.com/chand1012/claude-code-mlx-proxy) that rewrites the architecture to act as a lightweight translation proxy between **Claude Code** and an [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) OpenAI-compatible server. It allows you to use open-source models like Llama 3, GLM-4.5-Air, DeepSeek, and more, all running on your Apple Silicon Mac.

The original project loads models directly in-process. This fork instead forwards requests to a separately running mlx-lm server, which brings several advantages:

- **Decoupled processes** — Restart or update the proxy without reloading the model (which can be slow for large models).
- **Lightweight dependencies** — The proxy only needs `httpx` and `fastapi`; heavy ML dependencies (`mlx-lm`, `transformers`, `huggingface-hub`) live in the mlx-lm server.
- **Leverage mlx-lm server features** — Model management and inference optimizations are handled by mlx-lm directly, no need to reimplement.
- **Tool calling support** — Full Anthropic ↔ OpenAI format conversion for Claude Code's function calling.
- **Flexibility** — Can work with any OpenAI-compatible server in principle, not just mlx-lm.

## Why Use a Local Backend with Claude Code?

- **Total Privacy**: Your code, prompts, and conversations never leave your local machine.
- **Use Any Model**: Experiment with thousands of open-source models from the [MLX Community on Hugging Face](https://huggingface.co/mlx-community).
- **No API Keys or Costs**: Run powerful models without needing to manage API keys or pay for usage.

## How to Set It Up

There are three parts: starting the mlx-lm server, running the proxy, and configuring Claude Code to use it.

### Part 1: Start the mlx-lm Server

First, start an mlx-lm OpenAI-compatible server with the model you want to use:

```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/GLM-4.5-Air-3bit --port 8080
```

The first run will download the model, which may take some time. See the [mlx-lm documentation](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) for more options.

### Part 2: Run the Proxy

1. **Clone the repository:**

    ```bash
    git clone https://github.com/syarihu/claude-code-mlx-proxy.git
    cd claude-code-mlx-proxy
    ```

2. **Set up the environment:**
    Copy the example `.env` file:

    ```bash
    cp .env.example .env
    ```

    You can edit the `.env` file to customize the mlx-lm server URL, port, and other settings (see Configuration section below).

3. **Install dependencies:**
    This project uses `uv` for fast package management.

    ```bash
    uv sync
    ```

4. **Start the proxy:**

    ```bash
    uv run main.py
    ```

    The proxy will start on `http://localhost:8888` (or as configured in your `.env`) and begin forwarding requests to the mlx-lm server.

### Part 3: Configure Claude Code

Next, tell your Claude Code extension to send requests to your local server instead of the official Anthropic API.

As described in the [official Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/llm-gateway), you do this by setting the `ANTHROPIC_BASE_URL` environment variable.

The most reliable way to do this is to **launch your IDE from a terminal** where the variable has been set:

```bash
# Set the environment variable to point to your local server
export ANTHROPIC_BASE_URL=http://localhost:8888

# Now, launch Claude Code from this same terminal window
claude
```

Once your IDE is running, Claude Code will automatically use your local MLX backend. All requests will be translated from the Anthropic Messages API format to OpenAI format and forwarded to the mlx-lm server.

### Testing the Server

Before configuring Claude Code, you can verify the server is working correctly by sending it a `curl` request from your terminal:

#### Testing the Messages Endpoint

```bash
curl -X POST http://localhost:8888/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-4-sonnet-20250514",
    "max_tokens": 100,
    "messages": [
      {"role": "user", "content": "Explain what MLX is in one sentence."}
    ]
  }'
```

This will return a Claude-style response:

```json
{
  "id": "msg_12345678",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "MLX is Apple's machine learning framework optimized for efficient training and inference on Apple Silicon chips."
    }
  ],
  "model": "claude-4-sonnet-20250514",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 12,
    "output_tokens": 18
  }
}
```

#### Testing Token Counting

You can also test the token counting endpoint:

```bash
curl -X POST http://localhost:8888/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-4-sonnet-20250514",
    "messages": [
      {"role": "user", "content": "Explain what MLX is in one sentence."}
    ]
  }'
```

This returns the token count:

```json
{
  "input_tokens": 12
}
```

#### Streaming Support

The server also supports streaming responses using Server-Sent Events (SSE), just like the real Claude API:

```bash
curl -X POST http://localhost:8888/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-4-sonnet-20250514",
    "max_tokens": 100,
    "messages": [
      {"role": "user", "content": "Explain what MLX is in one sentence."}
    ],
    "stream": true
  }'
```

This will return a stream of events following the Claude streaming format.

## API Endpoints

The server implements the following Claude-compatible endpoints:

- `POST /v1/messages` - Create a message (supports both streaming and non-streaming)
- `POST /v1/messages/count_tokens` - Count tokens in a message
- `GET /v1/models` - List available models
- `GET /` - Root endpoint with server status
- `GET /health` - Health check endpoint (includes backend liveness check)

## Configuration (`.env`)

All server settings are managed through the `.env` file.

| Variable         | Default                        | Description                                                                                                                           |
| ---------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `HOST`           | `0.0.0.0`                      | The host address for the proxy server.                                                                                                |
| `PORT`           | `8888`                         | The port for the proxy server.                                                                                                        |
| `MLX_SERVER_URL`  | `http://localhost:8080/v1`     | URL of the mlx-lm OpenAI-compatible server (must include `/v1`).                                                                      |
| `API_MODEL_NAME` | `claude-4-sonnet-20250514`     | The model name that the API will report. Set this to a known Claude model to ensure client compatibility.                              |
| `TOOL_MODE`      | `slim`                         | Tool definition handling: `full` (keep all schemas), `slim` (name + description + required params only), `none` (omit tools entirely). |
| `VERBOSE`        | `false`                        | Enable verbose debug logging.                                                                                                         |
| `TAVILY_API_KEY` | _(unset)_                      | When set, the proxy intercepts Claude Code's `WebSearch` tool calls and runs them through the [Tavily Search API](https://tavily.com) (free tier: 1,000 searches/month, no credit card). Without a key, `WebSearch` falls through and typically returns no results because local backends cannot search the web. |
| `WEB_SEARCH_MAX_RESULTS` | `5`                    | Max results returned per Tavily search.                                                                                              |
| `WEB_SEARCH_MAX_ITERATIONS` | `5`                 | Safety limit on consecutive `WebSearch` calls inside a single proxy request.                                                          |

### Enabling web search (optional)

Local LLMs cannot perform real web searches on their own, so Claude Code's `WebSearch` tool returns empty results when pointed at a local backend. To make `WebSearch` work end-to-end:

1. Sign up at [tavily.com](https://tavily.com) and grab your API key (free tier gives 1,000 searches/month, no credit card required).
2. Add `TAVILY_API_KEY=tvly-...` to your `.env`.
3. Restart the proxy.

When `TAVILY_API_KEY` is set and Claude Code includes `WebSearch` in its tool list, the proxy intercepts the tool call internally, runs the query through Tavily, and feeds the results back to the local model — Claude Code only sees the model's final synthesized response.

## License

This project is licensed under the MIT License.
