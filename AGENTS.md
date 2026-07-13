# AGENTS.md — proxynim

## What This Project Is

A stateless Go reverse-proxy that translates OpenAI-compatible `/v1/chat/completions` requests into NVIDIA NIM API calls. It sanitizes payloads, injects model-specific parameters, fixes malformed IDs in responses, and transparently proxies both streaming and non-streaming calls.

Single-file Go binary — everything lives in `main.go`.

## Build & Run

```sh
go build -o proxynim .
PROVIDER_API_KEY=nvapi-xxx ./proxynim
```

No Makefile, no CI, no Dockerfile. Standard `go build` only.

## Configuration

Config is loaded from **environment variables** with an optional JSON file fallback (`config.json` by default, path overridden by `CONFIG_PATH`). Env vars always take precedence.

| Env Var | Fallback | Purpose |
|---|---|---|
| `PROVIDER_API_KEY` | `config.json → nvidia_key` | **Required.** NVIDIA API key |
| `UPSTREAM_URL` | `config.json → nvidia_url` or `https://integrate.api.nvidia.com/v1/chat/completions` | Upstream endpoint |
| `SERVER_API_KEY` | _(empty)_ | If set, requires inbound `Authorization: Bearer <key>` or `X-Api-Key` header |
| `ADDR` | `:3001` | Listen address |
| `UPSTREAM_TIMEOUT_SECONDS` | `300` (5 min) | Non-stream upstream timeout; streaming uses no timeout |
| `LOG_BODY_MAX_CHARS` | `4096` | Truncation limit for logged request/response bodies (0 = disabled) |
| `LOG_STREAM_TEXT_PREVIEW_CHARS` | `256` | Max text preview in stream-summary log line |
| `CONFIG_PATH` | `config.json` | Path to JSON config file |

## Architecture & Control Flow

```
Client → inbound auth check → parse JSON → injectModelSpecificParams → fixRequestData → forward to upstream → proxy response (stream or non-stream)
```

- **Inbound auth** (`checkInboundAuth`): constant-time comparison of Bearer token or X-Api-Key header.
- **`injectModelSpecificParams`**: Patches the request JSON with `chat_template_kwargs` or `reasoning_effort` for specific NVIDIA NIM models that require them (glm-5.1, glm-5.2, kimi-k2, nemotron-3, nemotron-3-super, deepseek-v4-pro, deepseek-v4-flash, mistral-medium, mistral-small-4, minimax-m2.7). Checks by `strings.Contains` on model name — not exact match.
- **`fixRequestData`**: Strips `reasoning_content` from assistant messages in history (NVIDIA rejects it), coerces numeric `tool_call_id` and `tool_calls[].id` to strings (JSON number→string mismatch).
- **`fixStreamIDs`**: Same numeric→string ID fix but on streaming response chunks.
- **BOM stripping**: Request body is stripped of UTF-8 BOM (`\xef\xbb\xbf`) before parsing.
- **Re-serialization**: The entire request is re-marshaled with `SetEscapeHTML(false)` before forwarding, ensuring HTML entities aren't double-escaped.
- **Streaming**: SSE lines are parsed, fixed, re-marshaled, and flushed individually. Original line endings (`\n` vs `\r\n`) are preserved. The `[DONE]` sentinel is passed through as-is.

## Key Gotchas

- **No external dependencies** — this is pure stdlib Go. No third-party packages.
- **`go.mod` requires Go 1.26.3** — older Go toolchains will refuse to build.
- **Streaming uses `http.Client{Timeout: 0}`** — no client-side timeout on streaming; the server's `WriteTimeout: 0` also disables the write deadline for long-running streams.
- **`injectModelSpecificParams` uses substring matching** (`strings.Contains`) — a model name like "my-glm-5.1-finetune" will still match. This is intentional but worth knowing.
- **Model-specific injections only fire if the key doesn't already exist** in the request — callers can override by pre-setting the field.
- **Numeric tool call IDs**: NVIDIA's API sometimes returns tool call IDs as JSON numbers instead of strings. Both `fixRequestData` (inbound) and `fixStreamIDs` (outbound) handle this, coercing `float64` → `string` via `fmt.Sprintf("%.0f", v)`.
- **`config.json` contains the API key in plaintext** — don't commit it. Prefer `PROVIDER_API_KEY` env var.
- **`HTTP-Referer: https://opencode.ai/` and `X-Title: opencode`** are hardcoded in upstream headers — some NVIDIA models may behave differently based on these.
- **Logging truncation**: Body logs truncate by rune count, not byte count, so multi-byte characters won't be sliced mid-sequence.

## Routes

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | Main proxy endpoint |
| `*` | `/` | Health check — returns `{"message":"openai-nvidia-proxy","health":"ok"}` |

## Testing

No tests exist in this project. If adding tests, use Go's standard `go test` and consider `httptest.NewRecorder` + `httptest.NewServer` for proxy handler testing.

## Code Style

- Single package (`main`), single file (`main.go`)
- Standard `log` package for logging — no structured logging library
- Request IDs use `req_<unix_nano>` format
- JSON error responses follow `{"error":{"type":"proxy_error","code":"<code>","message":"<code>"}}` shape
- All config via env vars with JSON file fallback — no CLI flags
