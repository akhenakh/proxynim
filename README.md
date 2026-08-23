# NVIDIA NIM OpenAI-Compatible Proxy

A lightweight, robust Go proxy that bridges the gap between standard OpenAI-compatible CLI clients (like [Crush](https://github.com/charmbracelet/crush)) and the strict, proprietary API requirements of NVIDIA NIM.

### Why is this needed?
NVIDIA NIM hosts a diverse set of models (Mistral, DeepSeek, Z.ai, Moonshot, Minimax, Meta), but requires specific root-level JSON parameters (like `chat_template_kwargs` or `reasoning_budget`) to enable reasoning features. Furthermore, NVIDIA's API Gateway strictly validates incoming schemas: it drops requests if `tool_call_id`s are numeric or if non-standard keys like `reasoning_content` are sent in the chat history.

Standard tools like `Crush` don't know about these NIM quirks. **This proxy sits in the middle and fixes everything transparently:**
- **Auto-injects proprietary kwargs** (`enable_thinking`, `reasoning_effort`, etc.) based on the specific model requested.
- **Sanitizes history** by safely stripping out `reasoning_content` from previous assistant turns.
- **Fixes Type Errors** by catching numeric `tool_call_id`s emitted by NVIDIA and converting them to strings before they hit your CLI, and doing the same on the way back up.

## 🚀 Quick Start

### 1. Configure the Proxy
Create a `config.json` in the root of the project to hold your NVIDIA API key:
```json
{
  "nvidia_key": "nvapi-YOUR-NVIDIA-API-KEY-HERE"
}
```

### 2. Run the Proxy
```bash
go run main.go
```
The proxy will start on `http://localhost:3001`. 

---

## 🛠️ Using with Crush

To use the proxy with Crush, you need to add a custom provider to your crush configuration file (usually found at `~/.config/crush/crush.json` or similar).

**Important:** You must use `"type": "openai-compat"` (not `"openai"`) because we are routing to a non-OpenAI provider. Because the proxy handles authentication with NVIDIA, you can set the `api_key` in crush to `"dummy"`.

Add the following block to the `providers` section of your configuration. This includes all the reasoning models the proxy currently knows how to auto-configure:

```json
{
  "$schema": "https://charm.land/crush.json",
  "providers": {
    "nimproxy": {
      "name": "NVIDIA NIM",
      "type": "openai-compat",
      "base_url": "http://localhost:3001/v1",
      "api_key": "dummy",
      "models":[
      {
        "id": "z-ai/glm-5.1",
        "name": "GLM 5.1 (Reasoning)",
        "context_window": 200000,
        "default_max_tokens": 131072
      },
      {
        "id": "z-ai/glm-5.2",
        "name": "GLM 5.2 (Reasoning)",
        "context_window": 200000,
        "default_max_tokens": 131072
      },
      {
        "id": "moonshotai/kimi-k2.6",
        "name": "Kimi k2.6",
        "context_window": 262144,
        "default_max_tokens": 32768
      },
      {
        "id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "name": "Nemotron 3 Nano Omni",
        "context_window": 256000,
        "default_max_tokens": 65536
      },
      {
        "id": "deepseek-ai/deepseek-v4-pro",
        "name": "DeepSeek v4 Pro",
        "context_window": 1048576,
        "default_max_tokens": 384000
      },
      {
        "id": "deepseek-ai/deepseek-v4-flash",
        "name": "DeepSeek v4 Flash",
        "context_window": 1048576,
        "default_max_tokens": 262144
      },
      {
        "id": "deepseek-ai/deepseek-v4-flash-0731",
        "name": "DeepSeek v4 Flash 0731",
        "context_window": 1048576,
        "default_max_tokens": 262144
      },
      {
        "id": "mistralai/mistral-medium-3.5-128b",
        "name": "Mistral Medium 3.5",
        "context_window": 262144,
        "default_max_tokens": 32768
      },
      {
        "id": "mistralai/mistral-small-4-119b-2603",
        "name": "Mistral Small 4",
        "context_window": 262144,
        "default_max_tokens": 32768
      },
      {
        "id": "nvidia/nemotron-3-super-120b-a12b",
        "name": "Nemotron 3 Super",
        "context_window": 256000,
        "default_max_tokens": 65536
      },
      {
        "id": "minimaxai/minimax-m2.7",
        "name": "MiniMax M2.7",
        "context_window": 200000,
        "default_max_tokens": 8192
      },
      {
        "id": "minimaxai/minimax-m3",
        "name": "MiniMax M3",
        "context_window": 200000,
        "default_max_tokens": 8192
      },
      {
        "id": "meta/muse-glimmer-30b",
        "name": "Muse Glimmer 30B",
        "context_window": 131072,
        "default_max_tokens": 8192
      }
      ]
    }
  }
}
```

*(Note: Adjust the `context_window` and pricing fields as needed for your specific use-case).*

### 3. Run Crush
Once configured and the proxy is running, you can invoke crush with any of the NIM models:
```bash
crush --model "nimproxy/deepseek-ai/deepseek-v4-flash-0731" "Look at main.go and list the functions"
```
You will see the proxy intercepting the call, injecting the correct `chat_template_kwargs`, and seamlessly streaming the response back to your terminal!

## 🔧 Environment Variables

You can override default proxy behaviors using the following environment variables:
- `ADDR`: Port to listen on (Default: `:3001`)
- `CONFIG_PATH`: Path to the config file (Default: `config.json`)
- `NVIDIA_API_KEY`: Alternative to using `config.json`
- `SERVER_API_KEY`: If set, requires `crush` to send this key via the `api_key` config field for inbound auth.
