package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// fileConfig holds optional JSON-file overrides for upstream URL and API key.
type fileConfig struct {
	NvidiaURL string `json:"nvidia_url"`
	NvidiaKey string `json:"nvidia_key"`
}

// serverConfig holds the fully resolved runtime configuration (env vars win over file).
type serverConfig struct {
	addr                string
	upstreamURL         string
	providerAPIKey      string
	serverAPIKey        string
	timeout             time.Duration
	logBodyMax          int
	logStreamPreviewMax int
	debug               bool
}

// Hardened transport to handle extremely slow reasoning models without dropping the connection
var hardenedTransport = &http.Transport{
	Proxy: http.ProxyFromEnvironment,
	DialContext: (&net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 60 * time.Second,
	}).DialContext,
	ForceAttemptHTTP2:     true,
	MaxIdleConns:          100,
	IdleConnTimeout:       90 * time.Second,
	TLSHandshakeTimeout:   15 * time.Second,
	ExpectContinueTimeout: 1 * time.Second,
	ResponseHeaderTimeout: 10 * time.Minute, // Give NIM up to 10 minutes to start responding
}

// main loads config, registers routes, and starts the HTTP server.
func main() {
	cfg, err := loadConfig()
	if err != nil {
		log.Fatalf("config error: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		handleChatCompletions(w, r, cfg)
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": "openai-nvidia-proxy",
			"health":  "ok",
		})
	})

	srv := &http.Server{
		Addr:              cfg.addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
		// Intentionally omitting ReadTimeout and WriteTimeout to support multi-minute reasoning streams
	}

	log.Printf("listening on %s", cfg.addr)
	log.Printf("upstream: %s", cfg.upstreamURL)
	if cfg.serverAPIKey != "" {
		log.Printf("inbound auth: enabled")
	} else {
		log.Printf("inbound auth: disabled (SERVER_API_KEY not set)")
	}
	log.Fatal(srv.ListenAndServe())
}

// loadConfig merges the optional JSON file with environment variables (env wins).
func loadConfig() (*serverConfig, error) {
	fc, _ := loadFileConfig(strings.TrimSpace(envOr("CONFIG_PATH", "config.json")))
	if fc == nil {
		fc = &fileConfig{}
	}

	addr := strings.TrimSpace(envOr("ADDR", ":3001"))
	defaultURL := "https://integrate.api.nvidia.com/v1/chat/completions"
	if fc.NvidiaURL != "" {
		defaultURL = fc.NvidiaURL
	}
	upstreamURL := strings.TrimSpace(envOr("UPSTREAM_URL", defaultURL))
	providerAPIKey := strings.TrimSpace(envOr("PROVIDER_API_KEY", fc.NvidiaKey))
	serverAPIKey := strings.TrimSpace(envOr("SERVER_API_KEY", ""))

	timeout := 10 * time.Minute
	if raw := strings.TrimSpace(envOr("UPSTREAM_TIMEOUT_SECONDS", "")); raw != "" {
		seconds, err := strconv.Atoi(raw)
		if err != nil || seconds <= 0 {
			return nil, fmt.Errorf("invalid UPSTREAM_TIMEOUT_SECONDS: %q", raw)
		}
		timeout = time.Duration(seconds) * time.Second
	}

	logBodyMax := 4096
	if raw := strings.TrimSpace(envOr("LOG_BODY_MAX_CHARS", "")); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n < 0 {
			return nil, fmt.Errorf("invalid LOG_BODY_MAX_CHARS: %q", raw)
		}
		logBodyMax = n
	}

	logStreamPreviewMax := 256
	if raw := strings.TrimSpace(envOr("LOG_STREAM_TEXT_PREVIEW_CHARS", "")); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n < 0 {
			return nil, fmt.Errorf("invalid LOG_STREAM_TEXT_PREVIEW_CHARS: %q", raw)
		}
		logStreamPreviewMax = n
	}

	debug := false
	if raw := strings.TrimSpace(envOr("DEBUG", "")); raw != "" {
		debug = strings.EqualFold(raw, "true") || raw == "1"
	}

	if providerAPIKey == "" {
		return nil, errors.New("missing nvidia_key in config.json (or PROVIDER_API_KEY)")
	}
	return &serverConfig{
		addr:                addr,
		upstreamURL:         upstreamURL,
		providerAPIKey:      providerAPIKey,
		serverAPIKey:        serverAPIKey,
		timeout:             timeout,
		logBodyMax:          logBodyMax,
		logStreamPreviewMax: logStreamPreviewMax,
		debug:               debug,
	}, nil
}

// loadFileConfig reads and parses the JSON config file at the given path.
func loadFileConfig(path string) (*fileConfig, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var fc fileConfig
	if err := json.Unmarshal(b, &fc); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return &fc, nil
}

// envOr returns the env var value for key, or fallback if unset.
func envOr(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return fallback
}

// injectModelSpecificParams patches the request with model-specific parameters
// (chat_template_kwargs, reasoning_effort, reasoning_budget) that NVIDIA NIM
// requires for certain models but OpenAI clients don't know about.
func injectModelSpecificParams(reqID string, model string, req map[string]any) {
	// z-ai/glm-5.1
	if strings.Contains(model, "glm-5.1") {
		if _, exists := req["chat_template_kwargs"]; !exists {
			req["chat_template_kwargs"] = map[string]any{"enable_thinking": true, "clear_thinking": false}
		}
	}

	// z-ai/glm-5.2
	if strings.Contains(model, "glm-5.2") {
		if _, exists := req["chat_template_kwargs"]; !exists {
			req["chat_template_kwargs"] = map[string]any{"enable_thinking": true, "clear_thinking": false}
		}
	}

	// moonshotai/kimi-k2.6
	if strings.Contains(model, "kimi-k2") {
		if _, exists := req["chat_template_kwargs"]; !exists {
			req["chat_template_kwargs"] = map[string]any{"thinking": true}
		}
	}

	// nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
	if strings.Contains(model, "nemotron-3") && strings.Contains(model, "reasoning") {
		if _, exists := req["chat_template_kwargs"]; !exists {
			req["chat_template_kwargs"] = map[string]any{"enable_thinking": true}
		}
		if _, exists := req["reasoning_budget"]; !exists {
			req["reasoning_budget"] = 16384
		}
	}

	// nvidia/nemotron-3-super-120b-a12b
	if strings.Contains(model, "nemotron-3-super") {
		if _, exists := req["chat_template_kwargs"]; !exists {
			req["chat_template_kwargs"] = map[string]any{"enable_thinking": true}
		}
		if _, exists := req["reasoning_budget"]; !exists {
			req["reasoning_budget"] = 16384
		}
	}

	// deepseek-ai/deepseek-v4-pro
	if strings.Contains(model, "deepseek-v4-pro") {
		req["chat_template_kwargs"] = map[string]any{"thinking": true} // Force override
	}

	// deepseek-ai/deepseek-v4-flash
	if strings.Contains(model, "deepseek-v4-flash") {
		if _, exists := req["chat_template_kwargs"]; !exists {
			req["chat_template_kwargs"] = map[string]any{"thinking": true, "reasoning_effort": "high"}
		}
	}

	// mistralai/mistral-medium-3.5-128b
	// Mistral only supports "none" and "high" — coerce unsupported values (e.g. "medium")
	if strings.Contains(model, "mistral-medium") {
		if v, exists := req["reasoning_effort"]; !exists {
			req["reasoning_effort"] = "high"
		} else if s, ok := v.(string); ok && s != "none" && s != "high" {
			req["reasoning_effort"] = "high"
		}
	}

	// mistralai/mistral-small-4-119b-2603
	// Mistral only supports "none" and "high" — coerce unsupported values
	if strings.Contains(model, "mistral-small-4") {
		if v, exists := req["reasoning_effort"]; !exists {
			req["reasoning_effort"] = "high"
		} else if s, ok := v.(string); ok && s != "none" && s != "high" {
			req["reasoning_effort"] = "high"
		}
	}

	// minimaxai/minimax-m2.7
	if strings.Contains(model, "minimax-m2.7") {
		if _, exists := req["temperature"]; !exists {
			req["temperature"] = 1.0
		}
		if _, exists := req["top_p"]; !exists {
			req["top_p"] = 0.95
		}
		if _, exists := req["top_k"]; !exists {
			req["top_k"] = 40
		}
	}

	// minimaxai/minimax-m3
	if strings.Contains(model, "minimax-m3") {
		if _, exists := req["temperature"]; !exists {
			req["temperature"] = 1.0
		}
		if _, exists := req["top_p"]; !exists {
			req["top_p"] = 0.95
		}
		if _, exists := req["top_k"]; !exists {
			req["top_k"] = 40
		}
	}
}

// fixRequestData sanitizes inbound messages: strips reasoning_content NVIDIA
// rejects, coerces numeric tool_call IDs to strings, ensures content is never
// null, and marshals non-string function arguments.
func fixRequestData(reqID string, req map[string]any) {
	msgs, ok := req["messages"].([]any)
	if !ok {
		return
	}
	for i, m := range msgs {
		msg, ok := m.(map[string]any)
		if !ok {
			continue
		}

		role, _ := msg["role"].(string)

		// Sanitize Assistant messages (History)
		if role == "assistant" {
			if _, hasReasoning := msg["reasoning_content"]; hasReasoning {
				delete(msg, "reasoning_content")
				log.Printf("[%s] Sanitized: stripped 'reasoning_content' from msg %d", reqID, i)
			}

			// Some strict parsers crash if content is completely null. Force to empty string.
			if msg["content"] == nil {
				msg["content"] = ""
				log.Printf("[%s] Sanitized: forced null content to empty string in msg %d", reqID, i)
			}

			if msg["tool_calls"] == nil {
				delete(msg, "tool_calls")
			}
		}

		if role == "tool" {
			if msg["content"] == nil {
				msg["content"] = ""
			}
		}

		if tcid, ok := msg["tool_call_id"]; ok && tcid != nil {
			if v, isFloat := tcid.(float64); isFloat {
				msg["tool_call_id"] = fmt.Sprintf("%.0f", v)
			}
		}

		// Deep-sanitize tool_calls array inside 'assistant' messages
		if tcs, ok := msg["tool_calls"].([]any); ok {
			for tcIdx, tc := range tcs {
				if tcMap, ok := tc.(map[string]any); ok {
					if _, hasType := tcMap["type"]; !hasType {
						tcMap["type"] = "function"
					}
					if id, ok := tcMap["id"]; ok {
						if v, isFloat := id.(float64); isFloat {
							tcMap["id"] = fmt.Sprintf("%.0f", v)
						}
					}

					// Bulletproof arguments mapping to prevent "Expecting value: line 1 column 1"
					if fn, ok := tcMap["function"].(map[string]any); ok {
						args, hasArgs := fn["arguments"]

						if !hasArgs || args == nil {
							fn["arguments"] = "{}"
							log.Printf("[%s] Sanitized: fixed missing/null arguments to '{}' in tool_call %d, msg %d", reqID, tcIdx, i)
						} else if strArgs, isStr := args.(string); isStr {
							// If the model generated an empty string, python json.loads("") will crash.
							// Force it to a valid empty JSON object string.
							if strings.TrimSpace(strArgs) == "" {
								fn["arguments"] = "{}"
								log.Printf("[%s] Sanitized: fixed empty string arguments to '{}' in tool_call %d, msg %d", reqID, tcIdx, i)
							}
						} else {
							// If a client accidentally serialized arguments as a JSON map instead of string
							b, _ := json.Marshal(args)
							fn["arguments"] = string(b)
						}
					}
				}
			}
		}
	}
}

// dumpCrashingPayload pretty-prints the request payload for crash diagnostics,
// truncating the system prompt to keep logs readable.
func dumpCrashingPayload(reqID string, payload []byte) string {
	var req map[string]any
	if err := json.Unmarshal(payload, &req); err == nil {
		if msgs, ok := req["messages"].([]any); ok && len(msgs) > 0 {
			if firstMsg, ok := msgs[0].(map[string]any); ok {
				if content, ok := firstMsg["content"].(string); ok && len(content) > 500 {
					firstMsg["content"] = content[:500] + "\n... [TRUNCATED SYSTEM PROMPT FOR LOGS]"
				}
			}
		}
		b, _ := json.MarshalIndent(req, "", "  ")
		return string(b)
	}
	return string(payload)
}

// handleChatCompletions is the core proxy handler: auth, decode, sanitize,
// inject model params, then forward to upstream (streaming or non-streaming).
func handleChatCompletions(w http.ResponseWriter, r *http.Request, cfg *serverConfig) {
	reqID := fmt.Sprintf("req_%d", time.Now().UnixNano())
	if cfg.serverAPIKey != "" && !checkInboundAuth(r, cfg.serverAPIKey) {
		log.Printf("[%s] inbound unauthorized", reqID)
		writeJSONError(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("[%s] error reading request: %v", reqID, err)
		writeJSONError(w, http.StatusBadRequest, "invalid_request")
		return
	}

	bodyBytes = bytes.TrimPrefix(bodyBytes, []byte("\xef\xbb\xbf"))

	var openaiReq map[string]any
	if err := json.Unmarshal(bodyBytes, &openaiReq); err != nil {
		log.Printf("[%s] invalid inbound json: %v", reqID, err)
		writeJSONError(w, http.StatusBadRequest, "invalid_json")
		return
	}

	isStream, _ := openaiReq["stream"].(bool)
	model, _ := openaiReq["model"].(string)

	injectModelSpecificParams(reqID, model, openaiReq)
	fixRequestData(reqID, openaiReq)

	cleanBodyBytes, err := json.Marshal(openaiReq)
	if err != nil {
		log.Printf("[%s] error marshaling clean request: %v", reqID, err)
		writeJSONError(w, http.StatusInternalServerError, "internal_error")
		return
	}

	var debugFile string
	if cfg.debug {
		debugFile = filepath.Join(os.TempDir(), fmt.Sprintf("nim_debug_%s.json", reqID))
		_ = os.WriteFile(debugFile, cleanBodyBytes, 0644)
		log.Printf("[%s] DEBUG full request saved to: %s", reqID, debugFile)
	}

	if isStream {
		if err := proxyStream(w, r, cfg, reqID, cleanBodyBytes, debugFile); err != nil {
			log.Printf("[%s] stream proxy error: %v", reqID, err)
		}
		return
	}

	upstreamRespBody, resp, err := doUpstream(r.Context(), cfg, cleanBodyBytes, isStream)
	if err != nil {
		log.Printf("[%s] upstream request failed: %v", reqID, err)
		writeJSONError(w, http.StatusBadGateway, "upstream_request_failed")
		return
	}
	defer func() { _ = resp.Body.Close() }()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(upstreamRespBody)

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		log.Printf("\n=============================================")
		log.Printf("[%s] 🚨 UPSTREAM CRASHED WITH %d 🚨", reqID, resp.StatusCode)
		log.Printf("[%s] ERROR MSG: %s", reqID, string(upstreamRespBody))
		log.Printf("[%s] PAYLOAD SENT:\n%s", reqID, dumpCrashingPayload(reqID, cleanBodyBytes))
		log.Printf("=============================================\n")
	} else if debugFile != "" {
		_ = os.Remove(debugFile)
	}
}

// checkInboundAuth validates the request against the expected API key via
// Authorization: Bearer or X-Api-Key header (constant-time compare).
func checkInboundAuth(r *http.Request, expected string) bool {
	auth := strings.TrimSpace(r.Header.Get("Authorization"))
	if strings.HasPrefix(strings.ToLower(auth), "bearer ") {
		got := strings.TrimSpace(auth[len("bearer "):])
		return subtle.ConstantTimeCompare([]byte(got), []byte(expected)) == 1
	}
	if got := strings.TrimSpace(r.Header.Get("x-api-key")); got != "" {
		return subtle.ConstantTimeCompare([]byte(got), []byte(expected)) == 1
	}
	return false
}

// setSafeUpstreamHeaders sets headers required by the NVIDIA NIM upstream.
func setSafeUpstreamHeaders(req *http.Request, cfg *serverConfig, isStream bool) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.providerAPIKey)
	req.Header.Set("HTTP-Referer", "https://opencode.ai/")
	req.Header.Set("X-Title", "opencode")

	if isStream {
		req.Header.Set("Accept", "text/event-stream")
	} else {
		req.Header.Set("Accept", "application/json")
	}
}

// doUpstream sends a non-streaming request to the upstream NIM endpoint and
// returns the response body (re-wrapped for re-reading by the caller).
func doUpstream(ctx context.Context, cfg *serverConfig, bodyBytes []byte, isStream bool) ([]byte, *http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.upstreamURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, nil, err
	}
	req.ContentLength = int64(len(bodyBytes))
	setSafeUpstreamHeaders(req, cfg, isStream)

	// Use our hardened transport for non-streaming requests too just in case
	client := &http.Client{
		Timeout:   cfg.timeout,
		Transport: hardenedTransport,
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, nil, err
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		_ = resp.Body.Close()
		return nil, nil, err
	}
	_ = resp.Body.Close()

	resp.Body = io.NopCloser(bytes.NewReader(respBody))
	return respBody, resp, nil
}

// fixStreamIDs coerces numeric tool_call IDs to strings in streaming chunks.
func fixStreamIDs(chunk map[string]any) {
	choices, ok := chunk["choices"].([]any)
	if !ok {
		return
	}
	for _, ch := range choices {
		choice, ok := ch.(map[string]any)
		if !ok {
			continue
		}
		delta, ok := choice["delta"].(map[string]any)
		if !ok {
			continue
		}
		tcs, ok := delta["tool_calls"].([]any)
		if !ok {
			continue
		}

		for _, tc := range tcs {
			tcMap, ok := tc.(map[string]any)
			if !ok {
				continue
			}
			if id, ok := tcMap["id"]; ok {
				if v, isFloat := id.(float64); isFloat {
					tcMap["id"] = fmt.Sprintf("%.0f", v)
				}
			}
		}
	}
}

// proxyStream forwards a streaming request to the upstream, parsing and fixing
// each SSE chunk before writing it to the client response.
func proxyStream(w http.ResponseWriter, r *http.Request, cfg *serverConfig, reqID string, bodyBytes []byte, debugFile string) error {
	upReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, cfg.upstreamURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}
	upReq.ContentLength = int64(len(bodyBytes))
	setSafeUpstreamHeaders(upReq, cfg, true)

	// Use our hardened transport with massive keep-alives and zero total timeout
	client := &http.Client{
		Timeout:   0,
		Transport: hardenedTransport,
	}
	upResp, err := client.Do(upReq)
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, "upstream_request_failed")
		return err
	}
	defer func() { _ = upResp.Body.Close() }()

	if upResp.StatusCode < 200 || upResp.StatusCode >= 300 {
		raw, _ := io.ReadAll(upResp.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(upResp.StatusCode)
		_, _ = w.Write(raw)

		log.Printf("\n=============================================")
		log.Printf("[%s] 🚨 UPSTREAM CRASHED WITH %d 🚨", reqID, upResp.StatusCode)
		log.Printf("[%s] ERROR MSG: %s", reqID, string(raw))
		log.Printf("[%s] PAYLOAD SENT:\n%s", reqID, dumpCrashingPayload(reqID, bodyBytes))
		log.Printf("=============================================\n")

		return fmt.Errorf("upstream status %d", upResp.StatusCode)
	}

	if debugFile != "" {
		_ = os.Remove(debugFile)
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSONError(w, http.StatusInternalServerError, "streaming_not_supported")
		return errors.New("http.Flusher not supported")
	}

	reader := bufio.NewReader(upResp.Body)
	chunkCount, textChars, reasoningChars := 0, 0, 0
	sawDone := false

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if !errors.Is(err, io.EOF) {
				log.Printf("[%s] stream read error: %v", reqID, err)
			}
			break
		}

		strLine := string(line)

		if !strings.HasPrefix(strLine, "data:") {
			_, _ = w.Write(line)
			if strLine == "\n" || strLine == "\r\n" {
				flusher.Flush()
			}
			continue
		}

		dataContent := strings.TrimSpace(strings.TrimPrefix(strLine, "data:"))
		if dataContent == "[DONE]" {
			sawDone = true
			_, _ = w.Write(line)
			flusher.Flush()
			continue
		}

		var chunk map[string]any
		if err := json.Unmarshal([]byte(dataContent), &chunk); err == nil {
			// Fix malformed IDs coming from NVIDIA
			fixStreamIDs(chunk)

			// Safely extract stats for logs
			if choices, ok := chunk["choices"].([]any); ok && len(choices) > 0 {
				if ch, ok := choices[0].(map[string]any); ok {
					if delta, ok := ch["delta"].(map[string]any); ok {
						if c, _ := delta["content"].(string); c != "" {
							textChars += len([]rune(c))
						}
						if r, _ := delta["reasoning_content"].(string); r != "" {
							reasoningChars += len([]rune(r))
						}
					}
				}
			}

			b, err := json.Marshal(chunk)
			if err != nil {
				_, _ = w.Write(line)
				flusher.Flush()
				continue
			}
			if strings.HasSuffix(strLine, "\r\n") {
				_, _ = w.Write([]byte("data: " + string(b) + "\r\n"))
			} else {
				_, _ = w.Write([]byte("data: " + string(b) + "\n"))
			}
			chunkCount++
		} else {
			// Fallback: write original line if not parseable
			_, _ = w.Write(line)
		}
		flusher.Flush()
	}

	log.Printf("[%s] stream summary chunks=%d text_chars=%d reasoning_chars=%d saw_done=%v", reqID, chunkCount, textChars, reasoningChars, sawDone)
	return nil
}

// writeJSONError writes a structured JSON error response with the given status and code.
func writeJSONError(w http.ResponseWriter, status int, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"type":    "proxy_error",
			"code":    code,
			"message": code,
		},
	})
}
