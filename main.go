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
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

type fileConfig struct {
	NvidiaURL string `json:"nvidia_url"`
	NvidiaKey string `json:"nvidia_key"`
}

type serverConfig struct {
	addr                string
	upstreamURL         string
	providerAPIKey      string
	serverAPIKey        string
	timeout             time.Duration
	logBodyMax          int
	logStreamPreviewMax int
}

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
		ReadTimeout:       60 * time.Second,
		WriteTimeout:      0,
		IdleTimeout:       60 * time.Second,
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

	timeout := 5 * time.Minute
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
	}, nil
}

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

func envOr(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return fallback
}

// injectModelSpecificParams modifies the payload automatically for strict NVIDIA NIM reasoning models
func injectModelSpecificParams(reqID string, model string, req map[string]any) {
	// z-ai/glm-5.1
	if strings.Contains(model, "glm-5.1") {
		if _, exists := req["chat_template_kwargs"]; !exists {
			req["chat_template_kwargs"] = map[string]any{"enable_thinking": true, "clear_thinking": false}
			log.Printf("[%s] injected chat_template_kwargs for glm-5.1", reqID)
		}
	}

	// moonshotai/kimi-k2.6
	if strings.Contains(model, "kimi-k2") {
		if _, exists := req["chat_template_kwargs"]; !exists {
			req["chat_template_kwargs"] = map[string]any{"thinking": true}
			log.Printf("[%s] injected chat_template_kwargs for kimi", reqID)
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

	// deepseek-ai/deepseek-v4-pro
	if strings.Contains(model, "deepseek-v4-pro") {
		if _, exists := req["chat_template_kwargs"]; !exists {
			req["chat_template_kwargs"] = map[string]any{"thinking": false}
		}
	}

	// mistralai/mistral-medium-3.5-128b
	if strings.Contains(model, "mistral-medium") {
		if _, exists := req["reasoning_effort"]; !exists {
			req["reasoning_effort"] = "high"
		}
	}
}

// fixRequestData sanitizes history to prevent NVIDIA NIM's API Gateway from dropping the payload
func fixRequestData(req map[string]any) {
	msgs, ok := req["messages"].([]any)
	if !ok {
		return
	}
	for _, m := range msgs {
		msg, ok := m.(map[string]any)
		if !ok {
			continue
		}

		// 1. Remove reasoning_content from history to prevent upstream schema rejection
		if role, _ := msg["role"].(string); role == "assistant" {
			delete(msg, "reasoning_content")
		}

		// 2. Fix incoming numeric tool_call_ids (must be strings)
		if tcid, ok := msg["tool_call_id"]; ok {
			if v, isFloat := tcid.(float64); isFloat {
				msg["tool_call_id"] = fmt.Sprintf("%.0f", v)
			}
		}

		// 3. Fix numeric tool_calls IDs inside assistant messages
		if tcs, ok := msg["tool_calls"].([]any); ok {
			for _, tc := range tcs {
				if tcMap, ok := tc.(map[string]any); ok {
					if id, ok := tcMap["id"]; ok {
						if v, isFloat := id.(float64); isFloat {
							tcMap["id"] = fmt.Sprintf("%.0f", v)
						}
					}
				}
			}
		}
	}
}

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
	fixRequestData(openaiReq)

	logForwardedRequest(reqID, cfg, model, isStream, openaiReq)

	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(openaiReq); err != nil {
		log.Printf("[%s] error marshaling clean request: %v", reqID, err)
		writeJSONError(w, http.StatusInternalServerError, "internal_error")
		return
	}
	cleanBodyBytes := buf.Bytes()

	if isStream {
		if err := proxyStream(w, r, cfg, reqID, cleanBodyBytes); err != nil {
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
	defer resp.Body.Close()

	log.Printf("[%s] upstream status=%d", reqID, resp.StatusCode)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(upstreamRespBody)

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		logForwardedUpstreamBody(reqID, cfg, upstreamRespBody)
	}
}

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

func doUpstream(ctx context.Context, cfg *serverConfig, bodyBytes []byte, isStream bool) ([]byte, *http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.upstreamURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, nil, err
	}
	req.ContentLength = int64(len(bodyBytes))
	setSafeUpstreamHeaders(req, cfg, isStream)

	client := &http.Client{Timeout: cfg.timeout}
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

// fixStreamIDs corrects outgoing streams from NVIDIA so clients don't crash or cache bad tool IDs
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

func proxyStream(w http.ResponseWriter, r *http.Request, cfg *serverConfig, reqID string, bodyBytes []byte) error {
	upReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, cfg.upstreamURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}
	upReq.ContentLength = int64(len(bodyBytes))
	setSafeUpstreamHeaders(upReq, cfg, true)

	client := &http.Client{Timeout: 0}
	upResp, err := client.Do(upReq)
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, "upstream_request_failed")
		return err
	}
	defer upResp.Body.Close()

	if upResp.StatusCode < 200 || upResp.StatusCode >= 300 {
		raw, _ := io.ReadAll(upResp.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(upResp.StatusCode)
		_, _ = w.Write(raw)
		logForwardedUpstreamBody(reqID, cfg, raw)
		return fmt.Errorf("upstream status %d", upResp.StatusCode)
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
	var preview strings.Builder
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

		// If it's an empty line or not a data event, write it as-is (maintaining exact SSE framing)
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
							if cfg.logStreamPreviewMax > 0 && preview.Len() < cfg.logStreamPreviewMax {
								preview.WriteString(takeFirstRunes(c, cfg.logStreamPreviewMax-preview.Len()))
							}
						}
						if r, _ := delta["reasoning_content"].(string); r != "" {
							reasoningChars += len([]rune(r))
						}
					}
				}
			}

			// Re-marshal and write using the same line ending it came with
			b, _ := json.Marshal(chunk)
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

	if cfg.logStreamPreviewMax > 0 {
		log.Printf("[%s] stream summary chunks=%d text_chars=%d reasoning_chars=%d saw_done=%v preview=%q", reqID, chunkCount, textChars, reasoningChars, sawDone, preview.String())
	} else {
		log.Printf("[%s] stream summary chunks=%d text_chars=%d reasoning_chars=%d saw_done=%v", reqID, chunkCount, textChars, reasoningChars, sawDone)
	}
	return nil
}

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

func logForwardedRequest(reqID string, cfg *serverConfig, model string, stream bool, payload map[string]any) {
	inSummary := map[string]any{
		"model":  model,
		"stream": stream,
	}
	log.Printf("[%s] inbound summary=%s", reqID, mustJSONTrunc(inSummary, cfg.logBodyMax))
	log.Printf("[%s] forward url=%s", reqID, cfg.upstreamURL)

	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(payload)

	s := buf.String()
	if cfg.logBodyMax > 0 && len([]rune(s)) > cfg.logBodyMax {
		s = string([]rune(s)[:cfg.logBodyMax]) + "...(truncated)"
	}
	log.Printf("[%s] forward body=%s", reqID, strings.TrimSpace(s))
}

func logForwardedUpstreamBody(reqID string, cfg *serverConfig, body []byte) {
	if cfg.logBodyMax == 0 {
		return
	}
	s := string(body)
	if len([]rune(s)) > cfg.logBodyMax {
		s = string([]rune(s)[:cfg.logBodyMax]) + "...(truncated)"
	}
	log.Printf("[%s] upstream body=%s", reqID, s)
}

func mustJSONTrunc(v any, maxChars int) string {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf(`{"_error":"json_marshal_failed"}`)
	}
	s := string(b)
	if maxChars == 0 {
		return "(disabled)"
	}
	if len([]rune(s)) > maxChars {
		return string([]rune(s)[:maxChars]) + "...(truncated)"
	}
	return s
}

func takeFirstRunes(s string, max int) string {
	if max <= 0 || s == "" {
		return ""
	}
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max])
}
