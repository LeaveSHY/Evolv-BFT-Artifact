package adaptive

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"strings"
	"time"
)

type Policy interface {
	Name() string
	Decide(observation Observation) Action
}

// FeedbackPolicy extends Policy with a feedback channel for online training.
// After each Tick(), the controller sends the trajectory sample to the Python
// server via Feedback(). Policies that do not support feedback simply omit
// this interface.
type FeedbackPolicy interface {
	Policy
	Feedback(sample TrajectorySample) error
}

type SafeBaselinePolicy struct{}

func (SafeBaselinePolicy) Name() string {
	return "safe-baseline"
}

func (SafeBaselinePolicy) Decide(observation Observation) Action {
	action := Action{
		CommitteeSize:             observation.CommitteeSize,
		PacemakerTimeoutMs:        observation.PacemakerTimeoutMs,
		MempoolMaxBatchTxs:        observation.MempoolMaxBatchTxs,
		MempoolProposalIntervalMs: observation.MempoolProposalIntervalMs,
		Reason:                    "hold",
	}

	targetConfig := int(maxUint64(observation.CurrentConfigID, observation.HighestKnownConfigID) + 1)
	if !observation.CanParticipate {
		action.HydraDiscoveryTarget = targetConfig
		action.Reason = "membership-catch-up"
		return action
	}
	if !observation.LocalValidator && observation.PendingJoins == 0 {
		action.SubmitJoin = true
		action.HydraDiscoveryTarget = targetConfig
		action.Reason = "membership-join"
		return action
	}
	if !observation.LocalValidator && observation.PendingJoins > 0 && observation.HighestKnownConfigID <= observation.CurrentConfigID {
		action.SubmitJoin = true
		action.HydraDiscoveryTarget = int(observation.CurrentConfigID + 1)
		action.Reason = "membership-join-retry"
		return action
	}
	if observation.LocalValidator && observation.PendingLeaves > 0 {
		action.SubmitLeave = true
		action.Reason = "membership-leave-retry"
		return action
	}

	degraded := observation.LatencyP95Ms >= 500 || observation.BacklogPending >= 256 || observation.RejectTotal > 0
	if degraded {
		action.PacemakerTimeoutMs = maxInt(observation.PacemakerTimeoutMs+250, 500)
		action.MempoolMaxBatchTxs = maxInt(observation.MempoolMaxBatchTxs/2, 128)
		action.MempoolProposalIntervalMs = minInt(observation.MempoolProposalIntervalMs+25, 500)
		action.Reason = "degraded-path"
		return action
	}

	if observation.ThroughputTPS > 0 && observation.LatencyP95Ms < 200 && observation.BacklogPending < 32 {
		action.PacemakerTimeoutMs = maxInt(observation.PacemakerTimeoutMs-100, 250)
		action.MempoolMaxBatchTxs = minInt(observation.MempoolMaxBatchTxs+256, 4096)
		action.MempoolProposalIntervalMs = maxInt(observation.MempoolProposalIntervalMs-10, 25)
		action.Reason = "healthy-path"
	}

	return action
}

type ScriptedPolicy struct {
	path string
}

func NewScriptedPolicy(path string) ScriptedPolicy {
	return ScriptedPolicy{path: path}
}

func (p ScriptedPolicy) Name() string {
	return "scripted"
}

func (p ScriptedPolicy) Decide(observation Observation) Action {
	if p.path == "" {
		return observationFallbackAction(observation, "empty-script")
	}

	raw, err := os.ReadFile(p.path)
	if err != nil {
		return observationFallbackAction(observation, "script-read-error")
	}

	var action Action
	if err := json.Unmarshal(raw, &action); err != nil {
		return observationFallbackAction(observation, "script-parse-error")
	}
	if action.Reason == "" {
		action.Reason = "scripted"
	}
	return action
}

type HTTPPolicy struct {
	name      string
	url       string
	client    *http.Client
	resilient *ResilientClient
}

func NewHTTPPolicy(url string, timeoutMs int) HTTPPolicy {
	return newNamedHTTPPolicy("http", url, timeoutMs)
}

func newNamedHTTPPolicy(name string, url string, timeoutMs int) HTTPPolicy {
	if timeoutMs <= 0 {
		timeoutMs = 1000
	}
	if name == "" {
		name = "http"
	}
	timeout := time.Duration(timeoutMs) * time.Millisecond
	return HTTPPolicy{
		name:      name,
		url:       url,
		client:    &http.Client{Timeout: timeout},
		resilient: DefaultResilientClient(timeout),
	}
}

func (p HTTPPolicy) Name() string {
	return p.name
}

func (p HTTPPolicy) Decide(observation Observation) Action {
	fallback := observationFallbackAction(observation, "")
	if p.url == "" {
		fallback.Reason = "empty-http-url"
		return fallback
	}
	body, err := json.Marshal(observation)
	if err != nil {
		fallback.Reason = "http-marshal-error"
		return fallback
	}
	req, err := http.NewRequest(http.MethodPost, p.url, bytes.NewReader(body))
	if err != nil {
		fallback.Reason = "http-request-error"
		return fallback
	}
	req.Header.Set("Content-Type", "application/json")
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(body)), nil
	}
	resp, err := p.resilient.Do(req)
	if err != nil {
		if _, ok := err.(*CircuitOpenError); ok {
			fallback.Reason = "http-circuit-open"
		} else {
			fallback.Reason = "http-do-error"
		}
		return fallback
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices || resp.StatusCode == http.StatusNoContent {
		fallback.Reason = "http-status-error"
		return fallback
	}
	contentType := resp.Header.Get("Content-Type")
	if contentType != "" {
		mediaType, _, err := mime.ParseMediaType(contentType)
		if err != nil || !strings.EqualFold(mediaType, "application/json") {
			fallback.Reason = "http-content-type-error"
			return fallback
		}
	}
	decoder := json.NewDecoder(resp.Body)
	decoder.DisallowUnknownFields()
	var action Action
	if err := decoder.Decode(&action); err != nil {
		fallback.Reason = "http-decode-error"
		return fallback
	}
	if err := decoder.Decode(new(struct{})); err != io.EOF {
		fallback.Reason = "http-decode-error"
		return fallback
	}
	action = mergeHTTPActionFallback(action, fallback)
	if action.Reason == "" {
		action.Reason = "http-policy"
	}
	return action
}

func observationFallbackAction(observation Observation, reason string) Action {
	return Action{
		CommitteeSize:             observation.CommitteeSize,
		PacemakerTimeoutMs:        observation.PacemakerTimeoutMs,
		MempoolMaxBatchTxs:        observation.MempoolMaxBatchTxs,
		MempoolProposalIntervalMs: observation.MempoolProposalIntervalMs,
		Reason:                    reason,
	}
}

func mergeHTTPActionFallback(action Action, fallback Action) Action {
	merged := action
	if merged.CommitteeSize == 0 {
		merged.CommitteeSize = fallback.CommitteeSize
	}
	if merged.PacemakerTimeoutMs == 0 {
		merged.PacemakerTimeoutMs = fallback.PacemakerTimeoutMs
	}
	if merged.MempoolMaxBatchTxs == 0 {
		merged.MempoolMaxBatchTxs = fallback.MempoolMaxBatchTxs
	}
	if merged.MempoolProposalIntervalMs == 0 {
		merged.MempoolProposalIntervalMs = fallback.MempoolProposalIntervalMs
	}
	return merged
}

// Feedback sends a trajectory sample to the Python server for online training.
// Implements FeedbackPolicy interface.
func (p HTTPPolicy) Feedback(sample TrajectorySample) error {
	if p.url == "" {
		return nil
	}
	// Derive feedback URL: replace trailing path with /trace/ingest
	feedbackURL := feedbackURLFromInfer(p.url)
	body, err := json.Marshal(sample)
	if err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPost, feedbackURL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(body)), nil
	}
	resp, err := p.resilient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("feedback HTTP %d", resp.StatusCode)
	}
	return nil
}

// feedbackURLFromInfer derives the /trace/ingest URL from the /infer URL.
// E.g. "http://127.0.0.1:8000/infer" → "http://127.0.0.1:8000/trace/ingest"
func feedbackURLFromInfer(inferURL string) string {
	if idx := strings.LastIndex(inferURL, "/"); idx >= 0 {
		return inferURL[:idx] + "/trace/ingest"
	}
	return inferURL + "/trace/ingest"
}

func PolicyByName(name string, scriptPath string, policyURL string) Policy {
	switch name {
	case "", "off":
		return nil
	case "safe-baseline":
		return SafeBaselinePolicy{}
	case "scripted":
		return NewScriptedPolicy(scriptPath)
	case "sfac":
		return NewSFACPolicy(policyURL, 2000)
	case "sfac-grpc":
		p, err := NewSFACGRPCPolicy(policyURL, 2000)
		if err != nil {
			// Cannot establish gRPC connection; fall back to safe baseline
			return SafeBaselinePolicy{}
		}
		return p
	case "http", "facmac-http":
		return newNamedHTTPPolicy(name, policyURL, 1000)
	default:
		return SafeBaselinePolicy{}
	}
}

func policyModeForName(name string) string {
	switch name {
	case "safe-baseline", "scripted", "http", "facmac-http", "sfac", "sfac-grpc":
		return name
	default:
		return "unknown"
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxUint64(a, b uint64) uint64 {
	if a > b {
		return a
	}
	return b
}
