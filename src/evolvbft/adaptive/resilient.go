package adaptive

import (
	"math"
	"net"
	"net/http"
	"sync"
	"time"
)

// ResilientClient wraps an http.Client with retry (exponential backoff)
// and circuit-breaker logic for the Go↔Python bridge.
//
// Retry policy: up to MaxRetries attempts with exponential backoff.
// Circuit breaker: after ConsecutiveFailures errors, open circuit for
// OpenDuration; then allow one probe request (half-open).
type ResilientClient struct {
	inner *http.Client

	// Retry
	MaxRetries     int
	BaseBackoff    time.Duration
	MaxBackoff     time.Duration
	BackoffJitter  float64 // 0.0–1.0 random jitter fraction
	RetryableCheck func(resp *http.Response, err error) bool

	// Circuit breaker
	ConsecutiveFailures int
	OpenDuration        time.Duration

	mu            sync.Mutex
	failures      int
	circuitOpenAt time.Time
	circuitState  circuitState
}

type circuitState int

const (
	circuitClosed circuitState = iota
	circuitOpen
	circuitHalfOpen
)

// DefaultResilientClient returns a client tuned for the SFAC/FACMAC bridge:
// 2 retries, 50ms base backoff, circuit opens after 5 consecutive failures
// for 10 seconds.
func DefaultResilientClient(timeout time.Duration) *ResilientClient {
	transport := &http.Transport{
		MaxIdleConns:        4,
		MaxIdleConnsPerHost: 4,
		IdleConnTimeout:     90 * time.Second,
		DialContext: (&net.Dialer{
			Timeout:   2 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
	}
	return &ResilientClient{
		inner: &http.Client{
			Timeout:   timeout,
			Transport: transport,
		},
		MaxRetries:          2,
		BaseBackoff:         50 * time.Millisecond,
		MaxBackoff:          500 * time.Millisecond,
		BackoffJitter:       0.25,
		ConsecutiveFailures: 5,
		OpenDuration:        10 * time.Second,
		RetryableCheck:      defaultRetryableCheck,
		circuitState:        circuitClosed,
	}
}

// Do executes the request with retry and circuit-breaker logic.
func (c *ResilientClient) Do(req *http.Request) (*http.Response, error) {
	if err := c.checkCircuit(); err != nil {
		return nil, err
	}

	var lastResp *http.Response
	var lastErr error

	for attempt := 0; attempt <= c.MaxRetries; attempt++ {
		if attempt > 0 {
			backoff := c.backoffDuration(attempt)
			time.Sleep(backoff)
		}

		// Clone request body for retry (only if body was already consumed)
		var reqCopy *http.Request
		if req.GetBody != nil {
			body, err := req.GetBody()
			if err != nil {
				return nil, err
			}
			reqCopy = req.Clone(req.Context())
			reqCopy.Body = body
		} else {
			reqCopy = req
		}

		lastResp, lastErr = c.inner.Do(reqCopy)
		if c.RetryableCheck != nil && c.RetryableCheck(lastResp, lastErr) {
			continue
		}
		// Success
		c.recordSuccess()
		return lastResp, lastErr
	}

	// All retries exhausted
	c.recordFailure()
	return lastResp, lastErr
}

func (c *ResilientClient) checkCircuit() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	switch c.circuitState {
	case circuitOpen:
		if time.Since(c.circuitOpenAt) >= c.OpenDuration {
			c.circuitState = circuitHalfOpen
			return nil // allow one probe
		}
		return &CircuitOpenError{
			OpenAt:   c.circuitOpenAt,
			Duration: c.OpenDuration,
			Failures: c.failures,
		}
	case circuitHalfOpen:
		return nil // probe in progress
	default:
		return nil
	}
}

func (c *ResilientClient) recordSuccess() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.failures = 0
	c.circuitState = circuitClosed
}

func (c *ResilientClient) recordFailure() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.failures++
	if c.failures >= c.ConsecutiveFailures {
		c.circuitState = circuitOpen
		c.circuitOpenAt = time.Now()
	}
}

func (c *ResilientClient) backoffDuration(attempt int) time.Duration {
	base := float64(c.BaseBackoff) * math.Pow(2, float64(attempt-1))
	if base > float64(c.MaxBackoff) {
		base = float64(c.MaxBackoff)
	}
	return time.Duration(base)
}

// CircuitState returns the current circuit breaker state for observability.
func (c *ResilientClient) CircuitState() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	switch c.circuitState {
	case circuitOpen:
		return "open"
	case circuitHalfOpen:
		return "half-open"
	default:
		return "closed"
	}
}

// ConsecutiveFailureCount returns current failure count for observability.
func (c *ResilientClient) ConsecutiveFailureCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.failures
}

// CircuitOpenError indicates the circuit breaker is open.
type CircuitOpenError struct {
	OpenAt   time.Time
	Duration time.Duration
	Failures int
}

func (e *CircuitOpenError) Error() string {
	return "bridge circuit breaker open"
}

func defaultRetryableCheck(resp *http.Response, err error) bool {
	if err != nil {
		// Network error — retryable
		return true
	}
	// Server errors (5xx) — retryable; client errors (4xx) — not retryable
	return resp.StatusCode >= 500
}
