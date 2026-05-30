package adaptive

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestResilientClientRetriesOnServerError(t *testing.T) {
	var attempts int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&attempts, 1)
		if n <= 2 {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer srv.Close()

	client := DefaultResilientClient(5 * time.Second)
	req, _ := http.NewRequest(http.MethodPost, srv.URL, strings.NewReader(`{}`))
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(strings.NewReader(`{}`)), nil
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("expected success after retries, got error: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if atomic.LoadInt32(&attempts) != 3 {
		t.Fatalf("expected 3 attempts (1 initial + 2 retries), got %d", attempts)
	}
}

func TestResilientClientNoRetryOnClientError(t *testing.T) {
	var attempts int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&attempts, 1)
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer srv.Close()

	client := DefaultResilientClient(5 * time.Second)
	req, _ := http.NewRequest(http.MethodPost, srv.URL, strings.NewReader(`{}`))
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
	if atomic.LoadInt32(&attempts) != 1 {
		t.Fatalf("expected exactly 1 attempt for 4xx, got %d", attempts)
	}
}

func TestResilientClientCircuitBreaker(t *testing.T) {
	var attempts int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&attempts, 1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	client := DefaultResilientClient(2 * time.Second)
	client.MaxRetries = 0
	client.ConsecutiveFailures = 3
	client.OpenDuration = 100 * time.Millisecond

	// Exhaust consecutive failures to trip the circuit
	for i := 0; i < 3; i++ {
		req, _ := http.NewRequest(http.MethodPost, srv.URL, nil)
		resp, _ := client.Do(req)
		if resp != nil {
			resp.Body.Close()
		}
	}
	attemptsBeforeOpen := atomic.LoadInt32(&attempts)

	if client.CircuitState() != "open" {
		t.Fatalf("expected circuit open after %d failures, got %s", 3, client.CircuitState())
	}

	// While circuit is open, request should fail immediately with CircuitOpenError
	// (no HTTP call made to the server)
	req, _ := http.NewRequest(http.MethodPost, srv.URL, nil)
	_, err := client.Do(req)
	if err == nil {
		t.Fatal("expected CircuitOpenError, got nil")
	}
	if _, ok := err.(*CircuitOpenError); !ok {
		t.Fatalf("expected CircuitOpenError, got %T: %v", err, err)
	}
	// Verify no actual HTTP call was made during open state
	if atomic.LoadInt32(&attempts) != attemptsBeforeOpen {
		t.Fatal("expected no HTTP call while circuit is open")
	}
}

func TestResilientClientCircuitRecovery(t *testing.T) {
	var failCount int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.LoadInt32(&failCount)
		if n < 3 {
			atomic.AddInt32(&failCount, 1)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer srv.Close()

	client := DefaultResilientClient(2 * time.Second)
	client.MaxRetries = 0
	client.ConsecutiveFailures = 3
	client.OpenDuration = 50 * time.Millisecond

	// Trip the circuit
	for i := 0; i < 3; i++ {
		req, _ := http.NewRequest(http.MethodPost, srv.URL, nil)
		resp, _ := client.Do(req)
		if resp != nil {
			resp.Body.Close()
		}
	}

	// Wait for recovery
	time.Sleep(100 * time.Millisecond)

	// Probe should succeed → circuit closed
	req, _ := http.NewRequest(http.MethodPost, srv.URL, nil)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("expected success on probe, got %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 on probe, got %d", resp.StatusCode)
	}
	if client.CircuitState() != "closed" {
		t.Fatalf("expected circuit closed after successful probe, got %s", client.CircuitState())
	}
}
