package adaptive

import (
	"encoding/json"
	"os"
	"sync"
)

type TraceWriter interface {
	Write(sample TrajectorySample) error
	Close() error
}

type JSONLTraceWriter struct {
	mu   sync.Mutex
	file *os.File
	enc  *json.Encoder
}

func NewJSONLTraceWriter(path string) (*JSONLTraceWriter, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return nil, err
	}
	return &JSONLTraceWriter{
		file: file,
		enc:  json.NewEncoder(file),
	}, nil
}

func (w *JSONLTraceWriter) Write(sample TrajectorySample) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.enc.Encode(sample)
}

func (w *JSONLTraceWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.file == nil {
		return nil
	}
	err := w.file.Close()
	w.file = nil
	return err
}
