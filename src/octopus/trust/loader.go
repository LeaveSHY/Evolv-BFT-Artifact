// Package trust — loader for SFAC-trained classifier weights.
//
// Reads JSON exported by experiments/export_trust_weights.py so the Go
// integration tests use the same (w, b) the Python MARL pipeline trained
// instead of a hand-tuned vector. Closes audit gap G4 / D-10.
package trust

import (
	"encoding/json"
	"fmt"
	"os"
)

// classifierWeightsJSON is the on-disk schema produced by
// experiments/export_trust_weights.py.
type classifierWeightsJSON struct {
	W          []float64 `json:"W"`
	B          float64   `json:"B"`
	Source     string    `json:"source,omitempty"`
	FeatureDim int       `json:"feature_dim,omitempty"`
}

// LoadClassifierFromJSON reads a 5-feature trust classifier from JSON.
// Returns an error if the file is missing, malformed, or has wrong feature dim.
func LoadClassifierFromJSON(path string) (ClassifierWeights, error) {
	var zero ClassifierWeights
	data, err := os.ReadFile(path)
	if err != nil {
		return zero, fmt.Errorf("trust loader: read %q: %w", path, err)
	}
	var raw classifierWeightsJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return zero, fmt.Errorf("trust loader: parse %q: %w", path, err)
	}
	if raw.FeatureDim != 0 && raw.FeatureDim != 5 {
		return zero, fmt.Errorf(
			"trust loader: feature_dim=%d, expected 5", raw.FeatureDim)
	}
	if len(raw.W) != 5 {
		return zero, fmt.Errorf(
			"trust loader: W length=%d, expected 5", len(raw.W))
	}
	var w [5]float64
	copy(w[:], raw.W)
	return ClassifierWeights{W: w, B: raw.B}, nil
}

// LoadClassifierOrFallback attempts to load trained weights from path; if any
// error occurs (typically missing checkpoint pre-G2), it returns the supplied
// fallback weights together with the error so callers can log a warning.
func LoadClassifierOrFallback(
	path string, fallback ClassifierWeights,
) (ClassifierWeights, error) {
	w, err := LoadClassifierFromJSON(path)
	if err != nil {
		return fallback, err
	}
	return w, nil
}
