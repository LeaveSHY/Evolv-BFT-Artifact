package trust

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadClassifierFromJSON_OK(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "w.json")
	body := `{"W":[1.1,2.2,3.3,4.4,5.5],"B":-1.5,"source":"test","feature_dim":5}`
	if err := os.WriteFile(p, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
	w, err := LoadClassifierFromJSON(p)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	want := [5]float64{1.1, 2.2, 3.3, 4.4, 5.5}
	if w.W != want {
		t.Errorf("W = %v, want %v", w.W, want)
	}
	if w.B != -1.5 {
		t.Errorf("B = %v, want -1.5", w.B)
	}
}

func TestLoadClassifierFromJSON_WrongLen(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "w.json")
	body := `{"W":[1,2,3],"B":0}`
	if err := os.WriteFile(p, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadClassifierFromJSON(p); err == nil {
		t.Error("expected length error")
	}
}

func TestLoadClassifierFromJSON_Missing(t *testing.T) {
	if _, err := LoadClassifierFromJSON("/nonexistent/x.json"); err == nil {
		t.Error("expected file error")
	}
}

func TestLoadClassifierOrFallback(t *testing.T) {
	fb := ClassifierWeights{W: [5]float64{2, 3, 1.5, 1, 0.5}, B: -1}
	w, err := LoadClassifierOrFallback("/nonexistent/x.json", fb)
	if err == nil {
		t.Error("expected fallback error")
	}
	if w != fb {
		t.Errorf("w = %v, want fallback %v", w, fb)
	}
}
