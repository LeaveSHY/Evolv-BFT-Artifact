package compress

import (
	"bytes"
	"crypto/rand"
	"testing"
)

func TestCompressDecompress_Gzip(t *testing.T) {
	// Create repetitive data (compresses well)
	data := bytes.Repeat([]byte("hello world consensus batch transaction "), 1000)

	compressed := Compress(data, CodecGzip, 100)
	if len(compressed) >= len(data) {
		t.Logf("Warning: compressed (%d) >= original (%d)", len(compressed), len(data))
	}

	decompressed, err := Decompress(compressed)
	if err != nil {
		t.Fatalf("decompress failed: %v", err)
	}
	if !bytes.Equal(decompressed, data) {
		t.Fatalf("decompressed data does not match original (got %d bytes, want %d)", len(decompressed), len(data))
	}

	ratio := Ratio(data, compressed)
	t.Logf("Compression ratio: %.3f (original=%d, compressed=%d)", ratio, len(data), len(compressed))
}

func TestCompressDecompress_None(t *testing.T) {
	data := []byte("small payload")

	compressed := Compress(data, CodecNone, 100)

	decompressed, err := Decompress(compressed)
	if err != nil {
		t.Fatalf("decompress failed: %v", err)
	}
	if !bytes.Equal(decompressed, data) {
		t.Fatal("decompressed data mismatch")
	}
}

func TestCompress_BelowMinSize(t *testing.T) {
	data := []byte("tiny")

	compressed := Compress(data, CodecGzip, 100)

	// Should NOT be gzip compressed because len(data) < minSize
	if Codec(compressed[0]) != CodecNone {
		t.Fatalf("expected no compression for small data, got codec=%d", compressed[0])
	}

	decompressed, err := Decompress(compressed)
	if err != nil {
		t.Fatalf("decompress failed: %v", err)
	}
	if !bytes.Equal(decompressed, data) {
		t.Fatal("data mismatch")
	}
}

func TestDecompress_TooShort(t *testing.T) {
	_, err := Decompress([]byte{1, 2})
	if err == nil {
		t.Fatal("expected error for short data")
	}
}

func TestDecompress_UnknownCodec(t *testing.T) {
	data := make([]byte, headerSize+10)
	data[0] = 255 // unknown codec
	_, err := Decompress(data)
	if err == nil {
		t.Fatal("expected error for unknown codec")
	}
}

func TestDecompress_DecompressionBomb(t *testing.T) {
	data := make([]byte, headerSize+5)
	data[0] = byte(CodecNone)
	// Claim uncompressed size is 100MB (exceeds 64MB limit)
	data[1] = 0x06
	data[2] = 0x40
	data[3] = 0x00
	data[4] = 0x00
	_, err := Decompress(data)
	if err == nil {
		t.Fatal("expected error for decompression bomb")
	}
}

func BenchmarkCompress_Gzip_1KB(b *testing.B) {
	data := make([]byte, 1024)
	rand.Read(data)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Compress(data, CodecGzip, 100)
	}
}

func BenchmarkCompress_Gzip_1MB(b *testing.B) {
	data := bytes.Repeat([]byte("batch transaction payload data "), 33000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Compress(data, CodecGzip, 100)
	}
}

func BenchmarkDecompress_Gzip_1MB(b *testing.B) {
	data := bytes.Repeat([]byte("batch transaction payload data "), 33000)
	compressed := Compress(data, CodecGzip, 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Decompress(compressed)
	}
}
