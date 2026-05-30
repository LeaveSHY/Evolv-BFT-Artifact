package compress

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"sync"
)

// Codec defines the compression algorithm.
type Codec uint8

const (
	CodecNone Codec = 0
	CodecGzip Codec = 1
)

// Header: [1 byte codec][4 bytes uncompressed length][compressed payload]
const headerSize = 5

var gzipWriterPool = sync.Pool{
	New: func() interface{} {
		w, _ := gzip.NewWriterLevel(nil, gzip.BestSpeed)
		return w
	},
}

var gzipReaderPool = sync.Pool{
	New: func() interface{} {
		return new(gzip.Reader)
	},
}

var bufPool = sync.Pool{
	New: func() interface{} {
		return new(bytes.Buffer)
	},
}

// Compress compresses data using the specified codec.
// Returns the original data with a header prepended.
// For payloads < minSize bytes, no compression is applied (overhead not worth it).
func Compress(data []byte, codec Codec, minSize int) []byte {
	if len(data) < minSize || codec == CodecNone {
		// No compression: just prepend header
		out := make([]byte, headerSize+len(data))
		out[0] = byte(CodecNone)
		binary.BigEndian.PutUint32(out[1:5], uint32(len(data)))
		copy(out[headerSize:], data)
		return out
	}

	switch codec {
	case CodecGzip:
		return compressGzip(data)
	default:
		// Unknown codec, no compression
		out := make([]byte, headerSize+len(data))
		out[0] = byte(CodecNone)
		binary.BigEndian.PutUint32(out[1:5], uint32(len(data)))
		copy(out[headerSize:], data)
		return out
	}
}

func compressGzip(data []byte) []byte {
	buf := bufPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer bufPool.Put(buf)

	// Write header placeholder
	var header [headerSize]byte
	header[0] = byte(CodecGzip)
	binary.BigEndian.PutUint32(header[1:5], uint32(len(data)))
	buf.Write(header[:])

	w := gzipWriterPool.Get().(*gzip.Writer)
	w.Reset(buf)
	w.Write(data)
	w.Close()
	gzipWriterPool.Put(w)

	return append([]byte(nil), buf.Bytes()...)
}

// Decompress reads the header and decompresses accordingly.
func Decompress(data []byte) ([]byte, error) {
	if len(data) < headerSize {
		return nil, fmt.Errorf("compress: data too short (%d bytes)", len(data))
	}

	codec := Codec(data[0])
	uncompressedLen := binary.BigEndian.Uint32(data[1:5])
	payload := data[headerSize:]

	// Sanity check: prevent decompression bombs
	const maxDecompressedSize = 64 * 1024 * 1024 // 64MB
	if uncompressedLen > maxDecompressedSize {
		return nil, fmt.Errorf("compress: uncompressed size %d exceeds limit", uncompressedLen)
	}

	switch codec {
	case CodecNone:
		if uint32(len(payload)) != uncompressedLen {
			return nil, fmt.Errorf("compress: length mismatch (header=%d, payload=%d)", uncompressedLen, len(payload))
		}
		return payload, nil

	case CodecGzip:
		return decompressGzip(payload, uncompressedLen)

	default:
		return nil, fmt.Errorf("compress: unknown codec %d", codec)
	}
}

func decompressGzip(payload []byte, expectedLen uint32) ([]byte, error) {
	r := gzipReaderPool.Get().(*gzip.Reader)
	defer gzipReaderPool.Put(r)

	if err := r.Reset(bytes.NewReader(payload)); err != nil {
		return nil, fmt.Errorf("compress: gzip reset: %w", err)
	}
	defer r.Close()

	out := make([]byte, 0, expectedLen)
	buf := bytes.NewBuffer(out)
	if _, err := io.Copy(buf, io.LimitReader(r, int64(expectedLen)+1)); err != nil {
		return nil, fmt.Errorf("compress: gzip decompress: %w", err)
	}
	return buf.Bytes(), nil
}

// Ratio returns the compression ratio (compressed/original). Lower is better.
func Ratio(original, compressed []byte) float64 {
	if len(original) == 0 {
		return 1.0
	}
	return float64(len(compressed)) / float64(len(original))
}
