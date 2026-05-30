// Package proto provides JSON-based gRPC codec for SFAC service.
// This avoids protoc codegen while maintaining full gRPC semantics.
package proto

import (
	"encoding/json"

	"google.golang.org/grpc"
	"google.golang.org/grpc/encoding"
)

func init() {
	encoding.RegisterCodec(jsonCodecInstance{})
}

// jsonCodecInstance implements encoding.Codec using JSON serialization.
type jsonCodecInstance struct{}

func (jsonCodecInstance) Marshal(v interface{}) ([]byte, error) {
	return json.Marshal(v)
}

func (jsonCodecInstance) Unmarshal(data []byte, v interface{}) error {
	return json.Unmarshal(data, v)
}

func (jsonCodecInstance) Name() string {
	return "json"
}

// JSONCallOption returns a grpc.CallOption that forces JSON encoding.
func JSONCallOption() grpc.CallOption {
	return grpc.CallContentSubtype("json")
}
