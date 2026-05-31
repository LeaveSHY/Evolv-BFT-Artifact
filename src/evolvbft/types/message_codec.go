package types

import (
	"bytes"
	"crypto/ed25519"
	"encoding/gob"
	"encoding/json"
)

// EncodeMessage serializes a Message using Go's gob binary encoder.
// gob is ~5x faster than JSON for structured Go types, critical for
// 1000-node consensus where the leader processes thousands of messages/sec.
func EncodeMessage(msg *Message) ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(msg); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// DecodeMessage deserializes a Message from gob binary format.
func DecodeMessage(data []byte) (*Message, error) {
	var msg Message
	dec := gob.NewDecoder(bytes.NewReader(data))
	if err := dec.Decode(&msg); err != nil {
		return nil, err
	}
	return &msg, nil
}

// SigningBytes returns deterministic bytes for signing using JSON.
// We keep JSON here for signing because:
// 1. Signing bytes must be deterministic across implementations
// 2. gob encoding order is not guaranteed to be deterministic across versions
// 3. The signing path is called once per message, not the hot path
func (m *Message) SigningBytes() ([]byte, error) {
	if m == nil {
		return nil, nil
	}
	copyMsg := *m
	copyMsg.Signature = nil
	return json.Marshal(&copyMsg)
}

func (m *Message) Sign(privateKey PrivateKey) error {
	payload, err := m.SigningBytes()
	if err != nil {
		return err
	}
	m.Signature = ed25519.Sign(privateKey, payload)
	return nil
}

func (m *Message) VerifySignature(publicKey PublicKey) bool {
	if m == nil || len(m.Signature) == 0 {
		return false
	}
	payload, err := m.SigningBytes()
	if err != nil {
		return false
	}
	return ed25519.Verify(publicKey, payload, m.Signature)
}
