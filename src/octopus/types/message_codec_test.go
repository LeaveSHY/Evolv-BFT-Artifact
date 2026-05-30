package types

import (
	"crypto/ed25519"
	"crypto/rand"
	"testing"
)

func TestMessageCodecRoundTrip(t *testing.T) {
	parent := make([]byte, 32)
	justify := NewQuorumCertificateWithIdentity(parent, 3, 1, 7, 2, PhasePrepare)
	block := NewBlock(4, parent, []byte("payload"), 4, 1, 0, 0, justify, []byte("seed"))
	block.ConfigID = 7
	block.LaneID = 2
	block.Hash = block.ComputeHash()

	var blockID Hash
	copy(blockID[:], block.Hash)
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate key failed: %v", err)
	}
	vote, err := NewVoteWithIdentity(blockID, block.View, block.Epoch, block.ConfigID, block.LaneID, pub, priv, nil)
	if err != nil {
		t.Fatalf("create vote failed: %v", err)
	}

	msg := &Message{
		Type:          MsgProposal,
		SenderID:      1,
		View:          block.View,
		Epoch:         block.Epoch,
		Height:        block.Height,
		Block:         block,
		Vote:          vote,
		QC:            justify,
		ConfigID:      block.ConfigID,
		Lane:          block.LaneID,
		LeaderSetHash: []byte("leaders"),
		BarrierView:   block.View,
	}

	data, err := EncodeMessage(msg)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	decoded, err := DecodeMessage(data)
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	if decoded.Type != MsgProposal {
		t.Fatalf("unexpected message type: %v", decoded.Type)
	}
	if decoded.Block == nil || decoded.Block.View != block.View {
		t.Fatalf("decoded block mismatch")
	}
	if decoded.QC == nil || decoded.QC.View != justify.View {
		t.Fatalf("decoded QC mismatch")
	}
	if decoded.ConfigID != block.ConfigID || decoded.Lane != block.LaneID {
		t.Fatalf("decoded identity mismatch")
	}
}
