// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package libp2p

import (
	"fmt"
)

// PublishConsensus publishes a message to the consensus topic
func (n *P2PNetwork) PublishConsensus(data []byte) error {
	if n.consensusTopic == nil {
		return fmt.Errorf("consensus topic not joined")
	}
	n.totalBytesSent.Add(uint64(len(data)))
	return n.consensusTopic.Publish(n.ctx, data)
}

// PublishMempool publishes a message to the mempool topic
func (n *P2PNetwork) PublishMempool(data []byte) error {
	if n.mempoolTopic == nil {
		return fmt.Errorf("mempool topic not joined")
	}
	n.totalBytesSent.Add(uint64(len(data)))
	return n.mempoolTopic.Publish(n.ctx, data)
}

// SubscribeConsensus subscribes to the consensus topic
func (n *P2PNetwork) SubscribeConsensus() (<-chan []byte, error) {
	if n.consensusSub != nil {
		return n.ConsensusChan, nil
	}

	if n.consensusTopic == nil {
		if err := n.JoinConsensusTopic(); err != nil {
			return nil, err
		}
	}

	return n.ConsensusChan, nil
}

// SubscribeMempool subscribes to the mempool topic
func (n *P2PNetwork) SubscribeMempool() (<-chan []byte, error) {
	if n.mempoolSub != nil {
		return n.MempoolChan, nil
	}

	if n.mempoolTopic == nil {
		if err := n.JoinMempoolTopic(); err != nil {
			return nil, err
		}
	}

	return n.MempoolChan, nil
}

// RegisterValidators registers pubsub message validators.
// Topic-level validation is handled by RegisterTopicPolicy in host.go.
// This method is kept for API compatibility.
func (n *P2PNetwork) RegisterValidators() {
	// Validation is handled via RegisterTopicPolicy + ensureTopicValidator
}
