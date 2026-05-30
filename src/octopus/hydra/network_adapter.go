// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package hydra

import (
	"encoding/json"
	"log"

	"octopus-bft/octopus/network/libp2p"
)

// HydraProtocolID is the libp2p protocol for Hydra messages (auto-transition, discovery).
const HydraProtocolID = "/octopus/hydra/1.0.0"

// HydraTopicName is the GossipSub topic for Hydra broadcast messages.
const HydraTopicName = "octopus-hydra"

// NetworkAdapter wraps P2PNetwork to implement Hydra's NetworkInterface.
// This bridges the gap between Hydra's generic Broadcast/Send and
// the concrete libp2p network layer.
type NetworkAdapter struct {
	net    *libp2p.P2PNetwork
	nodeID uint64
}

// NewNetworkAdapter creates a NetworkAdapter that satisfies hydra.NetworkInterface.
func NewNetworkAdapter(net *libp2p.P2PNetwork, nodeID uint64) *NetworkAdapter {
	return &NetworkAdapter{
		net:    net,
		nodeID: nodeID,
	}
}

// Broadcast publishes a Hydra message to all peers via GossipSub.
func (na *NetworkAdapter) Broadcast(msg interface{}) {
	data, err := json.Marshal(msg)
	if err != nil {
		log.Printf("Hydra NetworkAdapter: failed to marshal broadcast message: %v", err)
		return
	}
	if err := na.net.PublishTopic(HydraTopicName, data); err != nil {
		log.Printf("Hydra NetworkAdapter: failed to broadcast: %v", err)
	}
}

// Send sends a Hydra message to a specific node via direct stream.
func (na *NetworkAdapter) Send(to uint64, msg interface{}) {
	data, err := json.Marshal(msg)
	if err != nil {
		log.Printf("Hydra NetworkAdapter: failed to marshal send message: %v", err)
		return
	}
	if err := na.net.SendToNode(to, HydraProtocolID, data); err != nil {
		// Fallback to broadcast if direct send fails
		log.Printf("Hydra NetworkAdapter: direct send to node %d failed (%v), falling back to broadcast", to, err)
		na.Broadcast(msg)
	}
}

// Ensure NetworkAdapter implements NetworkInterface at compile time
var _ NetworkInterface = (*NetworkAdapter)(nil)
