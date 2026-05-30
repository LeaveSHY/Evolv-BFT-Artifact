package main

import (
	"crypto/ed25519"
	"errors"
	"fmt"
	"sort"

	"octopus-bft/octopus/consensus/gbc"
	"octopus-bft/octopus/consensus/hotstuff"
	"octopus-bft/octopus/network/libp2p"
	"octopus-bft/octopus/types"
)

var errNotGBCMember = errors.New("node is not a current GBC primary")

func isNotGBCMember(err error) bool {
	return errors.Is(err, errNotGBCMember)
}

type gbcPubSubAdapter struct {
	network *libp2p.P2PNetwork
}

func (a *gbcPubSubAdapter) PublishToTopic(topic string, data []byte) error {
	return a.network.PublishTopic(topic, data)
}

func (a *gbcPubSubAdapter) SubscribeToTopic(topic string, bufSize int) (<-chan []byte, error) {
	return a.network.SubscribeTopic(topic, bufSize)
}

func (a *gbcPubSubAdapter) SendDirect(peerID uint64, data []byte) error {
	_ = peerID
	_ = data
	return fmt.Errorf("gbc direct stream unavailable, using pubsub target routing")
}

func newRuntimeGBCNode(nodeID uint64, privateKey ed25519.PrivateKey, validators map[uint64]*types.Validator, network *libp2p.P2PNetwork, baseTopic string, inboxBuffer int) (*gbc.Node, error) {
	if network == nil {
		return nil, fmt.Errorf("network is nil")
	}
	if _, ok := validators[nodeID]; !ok {
		return nil, fmt.Errorf("%w: node %d", errNotGBCMember, nodeID)
	}
	peerIDs := sortedValidatorIDs(validators)
	publicKeys := make(map[uint64]ed25519.PublicKey, len(validators))
	for id, validator := range validators {
		if validator == nil || len(validator.PublicKey) != ed25519.PublicKeySize {
			return nil, fmt.Errorf("validator %d has invalid public key", id)
		}
		publicKeys[id] = validator.PublicKey
	}
	transport, err := gbc.NewP2PTransport(gbc.P2PTransportConfig{
		NodeID:      nodeID,
		PeerIDs:     peerIDs,
		Network:     &gbcPubSubAdapter{network: network},
		Topic:       baseTopic + ".gbc",
		InboxBuffer: inboxBuffer,
	})
	if err != nil {
		return nil, err
	}
	return gbc.NewNode(gbc.NodeConfig{
		NodeID:     nodeID,
		NumMembers: len(peerIDs),
		MemberIDs:  peerIDs,
		PrivateKey: privateKey,
		PublicKeys: publicKeys,
	}, transport), nil
}

func currentGBCPrimaries(engines []*hotstuff.Engine, validators map[uint64]*types.Validator) map[uint64]*types.Validator {
	primaries := make(map[uint64]*types.Validator)
	for _, engine := range engines {
		if engine == nil {
			continue
		}
		leaderID := engine.CurrentLeader()
		validator, ok := validators[leaderID]
		if !ok || validator == nil {
			continue
		}
		primaries[leaderID] = validator
	}
	return primaries
}

func sortedValidatorIDs(validators map[uint64]*types.Validator) []uint64 {
	ids := make([]uint64, 0, len(validators))
	for id := range validators {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}
