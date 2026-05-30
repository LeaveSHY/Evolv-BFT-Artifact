package hotstuff

import "octopus-bft/octopus/types"

type InstanceOutput struct {
	InstanceID       uint64
	LocalHeight      uint64
	Rank             uint64
	BlockHash        []byte
	Block            *types.Block
	EpochTransitions []*types.EpochTransition
	IsNil            bool
}

