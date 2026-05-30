package hotstuff

import "sync"

type RankState struct {
	mu           sync.RWMutex
	instanceID   uint64
	numInstances uint64
	highestLocal uint64
}

func NewRankState(instanceID uint64, numInstances uint64) *RankState {
	if numInstances == 0 {
		numInstances = 1
	}
	return &RankState{
		instanceID:   instanceID,
		numInstances: numInstances,
	}
}

func (rs *RankState) ExpectedRank(localHeight uint64) uint64 {
	return localHeight*rs.numInstances + rs.instanceID
}

func (rs *RankState) VerifyRank(localHeight uint64, rank int64) bool {
	return rank >= 0 && uint64(rank) == rs.ExpectedRank(localHeight)
}

func (rs *RankState) OnCommit(localHeight uint64) bool {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	if localHeight <= rs.highestLocal {
		return false
	}
	rs.highestLocal = localHeight
	return true
}

func (rs *RankState) HighestLocalHeight() uint64 {
	rs.mu.RLock()
	defer rs.mu.RUnlock()
	return rs.highestLocal
}

func (rs *RankState) SetHighestLocalHeight(localHeight uint64) {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	rs.highestLocal = localHeight
}
