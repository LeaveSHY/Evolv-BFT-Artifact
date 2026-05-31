package adaptive

// ═══════════════════════════════════════════════════════════════════════════════
// Versioned Adaptive Configuration with Reversible Rollback (§III-D, §IV App. B)
//
// The adaptive layer commits configuration changes as a versioned, append-only
// chain anchored in the Global Beacon Chain (GBC). Each committed version records
// the safety margin Phi_t (Definition 3) observed at commit time. When an applied
// version later regresses safety or liveness, the layer reverts to the nearest
// COMMITTED ancestor whose margin still meets the threshold delta_s.
//
// This adds a third defense gate on top of the existing two:
//   Gate 1 (EVALUATE/GUARDRAIL): SafetyFilter masks unsafe actions pre-commit.
//   Gate 2 (COMMIT): GBC quorum (2f+1 attestations, G4) durably logs the change.
//   Gate 3 (ROLLBACK): objective triggers revert to a proven-safe ancestor.
//
// Rollback is NOT a new trust source. It reduces to the existing invariant
// Phi >= delta_s plus quorum intersection. Every reachable effective config is a
// COMMITTED version that satisfied the safety filter when it was logged.
// ═══════════════════════════════════════════════════════════════════════════════

// VersionStatus labels the lifecycle state of a configuration version.
type VersionStatus string

const (
	// StatusProposed marks a version that has been formed but not yet GBC-committed.
	StatusProposed VersionStatus = "PROPOSED"
	// StatusCommitted marks a version durably logged by GBC quorum (G4) and applied.
	StatusCommitted VersionStatus = "COMMITTED"
	// StatusRolledBack marks a version produced by a rollback to a safe ancestor.
	StatusRolledBack VersionStatus = "ROLLEDBACK"
)

// InstanceConfig is the per-instance membership snapshot inside a version.
// Restoring these values pins each instance quorum invariant n_v >= 3f_v+1+delta_s.
type InstanceConfig struct {
	InstanceID     uint64 `json:"instance_id"`
	ValidatorCount int    `json:"validator_count"`
	FaultBound     int    `json:"fault_bound"`
}

// ConfigParams is the parameter vector carried by one configuration version.
// It mirrors the per-instance knobs the adaptive layer may tune.
type ConfigParams struct {
	CommitteeSize             int `json:"committee_size"`
	PacemakerTimeoutMs        int `json:"pacemaker_timeout_ms"`
	MempoolMaxBatchTxs        int `json:"mempool_max_batch_txs"`
	MempoolProposalIntervalMs int `json:"mempool_proposal_interval_ms"`
	// ValidatorCount and FaultBound pin the BFT quorum invariant n >= 3f+1+delta_s
	// for the effective config, so a restored ancestor remains provably safe.
	ValidatorCount int `json:"validator_count"`
	FaultBound     int `json:"fault_bound"`
	// InstanceCount is the number of BFT instances m in the effective config.
	InstanceCount int `json:"instance_count"`
	// Instances is the per-instance membership snapshot, enabling faithful restore.
	Instances []InstanceConfig `json:"instances,omitempty"`
}

// ConfigVersion is one record in the append-only configuration chain.
// Versions form a tree via ParentID; the effective config is the latest entry.
type ConfigVersion struct {
	VersionID   uint64        `json:"version_id"`
	ParentID    uint64        `json:"parent_id"`
	GBCHeight   uint64        `json:"gbc_height"`
	Params      ConfigParams  `json:"params"`
	PhiAtCommit int           `json:"phi_at_commit"`
	Status      VersionStatus `json:"status"`
}

// IsSafe reports whether the version met the safety threshold delta_s at commit.
func (v ConfigVersion) IsSafe(deltaS int) bool {
	return v.PhiAtCommit >= deltaS
}

// VersionChain is the append-only lineage of configuration versions.
// It is single-writer (the GBC commit path) and not safe for concurrent
// mutation; callers serialize Append* under their own lock.
type VersionChain struct {
	versions []ConfigVersion          // append order; index 0 is genesis
	byID     map[uint64]ConfigVersion // lineage lookup
	nextID   uint64
}

// NewVersionChain seeds the chain with a genesis version (Assumption A2).
// The genesis must be safe: phiAtCommit >= delta_s is the caller's obligation.
func NewVersionChain(genesis ConfigParams, phiAtCommit, gbcHeight int) *VersionChain {
	vc := &VersionChain{
		byID:   make(map[uint64]ConfigVersion),
		nextID: 1,
	}
	v0 := ConfigVersion{
		VersionID:   0,
		ParentID:    0,
		GBCHeight:   uint64(gbcHeight),
		Params:      genesis,
		PhiAtCommit: phiAtCommit,
		Status:      StatusCommitted,
	}
	vc.versions = append(vc.versions, v0)
	vc.byID[0] = v0
	return vc
}

// Latest returns the most recently appended version (the effective config).
func (vc *VersionChain) Latest() ConfigVersion {
	return vc.versions[len(vc.versions)-1]
}

// Get returns the version with the given ID.
func (vc *VersionChain) Get(id uint64) (ConfigVersion, bool) {
	v, ok := vc.byID[id]
	return v, ok
}

// Len returns the number of versions in the chain.
func (vc *VersionChain) Len() int { return len(vc.versions) }

// Versions returns a copy of the lineage in append order.
func (vc *VersionChain) Versions() []ConfigVersion {
	out := make([]ConfigVersion, len(vc.versions))
	copy(out, vc.versions)
	return out
}

// AppendCommitted records a new COMMITTED version whose parent is the current
// latest entry. Callers invoke this only AFTER the GBC publish succeeds
// (commit-then-apply), so a COMMITTED version is always durably logged.
func (vc *VersionChain) AppendCommitted(params ConfigParams, phiAtCommit, gbcHeight int) ConfigVersion {
	parent := vc.Latest()
	v := ConfigVersion{
		VersionID:   vc.nextID,
		ParentID:    parent.VersionID,
		GBCHeight:   uint64(gbcHeight),
		Params:      params,
		PhiAtCommit: phiAtCommit,
		Status:      StatusCommitted,
	}
	vc.nextID++
	vc.versions = append(vc.versions, v)
	vc.byID[v.VersionID] = v
	return v
}

// SafeAncestor walks parent pointers from the current latest version and
// returns the nearest COMMITTED ancestor that satisfies Phi >= delta_s.
// The genesis (Assumption A2) guarantees a safe ancestor always exists once
// the chain has more than one entry. Returns false only at the genesis.
func (vc *VersionChain) SafeAncestor(deltaS int) (ConfigVersion, bool) {
	cur := vc.Latest()
	id := cur.ParentID
	visited := 0
	for {
		anc, ok := vc.byID[id]
		if !ok {
			return ConfigVersion{}, false
		}
		if anc.Status == StatusCommitted && anc.IsSafe(deltaS) {
			return anc, true
		}
		if anc.VersionID == 0 {
			// Genesis reached. By A2 it is safe; return it as the floor.
			if anc.IsSafe(deltaS) {
				return anc, true
			}
			return ConfigVersion{}, false
		}
		id = anc.ParentID
		visited++
		if visited > len(vc.versions) {
			return ConfigVersion{}, false // cycle guard (should be unreachable)
		}
	}
}

// AppendRollback records a ROLLEDBACK version that restores the parameters of
// the given safe target. The new entry's parent is the current latest version,
// preserving the append-only lineage. The restored PhiAtCommit equals the
// target's, so the effective config provably satisfies Phi >= delta_s.
func (vc *VersionChain) AppendRollback(target ConfigVersion, gbcHeight int) ConfigVersion {
	parent := vc.Latest()
	v := ConfigVersion{
		VersionID:   vc.nextID,
		ParentID:    parent.VersionID,
		GBCHeight:   uint64(gbcHeight),
		Params:      target.Params,
		PhiAtCommit: target.PhiAtCommit,
		Status:      StatusRolledBack,
	}
	vc.nextID++
	vc.versions = append(vc.versions, v)
	vc.byID[v.VersionID] = v
	return v
}

// ObservedSafety carries the runtime signals that the rollback decision reads.
// All fields are objective and on-chain verifiable from GBC-logged metadata.
type ObservedSafety struct {
	// Phi is the current joint safety invariant value (ComputePhi, Definition 3).
	Phi int
	// ViewStalled is true when a view stall beyond 2*Delta is observed (liveness).
	ViewStalled bool
	// ThroughputTPS is the observed committed throughput.
	ThroughputTPS float64
	// ThroughputFloor is the minimum acceptable throughput tau over the window W.
	// A value <= 0 disables the throughput trigger.
	ThroughputFloor float64
}

// RollbackTriggered reports whether any objective trigger fires for the
// observed signals. Triggers (all on-chain verifiable):
//
//	(R1) Phi < 0                          joint safety invariant violated
//	(R2) ViewStalled                      liveness regression (view stall > 2*Delta)
//	(R3) ThroughputTPS < ThroughputFloor  throughput floor breached (when floor > 0)
func (o ObservedSafety) RollbackTriggered() bool {
	if o.Phi < 0 {
		return true
	}
	if o.ViewStalled {
		return true
	}
	if o.ThroughputFloor > 0 && o.ThroughputTPS < o.ThroughputFloor {
		return true
	}
	return false
}

// EvaluateRollback is the pure rollback decision (no side effects).
// It returns the safe target ancestor and true iff a trigger fires AND a safe
// COMMITTED ancestor exists. When no trigger fires, or the chain is at genesis,
// it returns (zero, false). The caller publishes the rollback to GBC and only
// then applies it locally (commit-then-apply).
func EvaluateRollback(chain *VersionChain, observed ObservedSafety, deltaS int) (ConfigVersion, bool) {
	if chain == nil || chain.Len() == 0 {
		return ConfigVersion{}, false
	}
	if !observed.RollbackTriggered() {
		return ConfigVersion{}, false
	}
	return chain.SafeAncestor(deltaS)
}
