package adaptive

import (
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Epoch-Boundary Parameter Gating — adaptive/epoch_gate.go
//
// Paper Mapping:
//   - §III-C Self-Evolving Adaptation: "parameters updated at epoch boundaries"
//   - §IV Theorem (liveness): adaptation preserves liveness by gating changes
//     to safe transition points (epoch boundaries, not mid-round)
//
// Mechanism:
//   The EpochGate sits between the Controller decision and the Actuator.
//   Actions produced by SFAC are NOT applied immediately — they are held
//   (gated) until the next epoch boundary arrives. This ensures:
//   (1) Consensus rounds in progress are not disrupted by parameter changes
//   (2) A minimum cooldown between consecutive actions prevents oscillation
//   (3) Emergency safety overrides can bypass gating when n < 3f+1 is imminent
//
//   Flow: Controller.Tick() → EpochGate.Submit(action) → [hold]
//         Runtime.AdvanceEpoch() → EpochGate.Release() → Actuator.Apply()
// ═══════════════════════════════════════════════════════════════════════════════

// EpochGateConfig controls the gating behavior.
type EpochGateConfig struct {
	// MinCooldownEpochs: minimum epochs between consecutive parameter changes.
	// Prevents rapid oscillation. Default: 2 (allow change every other epoch).
	MinCooldownEpochs uint64

	// AllowEmergencyBypass: if true, actions marked as emergency (safety-critical)
	// are applied immediately without waiting for epoch boundary.
	AllowEmergencyBypass bool

	// MaxPendingAge: if a pending action has waited longer than this duration
	// without an epoch advance, it is dropped (stale). 0 = no staleness check.
	MaxPendingAge time.Duration
}

// DefaultEpochGateConfig returns production defaults.
func DefaultEpochGateConfig() EpochGateConfig {
	return EpochGateConfig{
		MinCooldownEpochs:    2,
		AllowEmergencyBypass: true,
		MaxPendingAge:        30 * time.Second,
	}
}

// GateResult indicates what happened when an action was submitted.
type GateResult int

const (
	// GateApplied: action was applied immediately (first action or cooldown satisfied).
	GateApplied GateResult = iota
	// GateDeferred: action is held until next epoch boundary.
	GateDeferred
	// GateDropped: a newer action replaced the pending one (only latest kept).
	GateDropped
	// GateEmergency: action bypassed gating due to emergency flag.
	GateEmergency
)

// PendingAction wraps an action awaiting epoch boundary release.
type PendingAction struct {
	Action      Action
	SubmittedAt time.Time
	Epoch       uint64 // epoch at which it was submitted
}

// EpochGateStats provides monitoring diagnostics.
type EpochGateStats struct {
	CurrentEpoch      uint64 `json:"current_epoch"`
	LastAppliedEpoch  uint64 `json:"last_applied_epoch"`
	TotalSubmitted    int    `json:"total_submitted"`
	TotalApplied      int    `json:"total_applied"`
	TotalDeferred     int    `json:"total_deferred"`
	TotalDropped      int    `json:"total_dropped"`
	TotalEmergency    int    `json:"total_emergency"`
	TotalStale        int    `json:"total_stale"`
	HasPending        bool   `json:"has_pending"`
}

// EpochGate gates parameter changes to epoch boundaries for liveness preservation.
type EpochGate struct {
	mu               sync.Mutex
	config           EpochGateConfig
	currentEpoch     uint64
	lastAppliedEpoch uint64
	hasApplied       bool
	pending          *PendingAction
	totalSubmitted   int
	totalApplied     int
	totalDeferred    int
	totalDropped     int
	totalEmergency   int
	totalStale       int
}

// NewEpochGate creates an epoch-boundary parameter gate.
func NewEpochGate(config EpochGateConfig) *EpochGate {
	if config.MinCooldownEpochs == 0 {
		config.MinCooldownEpochs = 1
	}
	return &EpochGate{config: config}
}

// Submit proposes an action for gated application.
// Returns GateResult indicating immediate application, deferral, or replacement.
// If the action is applied immediately, the caller should execute it.
// If deferred, the caller should NOT execute — call Release() at epoch boundary.
func (eg *EpochGate) Submit(action Action, isEmergency bool, now time.Time) GateResult {
	eg.mu.Lock()
	defer eg.mu.Unlock()

	eg.totalSubmitted++

	// Emergency bypass: safety-critical actions (e.g., n approaching 3f+1)
	if isEmergency && eg.config.AllowEmergencyBypass {
		eg.totalEmergency++
		eg.lastAppliedEpoch = eg.currentEpoch
		eg.hasApplied = true
		eg.pending = nil // clear any pending
		return GateEmergency
	}

	// Check cooldown: can we apply at current epoch?
	if eg.canApplyAt(eg.currentEpoch) {
		eg.totalApplied++
		eg.lastAppliedEpoch = eg.currentEpoch
		eg.hasApplied = true
		eg.pending = nil
		return GateApplied
	}

	// Defer: hold until next epoch boundary
	result := GateDeferred
	if eg.pending != nil {
		// Replace existing pending (only latest action matters)
		eg.totalDropped++
		result = GateDropped
	}
	eg.pending = &PendingAction{
		Action:      action,
		SubmittedAt: now,
		Epoch:       eg.currentEpoch,
	}
	eg.totalDeferred++
	return result
}

// AdvanceEpoch notifies the gate that a new epoch has started.
// Returns the pending action if it should now be applied (cooldown satisfied),
// or nil if no action is ready.
func (eg *EpochGate) AdvanceEpoch(newEpoch uint64, now time.Time) *Action {
	eg.mu.Lock()
	defer eg.mu.Unlock()

	if newEpoch <= eg.currentEpoch {
		return nil // no actual advance
	}
	eg.currentEpoch = newEpoch

	if eg.pending == nil {
		return nil
	}

	// Check staleness
	if eg.config.MaxPendingAge > 0 && now.Sub(eg.pending.SubmittedAt) > eg.config.MaxPendingAge {
		eg.totalStale++
		eg.pending = nil
		return nil
	}

	// Check cooldown against new epoch
	if eg.canApplyAt(newEpoch) {
		action := eg.pending.Action
		eg.pending = nil
		eg.totalApplied++
		eg.lastAppliedEpoch = newEpoch
		eg.hasApplied = true
		return &action
	}

	return nil
}

// HasPending returns whether an action is waiting for release.
func (eg *EpochGate) HasPending() bool {
	eg.mu.Lock()
	defer eg.mu.Unlock()
	return eg.pending != nil
}

// CurrentEpoch returns the gate current epoch.
func (eg *EpochGate) CurrentEpoch() uint64 {
	eg.mu.Lock()
	defer eg.mu.Unlock()
	return eg.currentEpoch
}

// Stats returns monitoring diagnostics.
func (eg *EpochGate) Stats() EpochGateStats {
	eg.mu.Lock()
	defer eg.mu.Unlock()
	return EpochGateStats{
		CurrentEpoch:     eg.currentEpoch,
		LastAppliedEpoch: eg.lastAppliedEpoch,
		TotalSubmitted:   eg.totalSubmitted,
		TotalApplied:     eg.totalApplied,
		TotalDeferred:    eg.totalDeferred,
		TotalDropped:     eg.totalDropped,
		TotalEmergency:   eg.totalEmergency,
		TotalStale:       eg.totalStale,
		HasPending:       eg.pending != nil,
	}
}

// canApplyAt checks if cooldown has elapsed relative to newEpoch.
func (eg *EpochGate) canApplyAt(epoch uint64) bool {
	if !eg.hasApplied {
		return true // first action ever
	}
	return epoch >= eg.lastAppliedEpoch+eg.config.MinCooldownEpochs
}
