package hotstuff

import (
	"sync"
	"time"
)

type GlobalOrderer struct {
	mu sync.Mutex

	numInstances uint64
	nextRank     uint64
	barTimeout   time.Duration
	maxPending   int

	pending      map[uint64]InstanceOutput
	missingSince map[uint64]time.Time
	log          []InstanceOutput

	emitted uint64
	nilled  uint64
	late    uint64
	dropped uint64

	outChan chan InstanceOutput
	stopCh  chan struct{}
}

func NewGlobalOrderer(numInstances uint64, barTimeout time.Duration) *GlobalOrderer {
	return NewGlobalOrdererWithLimit(numInstances, barTimeout, 8192)
}

func NewGlobalOrdererWithLimit(numInstances uint64, barTimeout time.Duration, maxPending int) *GlobalOrderer {
	if numInstances == 0 {
		numInstances = 1
	}
	if barTimeout <= 0 {
		barTimeout = 2 * time.Second
	}
	if maxPending <= 0 {
		maxPending = 8192
	}
	return &GlobalOrderer{
		numInstances: numInstances,
		nextRank:     numInstances,
		barTimeout:   barTimeout,
		maxPending:   maxPending,
		pending:      make(map[uint64]InstanceOutput),
		missingSince: make(map[uint64]time.Time),
		log:          make([]InstanceOutput, 0, 1024),
		outChan:      make(chan InstanceOutput, 4096),
		stopCh:       make(chan struct{}),
	}
}

func (o *GlobalOrderer) Start(in <-chan InstanceOutput) <-chan InstanceOutput {
	go o.run(in)
	return o.outChan
}

func (o *GlobalOrderer) Stop() {
	close(o.stopCh)
}

func (o *GlobalOrderer) Stats() (emitted uint64, nilled uint64, late uint64) {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.emitted, o.nilled, o.late
}

func (o *GlobalOrderer) Dropped() uint64 {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.dropped
}

func (o *GlobalOrderer) BacklogStats() (pending uint64, missing uint64) {
	o.mu.Lock()
	defer o.mu.Unlock()
	return uint64(len(o.pending)), uint64(len(o.missingSince))
}

func (o *GlobalOrderer) Log() []InstanceOutput {
	o.mu.Lock()
	defer o.mu.Unlock()
	cp := make([]InstanceOutput, len(o.log))
	copy(cp, o.log)
	return cp
}

func (o *GlobalOrderer) run(in <-chan InstanceOutput) {
	ticker := time.NewTicker(o.barTimeout / 4)
	if o.barTimeout/4 <= 0 {
		ticker.Stop()
		ticker = time.NewTicker(500 * time.Millisecond)
	}
	defer ticker.Stop()

	for {
		select {
		case out := <-in:
			o.onInput(out)
			o.flush(false)
		case <-ticker.C:
			o.flush(true)
		case <-o.stopCh:
			close(o.outChan)
			return
		}
	}
}

func (o *GlobalOrderer) onInput(out InstanceOutput) {
	o.mu.Lock()
	defer o.mu.Unlock()

	if out.Rank < o.nextRank {
		o.late++
		return
	}
	if !o.validRankMapping(out) {
		return
	}
	if _, exists := o.pending[out.Rank]; exists {
		return
	}
	if len(o.pending) >= o.maxPending {
		maxRank := uint64(0)
		maxSet := false
		for rank := range o.pending {
			if !maxSet || rank > maxRank {
				maxRank = rank
				maxSet = true
			}
		}
		if !maxSet || out.Rank >= maxRank {
			o.dropped++
			return
		}
		delete(o.pending, maxRank)
		o.dropped++
	}
	o.pending[out.Rank] = out
}

func (o *GlobalOrderer) validRankMapping(out InstanceOutput) bool {
	if o.numInstances == 0 {
		return true
	}
	expectedInstanceID := out.Rank % o.numInstances
	expectedLocalHeight := out.Rank / o.numInstances
	if out.InstanceID != expectedInstanceID {
		return false
	}
	if out.LocalHeight != expectedLocalHeight {
		return false
	}
	return true
}

func (o *GlobalOrderer) flush(fromTick bool) {
	_ = fromTick
	for {
		next, ok, emitNil := o.nextToEmit()
		if !ok {
			return
		}
		_ = emitNil
		o.outChan <- next
	}
}

func (o *GlobalOrderer) nextToEmit() (entry InstanceOutput, ok bool, emitNil bool) {
	o.mu.Lock()
	defer o.mu.Unlock()

	if v, exists := o.pending[o.nextRank]; exists {
		delete(o.pending, o.nextRank)
		delete(o.missingSince, o.nextRank)
		o.log = append(o.log, v)
		o.emitted++
		o.nextRank++
		return v, true, false
	}

	since, exists := o.missingSince[o.nextRank]
	if !exists {
		o.missingSince[o.nextRank] = time.Now()
		return InstanceOutput{}, false, false
	}
	if time.Since(since) < o.barTimeout {
		return InstanceOutput{}, false, false
	}

	instanceID := o.nextRank % o.numInstances
	localHeight := o.nextRank / o.numInstances
	nilEntry := InstanceOutput{
		InstanceID:  instanceID,
		LocalHeight: localHeight,
		Rank:        o.nextRank,
		IsNil:       true,
	}
	delete(o.missingSince, o.nextRank)
	o.log = append(o.log, nilEntry)
	o.nilled++
	o.nextRank++
	return nilEntry, true, true
}
