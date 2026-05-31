package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"sync"
	"time"

	"evolvbft/evolvbft/bootstrap"
	"evolvbft/evolvbft/consensus/beacon"
	"evolvbft/evolvbft/crypto"
	"go.dedis.ch/kyber/v3"
)

// SimulatedNode represents a lightweight node in the simulation
type SimulatedNode struct {
	ID         uint64
	VRFPrivKey kyber.Scalar
	VRFPubKey  kyber.Point
	Beacon     *beacon.RandomBeacon
	
	// Stats
	SelectedCount int
}

func main() {
	cfg, err := bootstrap.ParseEngineConfig(os.Args[1:])
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Secondary entry evolvbft-sim started for Node %d, primary consensus path is cmd/evolvbft -> Engine\n", cfg.NodeID)
	fmt.Println("Starting Massive Scale Simulation (100 Nodes, 1000 Rounds)...")
	
	nodeCount := 100
	rounds := 1000
	committeeSize := 10 // Target size
	
	nodes := make([]*SimulatedNode, nodeCount)
	
	// 1. Initialize Nodes
	initialSeed := []byte("genesis-seed")
	for i := 0; i < nodeCount; i++ {
		scalar, point := crypto.GenerateVRFKey()
		nodes[i] = &SimulatedNode{
			ID:         uint64(i),
			VRFPrivKey: scalar,
			VRFPubKey:  point,
			Beacon:     beacon.NewRandomBeacon(initialSeed),
		}
	}
	
	// 2. Run Simulation
	start := time.Now()
	totalCommitteeSize := 0
	
	for r := 0; r < rounds; r++ {
		// Mock consensus: generate a random signature as "aggregated sig"
		// In real system, this comes from the committee
		mockSig := make([]byte, 32)
		rand.Read(mockSig)
		
		// Election Phase
		roundCommittee := 0
		var wg sync.WaitGroup
		var mu sync.Mutex
		
		for _, node := range nodes {
			wg.Add(1)
			go func(n *SimulatedNode) {
				defer wg.Done()
				
				// Update Beacon (simulating block commit)
				n.Beacon.UpdateRandomness(mockSig)
				
				// Check eligibility
				selected, _, _ := n.Beacon.AmICommitteeMember(n.VRFPrivKey, uint64(nodeCount), committeeSize)
				
				if selected {
					mu.Lock()
					roundCommittee++
					n.SelectedCount++
					mu.Unlock()
				}
			}(node)
		}
		wg.Wait()
		
		totalCommitteeSize += roundCommittee
		// fmt.Printf("Round %d: Committee Size %d\n", r, roundCommittee)
	}
	
	duration := time.Since(start)
	avgSize := float64(totalCommitteeSize) / float64(rounds)
	
	fmt.Printf("\n--- Simulation Results ---\n")
	fmt.Printf("Total Rounds: %d\n", rounds)
	fmt.Printf("Total Time: %v\n", duration)
	fmt.Printf("Avg Time per Round: %v\n", duration/time.Duration(rounds))
	fmt.Printf("Target Committee Size: %d\n", committeeSize)
	fmt.Printf("Actual Avg Committee Size: %.2f\n", avgSize)
	
	// Check fairness
	fmt.Println("\nNode Distribution (First 10 nodes):")
	for i := 0; i < 10; i++ {
		fmt.Printf("Node %d: Selected %d times\n", i, nodes[i].SelectedCount)
	}
}
