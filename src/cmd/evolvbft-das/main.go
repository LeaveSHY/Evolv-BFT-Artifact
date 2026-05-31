package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"evolvbft/evolvbft/bootstrap"
	"evolvbft/evolvbft/coding"
	"evolvbft/evolvbft/crypto"
)

// DASNode represents a node in the DAS network
type DASNode struct {
	ID        int
	Stored    map[int][]byte // Shard Index -> Data
	Available bool
}

func main() {
	cfg, err := bootstrap.ParseEngineConfig(os.Args[1:])
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Secondary entry evolvbft-das started for Node %d, primary consensus path is cmd/evolvbft -> Engine\n", cfg.NodeID)
	fmt.Println("Starting DAS Simulation (1000 Nodes)...")

	nodeCount := 1000
	dataShards := 10
	parityShards := 20 // 2/3 erasure coding redundancy
	totalShards := dataShards + parityShards
	blockSize := 1024 * 1024 // 1MB block

	// 1. Initialize Network
	nodes := make([]*DASNode, nodeCount)
	for i := 0; i < nodeCount; i++ {
		nodes[i] = &DASNode{
			ID:     i,
			Stored: make(map[int][]byte),
		}
	}

	// 2. Create Block & Encode
	fmt.Println("Encoding 1MB Block...")
	encoder, _ := coding.NewEncoder(dataShards, parityShards)
	data := make([]byte, blockSize)
	rand.Read(data)

	shards, err := encoder.Encode(data)
	if err != nil {
		panic(err)
	}

	// 3. Build Merkle Tree
	fmt.Println("Building Merkle Tree...")
	merkleTree := crypto.BuildMerkleTree(shards)
	root := merkleTree.Root
	fmt.Printf("Merkle Root: %x\n", root)

	// 4. Disseminate (Sharding)
	// Each shard is stored by (nodeCount / totalShards) nodes
	// Simple DHT: Shard i stored at Node i, i+totalShards, ...
	fmt.Println("Disseminating Shards...")
	start := time.Now()
	totalBytes := 0
	
	for i, shard := range shards {
		// Replicate each shard to ensure availability
		replicationFactor := nodeCount / totalShards
		for r := 0; r < replicationFactor; r++ {
			targetNodeID := (i + r*totalShards) % nodeCount
			nodes[targetNodeID].Stored[i] = shard
			totalBytes += len(shard)
		}
	}
	fmt.Printf("Dissemination took %v. Total Network Traffic: %d MB\n", time.Since(start), totalBytes/1024/1024)

	// 5. DAS Simulation (Sampling)
	// Simulate 100 random observers sampling 30 shards each
	observerCount := 100
	samplesPerObserver := 30
	successCount := 0
	
	fmt.Printf("Running DAS with %d observers, %d samples each...\n", observerCount, samplesPerObserver)
	start = time.Now()
	
	for i := 0; i < observerCount; i++ {
		// Randomly select samples
		indices := rand.Perm(totalShards)[:samplesPerObserver]
		collected := 0
		
		for _, idx := range indices {
			// Query network for shard[idx]
			// In real network: DHT lookup. Here: direct lookup
			// Find a node that has it
			found := false
			replicationFactor := nodeCount / totalShards
			for r := 0; r < replicationFactor; r++ {
				targetNodeID := (idx + r*totalShards) % nodeCount
				if shard, ok := nodes[targetNodeID].Stored[idx]; ok {
					// Verify Merkle Proof
					proof := merkleTree.GetProof(idx)
					if crypto.VerifyProof(root, shard, proof, idx) {
						found = true
						collected++
						break
					}
				}
			}
			if !found {
				fmt.Printf("Observer %d failed to find shard %d\n", i, idx)
			}
		}
		
		// If we collected all samples, we are confident
		if collected == samplesPerObserver {
			successCount++
		}
	}
	
	fmt.Printf("DAS took %v\n", time.Since(start))
	fmt.Printf("Success Rate: %d/%d (%.2f%%)\n", successCount, observerCount, float64(successCount)/float64(observerCount)*100)
	
	// 6. Reconstruction Test (Full Node)
	fmt.Println("Attempting Full Reconstruction from Committee...")
	committeeSize := 50
	collectedShards := make([][]byte, totalShards)
	shardsFound := 0
	
	// Simulate committee gathering shards
	for i := 0; i < committeeSize; i++ {
		// Committee members share what they have
		for idx, shard := range nodes[i].Stored {
			if collectedShards[idx] == nil {
				collectedShards[idx] = shard
				shardsFound++
			}
		}
	}
	
	if shardsFound >= dataShards {
		recovered, err := encoder.Reconstruct(collectedShards)
		if err != nil {
			fmt.Printf("Reconstruction failed: %v\n", err)
		} else {
			fmt.Printf("Reconstruction Successful! Recovered %d bytes.\n", len(recovered))
		}
	} else {
		fmt.Printf("Not enough shards for reconstruction: %d/%d\n", shardsFound, dataShards)
	}
}
