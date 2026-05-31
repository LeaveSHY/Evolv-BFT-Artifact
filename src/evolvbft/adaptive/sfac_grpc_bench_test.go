package adaptive

import (
	"context"
	"fmt"
	"net"
	"sort"
	"testing"
	"time"

	pb "evolvbft/evolvbft/adaptive/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// echoSFACServer implements SFACServiceServer returning immediate responses.
// This measures pure gRPC framework + JSON codec overhead.
type echoSFACServer struct{}

func (s *echoSFACServer) Decide(_ context.Context, req *pb.SFACRequest) (*pb.SFACResponse, error) {
	actions := make([]*pb.SFACAgentAction, len(req.Instances))
	for i, inst := range req.Instances {
		actions[i] = &pb.SFACAgentAction{InstanceID: inst.InstanceID}
	}
	return &pb.SFACResponse{Actions: actions, Value: 0.5}, nil
}

func (s *echoSFACServer) Feedback(_ context.Context, _ *pb.TrajectorySample) (*pb.SFACAck, error) {
	return &pb.SFACAck{Success: true}, nil
}

func (s *echoSFACServer) Reset(_ context.Context, _ *pb.ResetRequest) (*pb.SFACAck, error) {
	return &pb.SFACAck{Success: true}, nil
}

func (s *echoSFACServer) Health(_ context.Context, _ *pb.HealthRequest) (*pb.HealthResponse, error) {
	return &pb.HealthResponse{Ready: true, ModelVersion: "echo", DecisionsServed: 0}, nil
}

func TestSFACGRPCBridgeLatency(t *testing.T) {
	// Start echo gRPC server
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := lis.Addr().String()

	srv := grpc.NewServer()
	pb.RegisterSFACServiceServer(srv, &echoSFACServer{})
	go func() { _ = srv.Serve(lis) }()
	defer srv.GracefulStop()

	// Wait for server ready
	time.Sleep(50 * time.Millisecond)

	// Connect client
	conn, err := grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.CallContentSubtype("json")),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	client := pb.NewSFACServiceClient(conn)

	// Build realistic request (m=10 instances, ~25 agents each)
	req := &pb.SFACRequest{
		Epoch:        1,
		NumInstances: 10,
	}
	for i := 0; i < 10; i++ {
		inst := &pb.SFACInstanceRequest{
			InstanceID:     uint64(i),
			ValidatorCount: 25,
			Throughput:     50000,
			Latency:        12.5,
		}
		for j := 0; j < 25; j++ {
			inst.TrustFeatures = append(inst.TrustFeatures, &pb.SFACTrustFeature{
				AgentID:          uint64(j),
				TimeoutRate:      0.01,
				EquivocationRate: 0.001,
				ViewChangeRate:   0.005,
				MeanLatency:      10.5,
				StdLatency:       2.3,
			})
		}
		req.Instances = append(req.Instances, inst)
	}

	// Warmup
	for i := 0; i < 100; i++ {
		_, err := client.Decide(context.Background(), req)
		if err != nil {
			t.Fatal(err)
		}
	}

	// Benchmark
	const N = 1000
	latencies := make([]time.Duration, N)
	for i := 0; i < N; i++ {
		start := time.Now()
		_, err := client.Decide(context.Background(), req)
		latencies[i] = time.Since(start)
		if err != nil {
			t.Fatal(err)
		}
	}

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	median := latencies[N/2]
	p99 := latencies[int(0.99*float64(N))]

	var total time.Duration
	for _, l := range latencies {
		total += l
	}
	mean := total / N

	fmt.Printf("\n=== SFAC gRPC Bridge Latency (echo, m=10, 250 agents) ===\n")
	fmt.Printf("  Mean:   %v\n", mean)
	fmt.Printf("  Median: %v\n", median)
	fmt.Printf("  P99:    %v\n", p99)
	fmt.Printf("  Min:    %v\n", latencies[0])
	fmt.Printf("  Max:    %v\n", latencies[N-1])
	fmt.Printf("=== (Trust inference adds ~1ms; consensus epoch >= 100ms) ===\n\n")

	// Assert bridge overhead is <5ms (generous; typically <1ms)
	if median > 5*time.Millisecond {
		t.Errorf("gRPC bridge median latency %v exceeds 5ms budget", median)
	}
}
