package main

import "testing"

func TestBuildP2PMultiaddr_IPv4(t *testing.T) {
	got, err := buildP2PMultiaddr("127.0.0.1", 8080, "12D3KooWtest")
	if err != nil {
		t.Fatalf("build p2p multiaddr: %v", err)
	}
	want := "/ip4/127.0.0.1/tcp/8080/p2p/12D3KooWtest"
	if got != want {
		t.Fatalf("unexpected multiaddr: %s", got)
	}
}

func TestBuildP2PMultiaddr_DNS(t *testing.T) {
	got, err := buildP2PMultiaddr(" localhost ", 8080, "12D3KooWtest")
	if err != nil {
		t.Fatalf("build p2p multiaddr: %v", err)
	}
	want := "/dns/localhost/tcp/8080/p2p/12D3KooWtest"
	if got != want {
		t.Fatalf("unexpected multiaddr: %s", got)
	}
}

func TestBuildP2PMultiaddr_RejectsIPv6(t *testing.T) {
	if _, err := buildP2PMultiaddr("::1", 8080, "12D3KooWtest"); err == nil {
		t.Fatalf("expected ipv6 base-host to be rejected")
	}
}

func TestBuildP2PMultiaddr_RejectsEmptyHost(t *testing.T) {
	if _, err := buildP2PMultiaddr("   ", 8080, "12D3KooWtest"); err == nil {
		t.Fatalf("expected empty base-host to be rejected")
	}
}
