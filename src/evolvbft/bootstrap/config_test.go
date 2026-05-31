package bootstrap

import "testing"

func TestParseEngineConfig_DefaultsForTask4Tuning(t *testing.T) {
	cfg, err := ParseEngineConfig([]string{})
	if err != nil {
		t.Fatalf("parse defaults failed: %v", err)
	}
	if cfg.Instances != 10 {
		t.Fatalf("unexpected default instances: %d", cfg.Instances)
	}
	if cfg.BatchTxs != 4096 {
		t.Fatalf("unexpected default batch-txs: %d", cfg.BatchTxs)
	}
	if cfg.TimeoutMs != 500 {
		t.Fatalf("unexpected default timeout-ms: %d", cfg.TimeoutMs)
	}
	if cfg.InboundMsgQueue != 8192 {
		t.Fatalf("unexpected default inbound-msg-queue: %d", cfg.InboundMsgQueue)
	}
	if cfg.InboundTxQueue != 65536 {
		t.Fatalf("unexpected default inbound-tx-queue: %d", cfg.InboundTxQueue)
	}
	if cfg.OrdererPendingCap != 65536 {
		t.Fatalf("unexpected default orderer-pending-cap: %d", cfg.OrdererPendingCap)
	}
	if cfg.HTTPListenAddr != "127.0.0.1" {
		t.Fatalf("unexpected default http-listen-addr: %q", cfg.HTTPListenAddr)
	}
	if cfg.AdminPprofEnabled {
		t.Fatalf("admin pprof should be disabled by default")
	}
}

func TestParseEngineConfig_TuningOverrides(t *testing.T) {
	cfg, err := ParseEngineConfig([]string{
		"-manifest", "C:/tmp/genesis.json",
		"-instances", "8",
		"-batch-txs", "1024",
		"-timeout-ms", "1500",
		"-inbound-msg-queue", "2048",
		"-inbound-tx-queue", "16384",
		"-orderer-pending-cap", "4096",
		"-http-listen-addr", "0.0.0.0",
	})
	if err != nil {
		t.Fatalf("parse overrides failed: %v", err)
	}
	if cfg.Instances != 8 || cfg.BatchTxs != 1024 || cfg.TimeoutMs != 1500 {
		t.Fatalf("unexpected parsed core tuning: instances=%d batch=%d timeout=%d", cfg.Instances, cfg.BatchTxs, cfg.TimeoutMs)
	}
	if cfg.InboundMsgQueue != 2048 || cfg.InboundTxQueue != 16384 || cfg.OrdererPendingCap != 4096 {
		t.Fatalf("unexpected parsed queue tuning: msg=%d tx=%d orderer=%d", cfg.InboundMsgQueue, cfg.InboundTxQueue, cfg.OrdererPendingCap)
	}
	if cfg.HTTPListenAddr != "0.0.0.0" {
		t.Fatalf("unexpected http listen addr: %q", cfg.HTTPListenAddr)
	}
	if cfg.Manifest != "C:/tmp/genesis.json" {
		t.Fatalf("unexpected manifest path: %q", cfg.Manifest)
	}
}

func TestEngineConfigRequiresManifestForMultiNode(t *testing.T) {
	cfg, err := ParseEngineConfig([]string{})
	if err != nil {
		t.Fatalf("parse defaults failed: %v", err)
	}
	if !cfg.RequiresManifest() {
		t.Fatalf("default multi-node config should require a manifest")
	}

	singleNode, err := ParseEngineConfig([]string{
		"-total-nodes", "1",
		"-initial-validators", "1",
	})
	if err != nil {
		t.Fatalf("parse single-node config failed: %v", err)
	}
	if singleNode.RequiresManifest() {
		t.Fatalf("single-node config should not require a manifest")
	}
}

func TestParseEngineConfig_AdaptiveOverrides(t *testing.T) {
	cfg, err := ParseEngineConfig([]string{
		"-manifest", "C:/tmp/genesis.json",
		"-adaptive-enabled",
		"-adaptive-policy", "http",
		"-adaptive-interval-ms", "2500",
		"-adaptive-script", "C:/tmp/policy.json",
		"-adaptive-policy-url", "http://127.0.0.1:18080/infer",
		"-adaptive-trace-path", "C:/tmp/trace.jsonl",
		"-admin-pprof-enabled",
	})
	if err != nil {
		t.Fatalf("parse adaptive overrides failed: %v", err)
	}
	if !cfg.AdaptiveEnabled {
		t.Fatalf("adaptive-enabled not parsed")
	}
	if cfg.AdaptivePolicy != "http" {
		t.Fatalf("unexpected adaptive policy: %q", cfg.AdaptivePolicy)
	}
	if cfg.AdaptiveIntervalMs != 2500 {
		t.Fatalf("unexpected adaptive interval: %d", cfg.AdaptiveIntervalMs)
	}
	if cfg.AdaptiveScript != "C:/tmp/policy.json" {
		t.Fatalf("unexpected adaptive script: %q", cfg.AdaptiveScript)
	}
	if cfg.AdaptivePolicyURL != "http://127.0.0.1:18080/infer" {
		t.Fatalf("unexpected adaptive policy url: %q", cfg.AdaptivePolicyURL)
	}
	if cfg.AdaptiveTracePath != "C:/tmp/trace.jsonl" {
		t.Fatalf("unexpected adaptive trace path: %q", cfg.AdaptiveTracePath)
	}
	if !cfg.AdminPprofEnabled {
		t.Fatalf("admin-pprof-enabled not parsed")
	}
}

func TestParseEngineConfig_RejectsUnknownAdaptivePolicy(t *testing.T) {
	_, err := ParseEngineConfig([]string{
		"-adaptive-enabled",
		"-adaptive-policy", "bogus",
	})
	if err == nil {
		t.Fatalf("expected unknown adaptive policy to fail")
	}
}

func TestParseEngineConfig_RequiresAdaptiveCompanionInputs(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{
			name: "scripted requires script path",
			args: []string{"-adaptive-enabled", "-adaptive-policy", "scripted"},
		},
		{
			name: "http requires policy url",
			args: []string{"-adaptive-enabled", "-adaptive-policy", "http"},
		},
		{
			name: "facmac-http requires policy url",
			args: []string{"-adaptive-enabled", "-adaptive-policy", "facmac-http"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseEngineConfig(tc.args)
			if err == nil {
				t.Fatalf("expected adaptive companion input validation failure")
			}
		})
	}
}

func TestParseEngineConfig_AcceptsFacmacHTTPWithPolicyURL(t *testing.T) {
	cfg, err := ParseEngineConfig([]string{
		"-adaptive-enabled",
		"-adaptive-policy", "facmac-http",
		"-adaptive-policy-url", "http://127.0.0.1:18080/infer",
	})
	if err != nil {
		t.Fatalf("expected facmac-http config to parse: %v", err)
	}
	if cfg.AdaptivePolicy != "facmac-http" {
		t.Fatalf("unexpected adaptive policy: %q", cfg.AdaptivePolicy)
	}
}

func TestParseEngineConfig_NormalizesAdaptivePolicyWhitespace(t *testing.T) {
	cfg, err := ParseEngineConfig([]string{
		"-adaptive-enabled",
		"-adaptive-policy", " facmac-http ",
		"-adaptive-policy-url", "http://127.0.0.1:18080/infer",
	})
	if err != nil {
		t.Fatalf("expected whitespace-normalized adaptive policy to parse: %v", err)
	}
	if cfg.AdaptivePolicy != "facmac-http" {
		t.Fatalf("expected adaptive policy to be normalized, got %q", cfg.AdaptivePolicy)
	}
}
