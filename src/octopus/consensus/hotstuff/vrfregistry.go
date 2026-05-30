// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package hotstuff

import (
	"encoding/json"
	"fmt"

	"go.dedis.ch/kyber/v3"
	"go.dedis.ch/kyber/v3/suites"
)

// VRFRegistration 是 VRF 公钥注册消息。
// 节点启动时广播此消息，其他节点收到后注册发送者的 VRF 公钥。
// 这解决了 G6：之前每个节点只注册自身 VRF 公钥，导致 handleVote()
// 中 VRF 验证逻辑因缺少其他节点公钥而跳过验证。
type VRFRegistration struct {
	ValidatorID  uint64 `json:"validator_id"`
	VRFPubKeyRaw []byte `json:"vrf_pub_key"` // kyber.Point serialized bytes
}

// vrfSuite 用于 VRF 公钥的序列化/反序列化
var vrfSuite = suites.MustFind("Ed25519")

// EncodeVRFRegistration 编码 VRF 注册消息
func EncodeVRFRegistration(validatorID uint64, pubKey kyber.Point) ([]byte, error) {
	raw, err := pubKey.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("failed to marshal VRF public key: %w", err)
	}
	reg := &VRFRegistration{
		ValidatorID:  validatorID,
		VRFPubKeyRaw: raw,
	}
	return json.Marshal(reg)
}

// DecodeVRFRegistration 解码 VRF 注册消息，返回 validator ID 和 VRF 公钥
func DecodeVRFRegistration(data []byte) (uint64, kyber.Point, error) {
	var reg VRFRegistration
	if err := json.Unmarshal(data, &reg); err != nil {
		return 0, nil, fmt.Errorf("failed to unmarshal VRF registration: %w", err)
	}
	pubKey, err := DecodeVRFPublicKey(reg.VRFPubKeyRaw)
	if err != nil {
		return 0, nil, err
	}
	return reg.ValidatorID, pubKey, nil
}

func DecodeVRFPublicKey(raw []byte) (kyber.Point, error) {
	pubKey := vrfSuite.Point()
	if err := pubKey.UnmarshalBinary(raw); err != nil {
		return nil, fmt.Errorf("failed to unmarshal VRF public key point: %w", err)
	}
	return pubKey, nil
}

// BroadcastVRFPubKey 广播自身 VRF 公钥到共识 topic。
// 在 Engine.Start() 中调用。其他节点收到后通过 HandleVRFRegistration 注册。
func (e *Engine) BroadcastVRFPubKey() error {
	if e.vrfPubKey == nil {
		return nil // VRF not configured
	}

	data, err := EncodeVRFRegistration(e.nodeID, e.vrfPubKey)
	if err != nil {
		return fmt.Errorf("failed to encode VRF registration: %w", err)
	}

	// 使用共识 topic 广播（所有共识参与者都订阅了此 topic）
	if e.network != nil {
		if err := e.network.PublishTopic(e.consensusTopic, data); err != nil {
			return fmt.Errorf("failed to broadcast VRF pub key: %w", err)
		}
	}

	logger.Info("Broadcasted VRF public key for validator %d on topic %s", e.nodeID, e.consensusTopic)
	return nil
}

// HandleVRFRegistration 处理收到的 VRF 公钥注册消息。
// 验证发送者是当前验证者集成员后，存入 vrfPubKeys。
func (e *Engine) HandleVRFRegistration(data []byte) error {
	validatorID, pubKey, err := DecodeVRFRegistration(data)
	if err != nil {
		return err
	}

	// 验证发送者是活跃验证者
	if e.valSet != nil {
		if v, exists := e.valSet.Validators[validatorID]; !exists || v == nil || !v.IsActive {
			return fmt.Errorf("VRF registration from non-active validator %d rejected", validatorID)
		}
	}

	e.RegisterVRFPubKey(validatorID, pubKey)
	logger.Info("Registered VRF public key for validator %d", validatorID)
	return nil
}
