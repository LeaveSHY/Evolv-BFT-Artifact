// Copyright 2024 Octopus Project
// Licensed under Apache License 2.0

package proto

import (
	"context"
	"google.golang.org/grpc"
)

// BlockMsg wraps a block for transmission
type BlockMsg struct {
	BlockData []byte `protobuf:"bytes,1,opt,name=block_data,json=blockData,proto3" json:"block_data,omitempty"`
}

// VoteMsg wraps a vote for transmission
type VoteMsg struct {
	VoteData []byte `protobuf:"bytes,1,opt,name=vote_data,json=voteData,proto3" json:"vote_data,omitempty"`
}

// NewViewMsg wraps a new view message
type NewViewMsg struct {
	View     uint64 `protobuf:"varint,1,opt,name=view,proto3" json:"view,omitempty"`
	SenderId uint64 `protobuf:"varint,2,opt,name=sender_id,json=senderId,proto3" json:"sender_id,omitempty"`
	HighQc   []byte `protobuf:"bytes,3,opt,name=high_qc,json=highQc,proto3" json:"high_qc,omitempty"`
}

// Ack is a generic acknowledgement
type Ack struct {
	Success bool   `protobuf:"varint,1,opt,name=success,proto3" json:"success,omitempty"`
	Error   string `protobuf:"bytes,2,opt,name=error,proto3" json:"error,omitempty"`
}

// ConsensusServiceServer is the server API for ConsensusService service.
type ConsensusServiceServer interface {
	// ProposeBlock sends a new block proposal
	ProposeBlock(context.Context, *BlockMsg) (*Ack, error)
	// SubmitVote sends a vote for a block
	SubmitVote(context.Context, *VoteMsg) (*Ack, error)
	// NewView notifies about view change
	NewView(context.Context, *NewViewMsg) (*Ack, error)
	mustEmbedUnimplementedConsensusServiceServer()
}

// UnimplementedConsensusServiceServer must be embedded to have forward compatible implementations.
type UnimplementedConsensusServiceServer struct {
}

func (UnimplementedConsensusServiceServer) ProposeBlock(context.Context, *BlockMsg) (*Ack, error) {
	return nil, nil
}
func (UnimplementedConsensusServiceServer) SubmitVote(context.Context, *VoteMsg) (*Ack, error) {
	return nil, nil
}
func (UnimplementedConsensusServiceServer) NewView(context.Context, *NewViewMsg) (*Ack, error) {
	return nil, nil
}
func (UnimplementedConsensusServiceServer) mustEmbedUnimplementedConsensusServiceServer() {}

// UnsafeConsensusServiceServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to ConsensusServiceServer will
// result in compilation errors.
type UnsafeConsensusServiceServer interface {
	mustEmbedUnimplementedConsensusServiceServer()
}

func RegisterConsensusServiceServer(s grpc.ServiceRegistrar, srv ConsensusServiceServer) {
	s.RegisterService(&ConsensusService_ServiceDesc, srv)
}

// ConsensusServiceClient is the client API for ConsensusService service.
type ConsensusServiceClient interface {
	// ProposeBlock sends a new block proposal
	ProposeBlock(ctx context.Context, in *BlockMsg, opts ...grpc.CallOption) (*Ack, error)
	// SubmitVote sends a vote for a block
	SubmitVote(ctx context.Context, in *VoteMsg, opts ...grpc.CallOption) (*Ack, error)
	// NewView notifies about view change
	NewView(ctx context.Context, in *NewViewMsg, opts ...grpc.CallOption) (*Ack, error)
}

type consensusServiceClient struct {
	cc grpc.ClientConnInterface
}

func NewConsensusServiceClient(cc grpc.ClientConnInterface) ConsensusServiceClient {
	return &consensusServiceClient{cc}
}

func (c *consensusServiceClient) ProposeBlock(ctx context.Context, in *BlockMsg, opts ...grpc.CallOption) (*Ack, error) {
	out := new(Ack)
	err := c.cc.Invoke(ctx, "/network.ConsensusService/ProposeBlock", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *consensusServiceClient) SubmitVote(ctx context.Context, in *VoteMsg, opts ...grpc.CallOption) (*Ack, error) {
	out := new(Ack)
	err := c.cc.Invoke(ctx, "/network.ConsensusService/SubmitVote", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *consensusServiceClient) NewView(ctx context.Context, in *NewViewMsg, opts ...grpc.CallOption) (*Ack, error) {
	out := new(Ack)
	err := c.cc.Invoke(ctx, "/network.ConsensusService/NewView", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ConsensusService_ServiceDesc is the grpc.ServiceDesc for ConsensusService service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var ConsensusService_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "network.ConsensusService",
	HandlerType: (*ConsensusServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "ProposeBlock",
			Handler:    _ConsensusService_ProposeBlock_Handler,
		},
		{
			MethodName: "SubmitVote",
			Handler:    _ConsensusService_SubmitVote_Handler,
		},
		{
			MethodName: "NewView",
			Handler:    _ConsensusService_NewView_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "octopus/network/proto/consensus.proto",
}

func _ConsensusService_ProposeBlock_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(BlockMsg)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ConsensusServiceServer).ProposeBlock(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/network.ConsensusService/ProposeBlock",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ConsensusServiceServer).ProposeBlock(ctx, req.(*BlockMsg))
	}
	return interceptor(ctx, in, info, handler)
}

func _ConsensusService_SubmitVote_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(VoteMsg)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ConsensusServiceServer).SubmitVote(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/network.ConsensusService/SubmitVote",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ConsensusServiceServer).SubmitVote(ctx, req.(*VoteMsg))
	}
	return interceptor(ctx, in, info, handler)
}

func _ConsensusService_NewView_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NewViewMsg)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ConsensusServiceServer).NewView(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/network.ConsensusService/NewView",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ConsensusServiceServer).NewView(ctx, req.(*NewViewMsg))
	}
	return interceptor(ctx, in, info, handler)
}
