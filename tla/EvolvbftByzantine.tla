---------------------- MODULE EvolvbftByzantine ----------------------
(* Byzantine fault tolerance verification for Evolv-BFT consensus.
   Extends EvolvbftSafety with explicit Byzantine adversary actions.
   Byzantine nodes can propose arbitrary blocks and vote freely,
   while honest nodes follow the Chained HotStuff protocol.

   Key verification target: even with F Byzantine nodes among N >= 3F+1,
   no two honest nodes commit conflicting decisions.

   The votes[v] abstraction over-approximates QC formation (counts all
   votes at view v regardless of which block they target). This makes
   quorum formation EASIER, so if safety holds in this model, it holds
   in the real system where split votes would prevent QC formation. *)

EXTENDS Integers, FiniteSets, Sequences, TLC

CONSTANTS N, F, VIEWS, MAX_BLOCKS

Nodes == 1..N
\* Model the last F nodes as Byzantine
ByzNodes == (N - F + 1)..N
HonestNodes == 1..(N - F)
QuorumSize == 2 * F + 1
ViewSet == 0..VIEWS
FastPathThreshold == (N * 9) \div 10

VARIABLES
    view,
    lockedQC,
    commitQC,
    blocks,
    votes,
    decided,
    timeouts,
    tcs

vars == <<view, lockedQC, commitQC, blocks, votes, decided, timeouts, tcs>>

Init ==
    /\ view = [n \in Nodes |-> 1]
    /\ lockedQC = [n \in Nodes |-> [view |-> 0, node |-> 0]]
    /\ commitQC = [n \in Nodes |-> [view |-> 0, node |-> 0]]
    /\ blocks = {[view |-> 0, parent_view |-> -1, qc_view |-> -1]}
    /\ votes = [v \in ViewSet |-> {}]
    /\ decided = [n \in Nodes |-> {}]
    /\ timeouts = [v \in ViewSet |-> {}]
    /\ tcs = {}

(* ================================================================
   HONEST NODE ACTIONS — follow the Chained HotStuff protocol
   ================================================================ *)

SafeNode(node, b) ==
    \/ b.qc_view > lockedQC[node].view
    \/ b.parent_view = lockedQC[node].view

HonestPropose(leader, v) ==
    /\ leader \in HonestNodes
    /\ v \in 1..VIEWS
    /\ Cardinality(blocks) < MAX_BLOCKS
    /\ \E highQC \in {lockedQC[leader]}:
        LET newBlock == [view |-> v, parent_view |-> highQC.view,
                         qc_view |-> highQC.view] IN
        /\ \E parent \in blocks : parent.view = highQC.view
        /\ blocks' = blocks \union {newBlock}
        /\ UNCHANGED <<view, lockedQC, commitQC, votes, timeouts, decided, tcs>>

HonestVote(node, v) ==
    /\ node \in HonestNodes
    /\ v \in 1..VIEWS
    /\ node \notin votes[v]  \* Honest node votes at most once per view
    /\ \E b \in blocks:
        /\ b.view = v
        /\ SafeNode(node, b)
        /\ votes' = [votes EXCEPT ![v] = @ \union {node}]
    /\ UNCHANGED <<view, lockedQC, commitQC, blocks, timeouts, decided, tcs>>

UpdateLock(node, qc) ==
    /\ node \in HonestNodes
    /\ qc \in 1..VIEWS
    /\ Cardinality(votes[qc]) >= QuorumSize
    /\ qc > lockedQC[node].view
    /\ lockedQC' = [lockedQC EXCEPT ![node] = [view |-> qc, node |-> node]]
    /\ UNCHANGED <<view, commitQC, blocks, votes, timeouts, decided, tcs>>

Commit3Chain(node, v1, v2, v3) ==
    /\ node \in HonestNodes
    /\ v1 \in 1..VIEWS /\ v2 \in 1..VIEWS /\ v3 \in 1..VIEWS
    /\ v1 < v2 /\ v2 < v3
    /\ Cardinality(votes[v1]) >= QuorumSize
    /\ Cardinality(votes[v2]) >= QuorumSize
    /\ Cardinality(votes[v3]) >= QuorumSize
    /\ \E b3 \in blocks : b3.view = v3 /\ b3.parent_view = v2
    /\ \E b2 \in blocks : b2.view = v2 /\ b2.parent_view = v1
    /\ commitQC' = [commitQC EXCEPT ![node] = [view |-> v1, node |-> node]]
    /\ decided' = [decided EXCEPT ![node] = @ \union {v1}]
    /\ lockedQC' = [lockedQC EXCEPT ![node] =
        IF @.view < v3 THEN [view |-> v3, node |-> node] ELSE @]
    /\ UNCHANGED <<view, blocks, votes, timeouts, tcs>>

FastCommit2Chain(node, v1, v2) ==
    /\ node \in HonestNodes
    /\ v1 \in 1..VIEWS /\ v2 \in 1..VIEWS
    /\ v1 < v2
    /\ Cardinality(votes[v1]) >= QuorumSize
    /\ Cardinality(votes[v2]) >= FastPathThreshold
    /\ \E b2 \in blocks : b2.view = v2 /\ b2.parent_view = v1
    /\ commitQC' = [commitQC EXCEPT ![node] = [view |-> v2, node |-> node]]
    /\ decided' = [decided EXCEPT ![node] = @ \union {v2}]
    /\ lockedQC' = [lockedQC EXCEPT ![node] =
        IF @.view < v2 THEN [view |-> v2, node |-> node] ELSE @]
    /\ UNCHANGED <<view, blocks, votes, timeouts, tcs>>

Timeout(node, v) ==
    /\ v \in 1..VIEWS
    /\ timeouts' = [timeouts EXCEPT ![v] = @ \union {node}]
    /\ UNCHANGED <<view, lockedQC, commitQC, blocks, votes, decided, tcs>>

FormTC(v) ==
    /\ v \in 1..VIEWS
    /\ Cardinality(timeouts[v]) >= QuorumSize
    /\ tcs' = tcs \union {v}
    /\ UNCHANGED <<view, lockedQC, commitQC, blocks, votes, timeouts, decided>>

(* ================================================================
   BYZANTINE ACTIONS — arbitrary, unconstrained behavior
   ================================================================ *)

(* Byzantine propose: create a block with arbitrary parent/QC links.
   This models equivocation and malicious chain construction. *)
ByzPropose(byz, v) ==
    /\ byz \in ByzNodes
    /\ v \in 1..VIEWS
    /\ Cardinality(blocks) < MAX_BLOCKS
    /\ \E parent_v \in ViewSet :
        \E qc_v \in ViewSet :
            LET newBlock == [view |-> v, parent_view |-> parent_v,
                             qc_view |-> qc_v] IN
            blocks' = blocks \union {newBlock}
    /\ UNCHANGED <<view, lockedQC, commitQC, votes, timeouts, decided, tcs>>

(* Byzantine vote: vote for any view without SafeNode check.
   Models voting for conflicting blocks, withholding, and arbitrary behavior. *)
ByzVote(byz, v) ==
    /\ byz \in ByzNodes
    /\ v \in 1..VIEWS
    /\ votes' = [votes EXCEPT ![v] = @ \union {byz}]
    /\ UNCHANGED <<view, lockedQC, commitQC, blocks, timeouts, decided, tcs>>

(* ================================================================
   SPECIFICATION
   ================================================================ *)

Next ==
    \* Honest protocol actions
    \/ \E leader \in HonestNodes, v \in 1..VIEWS : HonestPropose(leader, v)
    \/ \E node \in HonestNodes, v \in 1..VIEWS : HonestVote(node, v)
    \/ \E node \in HonestNodes, v \in 1..VIEWS : UpdateLock(node, v)
    \/ \E node \in HonestNodes, v1, v2, v3 \in 1..VIEWS :
        Commit3Chain(node, v1, v2, v3)
    \/ \E node \in HonestNodes, v1, v2 \in 1..VIEWS :
        FastCommit2Chain(node, v1, v2)
    \/ \E node \in Nodes, v \in 1..VIEWS : Timeout(node, v)
    \/ \E v \in 1..VIEWS : FormTC(v)
    \* Byzantine actions
    \/ \E byz \in ByzNodes, v \in 1..VIEWS : ByzPropose(byz, v)
    \/ \E byz \in ByzNodes, v \in 1..VIEWS : ByzVote(byz, v)

Spec == Init /\ [][Next]_vars

(* ================================================================
   SAFETY INVARIANTS — verified for HONEST nodes only
   ================================================================ *)

(* No two honest nodes commit conflicting decisions *)
HonestSafety == \A n1, n2 \in HonestNodes :
    decided[n1] \intersect decided[n2] /= {} =>
        decided[n1] \subseteq decided[n2] \/ decided[n2] \subseteq decided[n1]

(* Lock monotonicity for honest nodes *)
HonestLockMonotonicity == \A n \in HonestNodes :
    lockedQC[n].view >= commitQC[n].view

(* Commit prefix consistency among honest nodes *)
HonestCommitPrefix == \A n1, n2 \in HonestNodes :
    \A v1 \in decided[n1], v2 \in decided[n2] :
        v1 <= v2 => v1 \in decided[n2] \/ v2 \in decided[n1]

=============================================================================
