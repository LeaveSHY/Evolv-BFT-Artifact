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

Nodes == 1 .. N
\* Model the last F nodes as Byzantine
ByzNodes == ( N - F + 1 ) .. N
HonestNodes == 1 .. ( N - F )
QuorumSize == 2 * F + 1
ViewSet == 0 .. VIEWS
FastPathThreshold == ( N * 9 ) \div 10

VARIABLES view, lockedQC, commitQC, blocks, votes, decided, timeouts, tcs

vars == << view, lockedQC, commitQC, blocks, votes, decided, timeouts, tcs >>

Init ==
  /\ view = [n \in Nodes |-> 1]
  /\ lockedQC = [n \in Nodes |-> [ view |-> 0, node |-> 0 ]]
  /\ commitQC = [n \in Nodes |-> [ view |-> 0, node |-> 0 ]]
  /\ blocks = { [ view |-> 0, parent_view |-> -1, qc_view |-> -1 ] }
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

(* Helper: maximum lockedQC view across all honest nodes *)
MaxHonestQCView ==
  LET allQCViews == { lockedQC[n].view : n \in HonestNodes }
  IN CHOOSE mv \in allQCViews : \A ov \in allQCViews : mv >= ov

HonestPropose(leader, v) ==
  /\ leader \in HonestNodes
  /\ v = view[leader]    \* Leader can only propose at its current view
  /\ view[leader] \notin tcs  \* Must ViewChange first if TC exists for current view
  /\ Cardinality(blocks) < MAX_BLOCKS
  /\ LET highQCView == lockedQC[leader].view
     IN /\ \E parent \in blocks: parent.view = highQCView
        /\ LET newBlock == [ view |-> v,
                             parent_view |-> highQCView,
                             qc_view |-> highQCView ]
           IN blocks' = blocks \union { newBlock }
  /\ UNCHANGED << view, lockedQC, commitQC, votes, timeouts, decided, tcs >>

(* SyncQC: propagate the highest lockedQC to an honest node.
   Models QC piggybacking in the real HotStuff protocol. *)
SyncQC(node) ==
  /\ node \in HonestNodes
  /\ lockedQC[node].view < MaxHonestQCView
  /\ lockedQC' = [lockedQC EXCEPT ![node] =
       [view |-> MaxHonestQCView, node |-> node]]
  /\ UNCHANGED << view, commitQC, blocks, votes, decided, timeouts, tcs >>

HonestVote(node, v) ==
  /\ node \in HonestNodes
  /\ v = view[node]      \* Node can only vote at its current view
  /\ view[node] \notin tcs  \* Must ViewChange first if TC exists for current view
  /\ node \notin votes[v]
  \* Honest node votes at most once per view
  /\ \E b \in blocks: /\ b.view = v
                      /\ SafeNode(node, b)
                      /\ votes' = [votes EXCEPT ![v] = @ \union { node }]
  /\ UNCHANGED << view, lockedQC, commitQC, blocks, timeouts, decided, tcs >>

UpdateLock(node, qc) ==
  /\ node \in HonestNodes
  /\ qc \in 1 .. VIEWS
  /\ Cardinality(votes[qc]) >= QuorumSize
  /\ qc > lockedQC[node].view
  /\ view[node] \notin tcs  \* Must ViewChange first if TC exists for current view
  /\ lockedQC' = [lockedQC EXCEPT ![node] = [ view |-> qc, node |-> node ]]
  /\ UNCHANGED << view, commitQC, blocks, votes, timeouts, decided, tcs >>

Commit3Chain(node, v1, v2, v3) ==
  /\ node \in HonestNodes
  /\ v1 \in 1 .. VIEWS /\ v2 \in 1 .. VIEWS /\ v3 \in 1 .. VIEWS
  /\ v1 < v2 /\ v2 < v3
  /\ Cardinality(votes[v1]) >= QuorumSize
  /\ Cardinality(votes[v2]) >= QuorumSize
  /\ Cardinality(votes[v3]) >= QuorumSize
  /\ \E b3 \in blocks: b3.view = v3 /\ b3.parent_view = v2
  /\ \E b2 \in blocks: b2.view = v2 /\ b2.parent_view = v1
  /\ commitQC' = [commitQC EXCEPT ![node] = [ view |-> v1, node |-> node ]]
  /\ decided' = [decided EXCEPT ![node] = @ \union { v1 }]
  /\ lockedQC' =
       [lockedQC EXCEPT
       ![node] =
       IF @.view < v3 THEN [ view |-> v3, node |-> node ] ELSE @]
  /\ UNCHANGED << view, blocks, votes, timeouts, tcs >>

FastCommit2Chain(node, v1, v2) ==
  /\ node \in HonestNodes
  /\ v1 \in 1 .. VIEWS /\ v2 \in 1 .. VIEWS
  /\ v1 < v2
  /\ Cardinality(votes[v1]) >= QuorumSize
  /\ Cardinality(votes[v2]) >= FastPathThreshold
  /\ \E b2 \in blocks: b2.view = v2 /\ b2.parent_view = v1
  /\ commitQC' = [commitQC EXCEPT ![node] = [ view |-> v2, node |-> node ]]
  /\ decided' = [decided EXCEPT ![node] = @ \union { v2 }]
  /\ lockedQC' =
       [lockedQC EXCEPT
       ![node] =
       IF @.view < v2 THEN [ view |-> v2, node |-> node ] ELSE @]
  /\ UNCHANGED << view, blocks, votes, timeouts, tcs >>

Timeout(node, v) ==
  /\ v \in 1 .. VIEWS
  /\ v = view[node]  \* Node can only timeout at its current view
  /\ ~\E b \in blocks: b.view = v  \* Can only timeout if no proposal exists (realistic)
  /\ timeouts' = [timeouts EXCEPT ![v] = @ \union { node }]
  /\ UNCHANGED << view, lockedQC, commitQC, blocks, votes, decided, tcs >>

FormTC(v) ==
  /\ v \in 1 .. VIEWS
  /\ Cardinality(timeouts[v]) >= QuorumSize
  /\ tcs' = tcs \union { v }
  /\ UNCHANGED << view, lockedQC, commitQC, blocks, votes, timeouts, decided >>

(* ViewChange: after a TC forms for the current view, an honest node
   advances to the next view. This models the HotStuff view-change
   mechanism that ensures liveness under partial synchrony. *)
ViewChange(node) ==
  /\ node \in HonestNodes
  /\ view[node] < VIEWS
  /\ view[node] \in tcs   \* A TC exists for the current view
  /\ view' = [view EXCEPT ![node] = view[node] + 1]
  /\ UNCHANGED << lockedQC, commitQC, blocks, votes, decided, timeouts, tcs >>

(* SyncViewChange: models synchronized view change under partial synchrony.
   After GST, all honest nodes receive TC and advance view together.
   This prevents view drift and ensures liveness.
   Guard: ALL honest nodes must be at the same view with a TC. *)
SyncViewChange ==
  /\ \E v \in 1 .. VIEWS:
       /\ v \in tcs
       /\ \A n \in HonestNodes: view[n] = v
       /\ v < VIEWS
       /\ view' = [n \in Nodes |->
            IF n \in HonestNodes THEN v + 1 ELSE view[n]]
  /\ UNCHANGED << lockedQC, commitQC, blocks, votes, decided, timeouts, tcs >>

(* AdvanceAfterQC: after a quorum forms at the current view, honest nodes
   advance to the next view. Models normal protocol progression where
   the next leader takes over after a QC is formed.
   Also updates lockedQC to the quorum view (models QC propagation),
   but only if it increases the current lockedQC (monotonicity). *)
AdvanceAfterQC(node) ==
  /\ node \in HonestNodes
  /\ view[node] \in 1 .. VIEWS
  /\ Cardinality(votes[view[node]]) >= QuorumSize
  /\ view[node] < VIEWS
  /\ view' = [view EXCEPT ![node] = view[node] + 1]
  /\ lockedQC' = [lockedQC EXCEPT ![node] =
       IF view[node] > lockedQC[node].view
       THEN [view |-> view[node], node |-> node]
       ELSE lockedQC[node]]
  /\ UNCHANGED << commitQC, blocks, votes, decided, timeouts, tcs >>


(* ================================================================
   BYZANTINE ACTIONS — arbitrary, unconstrained behavior
   ================================================================ *)
(* Byzantine propose: create a block with arbitrary parent/QC links.
   This models equivocation and malicious chain construction. *)
ByzPropose(byz, v) ==
  /\ byz \in ByzNodes
  /\ v \in 1 .. VIEWS
  /\ Cardinality(blocks) < MAX_BLOCKS
  /\ \E parent_v \in ViewSet:
       \E qc_v \in ViewSet:
         LET newBlock ==
               [ view |-> v, parent_view |-> parent_v, qc_view |-> qc_v ]
         IN blocks' = blocks \union { newBlock }
  /\ UNCHANGED << view, lockedQC, commitQC, votes, timeouts, decided, tcs >>

(* Byzantine vote: vote for any view without SafeNode check.
   Models voting for conflicting blocks, withholding, and arbitrary behavior. *)
ByzVote(byz, v) ==
  /\ byz \in ByzNodes
  /\ v \in 1 .. VIEWS
  /\ votes' = [votes EXCEPT ![v] = @ \union { byz }]
  /\ UNCHANGED << view, lockedQC, commitQC, blocks, timeouts, decided, tcs >>


(* ================================================================
   SPECIFICATION
   ================================================================ *)
Next ==
  \* Honest protocol actions
  \/ \E leader \in HonestNodes, v \in 1 .. VIEWS: HonestPropose(leader, v)
  \/ \E node \in HonestNodes, v \in 1 .. VIEWS: HonestVote(node, v)
  \/ \E node \in HonestNodes, v \in 1 .. VIEWS: UpdateLock(node, v)
  \/ \E node \in HonestNodes,
       v1, v2, v3 \in 1 .. VIEWS:
       Commit3Chain(node, v1, v2, v3)
  \/ \E node \in HonestNodes,
       v1, v2 \in 1 .. VIEWS:
       FastCommit2Chain(node, v1, v2)
  \/ \E node \in Nodes, v \in 1 .. VIEWS: Timeout(node, v)
  \/ \E v \in 1 .. VIEWS: FormTC(v)
  \/ \E node \in HonestNodes: ViewChange(node)
  \/ SyncViewChange
  \/ \E node \in HonestNodes: AdvanceAfterQC(node)
  \/ \E node \in HonestNodes: SyncQC(node)
  \* Byzantine actions
  \/ \E byz \in ByzNodes, v \in 1 .. VIEWS: ByzPropose(byz, v)
  \/ \E byz \in ByzNodes, v \in 1 .. VIEWS: ByzVote(byz, v)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)


(* ================================================================
   SAFETY INVARIANTS — verified for HONEST nodes only
   ================================================================ *)
(* No two honest nodes commit conflicting decisions *)
HonestSafety ==
  \A n1, n2 \in HonestNodes:
    decided[n1] \intersect decided[n2] /= {} =>
      decided[n1] \subseteq decided[n2] \/ decided[n2] \subseteq decided[n1]

(* Lock monotonicity for honest nodes *)
HonestLockMonotonicity ==
  \A n \in HonestNodes: lockedQC[n].view >= commitQC[n].view

(* Commit prefix consistency among honest nodes *)
HonestCommitPrefix ==
  \A n1, n2 \in HonestNodes:
    \A v1 \in decided[n1],
      v2 \in decided[n2]:
      v1 <= v2 => v1 \in decided[n2] \/ v2 \in decided[n1]


(* ================================================================
   LIVENESS — verified for HONEST nodes under partial synchrony
   ================================================================ *)
(* Liveness: under partial synchrony with honest majority (N >= 3F+1),
   eventually some honest node commits a decision.
   Byzantine nodes cannot prevent progress indefinitely because
   honest nodes form a quorum (2F+1 out of 3F+1). *)
Liveness == <>( \E n \in HonestNodes: decided[n] /= {} )

=============================================================================
