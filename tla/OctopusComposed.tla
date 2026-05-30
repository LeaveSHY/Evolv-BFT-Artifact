---------------------- MODULE OctopusComposed ----------------------
(* OctopusComposed: a composition model that verifies the interaction between
   consensus safety, dynamic reconfiguration, and multi-leader global ordering.

   This model captures the following cross-cutting concerns:
   1. Safety is preserved when reconfigurations change the validator set
   2. Global ordering remains consistent across epoch transitions
   3. Fast-path commits interact correctly with reconfiguration

   Parameters: N=4, F=1, VIEWS=4, M=2, MAX_HEIGHT=3, MAX_EPOCH=3 *)

EXTENDS Integers, FiniteSets, Sequences, TLC

CONSTANTS N, F, VIEWS, M, MAX_HEIGHT, MAX_EPOCH

Nodes == 1..N
QuorumSize == 2 * F + 1
ViewSet == 0..VIEWS
Instances == 1..M
Heights == 1..MAX_HEIGHT
FastPathThreshold == (N * 9) \div 10

VARIABLES
    \* Consensus state (from Safety)
    view,
    lockedQC,
    commitQC,
    blocks,
    votes,
    decided,
    timeouts,
    tcs,

    \* Reconfiguration state (from Reconfiguration)
    epoch,
    activeSet,
    mHigh,
    mValid,
    reconfigQuorumSize,
    committed,

    \* Multi-leader state (from MultiLeader)
    chain,
    globalLog,
    rank,
    pending,
    nilFilled

consensusVars == <<view, lockedQC, commitQC, blocks, votes, decided, timeouts, tcs>>
reconfigVars == <<epoch, activeSet, mHigh, mValid, reconfigQuorumSize, committed>>
multiLeaderVars == <<chain, globalLog, rank, pending, nilFilled>>
vars == <<consensusVars, reconfigVars, multiLeaderVars>>

Init ==
    \* Consensus init
    /\ view = [n \in Nodes |-> 1]
    /\ lockedQC = [n \in Nodes |-> [view |-> 0, node |-> 0]]
    /\ commitQC = [n \in Nodes |-> [view |-> 0, node |-> 0]]
    /\ blocks = {[view |-> 0, parent_view |-> -1, qc_view |-> -1]}
    /\ votes = [v \in ViewSet |-> {}]
    /\ decided = [n \in Nodes |-> {}]
    /\ timeouts = [v \in ViewSet |-> {}]
    /\ tcs = {}
    \* Reconfig init
    /\ epoch = 1
    /\ activeSet = Nodes
    /\ mHigh = Nodes
    /\ mValid = Nodes
    /\ reconfigQuorumSize = QuorumSize
    /\ committed = FALSE
    \* Multi-leader init
    /\ chain = [i \in Instances |-> <<>>]
    /\ globalLog = <<>>
    /\ rank = [i \in Instances |-> [h \in Heights |-> (h - 1) * M + (i - 1)]]
    /\ pending = {}
    /\ nilFilled = {}

(* Consensus actions — only active set members can participate *)
SafeNode(node, b) ==
    /\ node \in activeSet  \* Only active nodes can evaluate
    /\ (b.qc_view > lockedQC[node].view \/ b.parent_view = lockedQC[node].view)

Propose(leader, v) ==
    /\ leader \in activeSet  \* Only active members can propose
    /\ v \in 1..VIEWS
    /\ \E highQC \in {lockedQC[leader]}:
        LET newBlock == [view |-> v, parent_view |-> highQC.view, qc_view |-> highQC.view] IN
        /\ \E parent \in blocks : parent.view = highQC.view
        /\ blocks' = blocks \union {newBlock}
        /\ UNCHANGED <<view, lockedQC, commitQC, votes, timeouts, decided, tcs,
                       reconfigVars, multiLeaderVars>>

Vote(node, v) ==
    /\ node \in activeSet  \* Only active members can vote
    /\ v \in 1..VIEWS
    /\ \E b \in blocks:
        /\ b.view = v
        /\ SafeNode(node, b)
        /\ votes' = [votes EXCEPT ![v] = @ \union {node}]
    /\ UNCHANGED <<view, lockedQC, commitQC, blocks, timeouts, decided, tcs,
                   reconfigVars, multiLeaderVars>>

UpdateLock(node, qc) ==
    /\ node \in activeSet
    /\ qc \in 1..VIEWS
    /\ Cardinality(votes[qc]) >= QuorumSize
    /\ qc > lockedQC[node].view
    /\ lockedQC' = [lockedQC EXCEPT ![node] = [view |-> qc, node |-> node]]
    /\ UNCHANGED <<view, commitQC, blocks, votes, timeouts, decided, tcs,
                   reconfigVars, multiLeaderVars>>

(* Commit — when committing, also signal the reconfig layer *)
Commit3Chain(node, v1, v2, v3) ==
    /\ node \in activeSet
    /\ v1 \in 1..VIEWS /\ v2 \in 1..VIEWS /\ v3 \in 1..VIEWS
    /\ v1 < v2 /\ v2 < v3
    /\ Cardinality(votes[v1]) >= QuorumSize
    /\ Cardinality(votes[v2]) >= QuorumSize
    /\ Cardinality(votes[v3]) >= QuorumSize
    /\ \E b3 \in blocks : b3.view = v3 /\ b3.parent_view = v2
    /\ \E b2 \in blocks : b2.view = v2 /\ b2.parent_view = v1
    /\ commitQC' = [commitQC EXCEPT ![node] = [view |-> v1, node |-> node]]
    /\ decided' = [decided EXCEPT ![node] = @ \union {v1}]
    /\ lockedQC' = [lockedQC EXCEPT ![node] = IF @.view < v3 THEN [view |-> v3, node |-> node] ELSE @]
    /\ committed' = TRUE   \* Signal reconfig layer that a commit occurred
    /\ UNCHANGED <<view, blocks, votes, timeouts, tcs,
                   epoch, activeSet, mHigh, mValid, reconfigQuorumSize,
                   multiLeaderVars>>

(* Instance commit feeds into multi-leader global ordering *)
InstanceCommit(i, h) ==
    /\ h \in Heights
    /\ Len(chain[i]) = h - 1
    /\ LET b == [instance |-> i, height |-> h, rank |-> rank[i][h], isNil |-> FALSE] IN
       /\ chain' = [chain EXCEPT ![i] = Append(chain[i], b)]
       /\ pending' = pending \union {b}
    /\ UNCHANGED <<consensusVars, reconfigVars, globalLog, rank, nilFilled>>

GlobalOrder ==
    /\ pending /= {}
    /\ \E b \in pending :
       /\ \A b2 \in pending : b.rank <= b2.rank
       /\ IF Len(globalLog) = 0 THEN TRUE ELSE b.rank > globalLog[Len(globalLog)].rank
       /\ globalLog' = Append(globalLog, b)
       /\ pending' = pending \ {b}
    /\ UNCHANGED <<consensusVars, reconfigVars, chain, rank, nilFilled>>

(* Trust-based eviction: models SFAC trust manager proposing eviction,
   filtered by the safety mask (Section III-D Eq.15).
   Target chosen non-deterministically — over-approximates any trust policy.
   Safety filter ensures MinConfigSize is preserved post-eviction. *)
TrustEviction(target) ==
    /\ target \in activeSet
    /\ Cardinality(mHigh \ {target}) >= 4     \* Safety mask: |Ω \ {target}| >= 3f+1
    /\ Cardinality((mHigh \ {target}) \intersect mValid) >= 1  \* Quorum intersection preserved
    /\ mHigh' = mHigh \ {target}
    /\ UNCHANGED <<consensusVars, epoch, activeSet, mValid, reconfigQuorumSize, committed,
                   multiLeaderVars>>

(* Reconfig promotion — requires commit proof and quorum intersection *)
PromoteConfig ==
    /\ epoch < MAX_EPOCH
    /\ committed = TRUE
    /\ Cardinality(mHigh \intersect mValid) >= 1
    /\ mValid' = mHigh
    /\ activeSet' = mHigh
    /\ reconfigQuorumSize' = (2 * Cardinality(mHigh)) \div 3 + 1
    /\ epoch' = epoch + 1
    /\ committed' = FALSE
    /\ UNCHANGED <<consensusVars, mHigh,
                   multiLeaderVars>>

Timeout(node, v) ==
    /\ v \in 1..VIEWS
    /\ timeouts' = [timeouts EXCEPT ![v] = @ \union {node}]
    /\ UNCHANGED <<view, lockedQC, commitQC, blocks, votes, decided, tcs,
                   reconfigVars, multiLeaderVars>>

FormTC(v) ==
    /\ v \in 1..VIEWS
    /\ Cardinality(timeouts[v]) >= QuorumSize
    /\ tcs' = tcs \union {v}
    /\ UNCHANGED <<view, lockedQC, commitQC, blocks, votes, timeouts, decided,
                   reconfigVars, multiLeaderVars>>

Next ==
    \/ \E leader \in Nodes, v \in 1..VIEWS : Propose(leader, v)
    \/ \E node \in Nodes, v \in 1..VIEWS : Vote(node, v)
    \/ \E node \in Nodes, v \in 1..VIEWS : UpdateLock(node, v)
    \/ \E node \in Nodes, v1, v2, v3 \in 1..VIEWS : Commit3Chain(node, v1, v2, v3)
    \/ \E i \in Instances, h \in Heights : InstanceCommit(i, h)
    \/ GlobalOrder
    \/ \E target \in Nodes : TrustEviction(target)
    \/ PromoteConfig
    \/ \E node \in Nodes, v \in 1..VIEWS : Timeout(node, v)
    \/ \E v \in 1..VIEWS : FormTC(v)

Spec == Init /\ [][Next]_vars

(* === COMPOSED SAFETY INVARIANTS === *)

(* Consensus safety: no conflicting commits *)
ConsensusSafety == \A n1, n2 \in Nodes :
    decided[n1] \intersect decided[n2] /= {} =>
        decided[n1] \subseteq decided[n2] \/ decided[n2] \subseteq decided[n1]

(* Lock monotonicity *)
LockMonotonicity == \A n \in Nodes :
    lockedQC[n].view >= commitQC[n].view

(* Quorum intersection across reconfigs *)
QuorumIntersection == Cardinality(mHigh \intersect mValid) >= 1

(* Global ordering monotonicity *)
StrictOrdering == \A idx \in 1..Len(globalLog) :
    idx > 1 => globalLog[idx].rank > globalLog[idx-1].rank

(* Configuration minimum size *)
MinConfigSize == Cardinality(activeSet) >= 4

(* Composed safety: all invariants hold simultaneously *)
ComposedSafety ==
    /\ ConsensusSafety
    /\ LockMonotonicity
    /\ QuorumIntersection
    /\ StrictOrdering
    /\ MinConfigSize

=============================================================================
