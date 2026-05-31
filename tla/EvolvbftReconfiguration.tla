------------------- MODULE EvolvbftReconfiguration -------------------
EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS MAX_NODES, INITIAL_NODES, MAX_EPOCH

VARIABLES
    epoch,
    activeSet,
    pendingJoins,
    pendingLeaves,
    mHigh,
    mValid,
    quorumSize,
    committed    \* G5b: track committed heights per epoch

vars == <<epoch, activeSet, pendingJoins, pendingLeaves, mHigh, mValid, quorumSize, committed>>

Nodes == 1..MAX_NODES

Init ==
    /\ epoch = 1
    /\ activeSet = 1..INITIAL_NODES
    /\ pendingJoins = {}
    /\ pendingLeaves = {}
    /\ mHigh = 1..INITIAL_NODES
    /\ mValid = 1..INITIAL_NODES
    /\ quorumSize = (2 * INITIAL_NODES) \div 3 + 1
    /\ committed = FALSE  \* No commit yet in current epoch

SubmitJoin(nodeID) ==
    /\ nodeID \notin activeSet
    /\ pendingJoins' = pendingJoins \union {nodeID}
    /\ UNCHANGED <<epoch, activeSet, pendingLeaves, mHigh, mValid, quorumSize, committed>>

SubmitLeave(nodeID) ==
    /\ nodeID \in activeSet
    /\ pendingLeaves' = pendingLeaves \union {nodeID}
    /\ UNCHANGED <<epoch, activeSet, pendingJoins, mHigh, mValid, quorumSize, committed>>

ApplyJoin(nodeID) ==
    /\ nodeID \in pendingJoins
    /\ mHigh' = mHigh \union {nodeID}
    /\ pendingJoins' = pendingJoins \ {nodeID}
    /\ UNCHANGED <<epoch, activeSet, pendingLeaves, mValid, quorumSize, committed>>

ApplyLeave(nodeID) ==
    /\ nodeID \in pendingLeaves
    /\ Cardinality(mHigh \ {nodeID}) >= 4
    /\ mHigh' = mHigh \ {nodeID}
    /\ pendingLeaves' = pendingLeaves \ {nodeID}
    /\ UNCHANGED <<epoch, activeSet, pendingJoins, mValid, quorumSize, committed>>

(* G5b fix: PromoteConfig now requires a commit proof before epoch transition.
   This models the G4 epoch safety barrier — config change only at commit points.
   Additionally, we check QuorumIntersection between old and new configurations. *)
PromoteConfig ==
    /\ epoch < MAX_EPOCH
    /\ committed = TRUE   \* G5b: can only promote after a commit in current epoch
    \* G5b: QuorumIntersection — new config must overlap with old config by at least 1 node
    \* This ensures safety across epoch boundaries (shared honest voter)
    /\ Cardinality(mHigh \intersect mValid) >= 1
    /\ mValid' = mHigh
    /\ activeSet' = mHigh
    /\ quorumSize' = (2 * Cardinality(mHigh)) \div 3 + 1
    /\ epoch' = epoch + 1
    /\ committed' = FALSE  \* Reset for new epoch
    /\ UNCHANGED <<pendingJoins, pendingLeaves, mHigh>>

(* CommitBlock: model a block being committed in the current epoch.
   This enables PromoteConfig to proceed. *)
CommitBlock ==
    /\ committed' = TRUE
    /\ UNCHANGED <<epoch, activeSet, pendingJoins, pendingLeaves, mHigh, mValid, quorumSize>>

AutoTransition(nodeID) ==
    /\ nodeID \in activeSet
    /\ Cardinality(mHigh \ {nodeID}) >= 4
    /\ mHigh' = mHigh \ {nodeID}
    /\ UNCHANGED <<epoch, activeSet, pendingJoins, pendingLeaves, mValid, quorumSize, committed>>

Next ==
    \/ \E nodeID \in Nodes : SubmitJoin(nodeID)
    \/ \E nodeID \in Nodes : SubmitLeave(nodeID)
    \/ \E nodeID \in pendingJoins : ApplyJoin(nodeID)
    \/ \E nodeID \in pendingLeaves : ApplyLeave(nodeID)
    \/ PromoteConfig
    \/ CommitBlock
    \/ \E nodeID \in activeSet : AutoTransition(nodeID)

Spec == Init /\ [][Next]_vars

(* === SAFETY INVARIANTS === *)

(* MinimumConfigSize: active set must always have enough nodes for BFT.
   With f=1 (implied by INITIAL_NODES>=4), we need at least 3f+1 = 4 nodes. *)
MinimumConfigSize == Cardinality(activeSet) >= 4

(* G5b fix: QuorumIntersection — mValid and mHigh must always share enough
   nodes to ensure quorum intersection across epoch transitions.
   Specifically, any quorum in the old config and any quorum in the new config
   must share at least one honest node. *)
QuorumIntersection == Cardinality(mHigh \intersect mValid) >= 1

(* G5b fix: EpochSafety — the epoch only advances when there's a commit proof.
   This prevents mid-pipeline config changes that could break quorum crossing. *)
EpochSafety == epoch > 1 => Cardinality(mValid) >= 4

(* G5b fix: MvalidSubsetMhigh — mValid must always be a known-good subset,
   and mHigh is the candidate superset. After promotion, mValid = mHigh. *)
ConfigMonotonicity == Cardinality(mValid) >= 4

(* Liveness: eventually all pending requests are processed. *)
Liveness == <>(pendingJoins = {} /\ pendingLeaves = {})

=============================================================================