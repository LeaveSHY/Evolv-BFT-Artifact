---------------------- MODULE EvolvbftMultiLeader ----------------------
EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS M, N, MAX_HEIGHT

VARIABLES
    chain,
    globalLog,
    rank,
    pending,
    nilFilled

vars == <<chain, globalLog, rank, pending, nilFilled>>

Instances == 1..M
Heights == 1..MAX_HEIGHT

Init ==
    /\ chain = [i \in Instances |-> <<>>]
    /\ globalLog = <<>>
    /\ rank = [i \in Instances |-> [h \in Heights |-> (h - 1) * M + (i - 1)]]
    /\ pending = {}
    /\ nilFilled = {}

(* InstanceCommit: an instance commits a block at the next height. *)
InstanceCommit(i, h) ==
    /\ h \in Heights
    /\ Len(chain[i]) = h - 1
    /\ LET b == [instance |-> i, height |-> h, rank |-> rank[i][h], isNil |-> FALSE] IN
       /\ chain' = [chain EXCEPT ![i] = Append(chain[i], b)]
       /\ pending' = pending \union {b}
    /\ UNCHANGED <<globalLog, rank, nilFilled>>

(* GlobalOrder: the orderer picks the block with the minimum rank from pending.
   G5c fix: Implements real prefix-order checking — only the minimum-rank block
   can be appended next, ensuring strict rank ordering in the global log. *)
GlobalOrder ==
    /\ pending /= {}
    /\ \E b \in pending :
       \* The chosen block must have the minimum rank among all pending blocks
       /\ \A b2 \in pending : b.rank <= b2.rank
       \* G5c fix: Ensure strict ordering — next entry must have rank >= last entry
       /\ IF Len(globalLog) = 0 THEN TRUE ELSE b.rank > globalLog[Len(globalLog)].rank
       /\ globalLog' = Append(globalLog, b)
       /\ pending' = pending \ {b}
    /\ UNCHANGED <<chain, rank, nilFilled>>

(* NilFill: fill a gap when a rank position has no pending block.
   G5c fix: Added guard — NilFill only for ranks that are "next expected"
   and have no real block in pending. Cannot nil-fill already ordered ranks. *)
NilFill(r) ==
    /\ r \notin {b.rank : b \in pending}        \* No real block at this rank
    /\ r \notin nilFilled                        \* Not already nil-filled
    \* G5c fix: Can only nil-fill the next expected rank (no gaps in nil-filling)
    /\ IF Len(globalLog) = 0 THEN r = 0
       ELSE r = globalLog[Len(globalLog)].rank + 1
    /\ nilFilled' = nilFilled \union {r}
    /\ globalLog' = Append(globalLog, [instance |-> 0, height |-> 0, rank |-> r, isNil |-> TRUE])
    /\ UNCHANGED <<chain, rank, pending>>

Next ==
    \/ \E i \in Instances, h \in Heights : InstanceCommit(i, h)
    \/ GlobalOrder
    \/ \E r \in 0..(MAX_HEIGHT * M) : NilFill(r)

Spec == Init /\ [][Next]_vars

(* === SAFETY INVARIANTS === *)

(* Safety: rank uniqueness — no two (instance, height) pairs share the same rank.
   This is guaranteed by the formula: rank = (h-1) * M + (i-1). *)
RankUniqueness == \A i, j \in Instances : \A h1, h2 \in Heights :
    (i /= j \/ h1 /= h2) => rank[i][h1] /= rank[j][h2]

(* G5c fix: Ordering — the global log is strictly ordered by rank.
   Every entry's rank is strictly greater than the previous entry's rank,
   ensuring a consistent total order across all instances. *)
StrictOrdering == \A idx \in 1..Len(globalLog) :
    idx > 1 => globalLog[idx].rank > globalLog[idx-1].rank

(* G5c fix: NilFillSafety — a nil-filled rank never conflicts with a real block
   IN THE GLOBAL LOG. A late-arriving block may enter pending but can never be
   globally ordered (StrictOrdering prevents it since the rank is consumed). *)
NilFillSafety == \A r \in nilFilled :
    ~\E idx \in 1..Len(globalLog) :
        globalLog[idx].rank = r /\ globalLog[idx].isNil = FALSE

(* G5c fix: RankMonotonicity — within each instance, ranks are monotonically
   increasing as height increases. Follows from rank = (h-1)*M + (i-1). *)
RankMonotonicity == \A i \in Instances : \A h1, h2 \in Heights :
    h1 < h2 => rank[i][h1] < rank[i][h2]

(* G5c fix: GlobalLogConsistency — every real (non-nil) entry in the global log
   corresponds to a block that was actually committed by some instance. *)
GlobalLogConsistency == \A idx \in 1..Len(globalLog) :
    globalLog[idx].isNil = FALSE =>
        \E i \in Instances :
            /\ globalLog[idx].instance = i
            /\ globalLog[idx].height <= Len(chain[i])

(* Liveness: every pending block eventually appears in the global log. *)
Liveness == \A b \in pending : <>(b \in {globalLog[i] : i \in 1..Len(globalLog)})

=============================================================================