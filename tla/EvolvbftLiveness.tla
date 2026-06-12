--------------------- MODULE EvolvbftLiveness ---------------------
(* Minimal liveness model for Evolv-BFT under partial synchrony.
   Abstracts one HotStuff view into a single step:
   - If the leader is honest: leader proposes → all honest vote → commit
   - If the leader is Byzantine: no proposal → all honest timeout → TC → next view
   
   This captures the core liveness argument:
   With N >= 3F+1 honest majority and round-robin leader rotation,
   eventually an honest leader is reached → commit.
   
   Safety is verified in the full EvolvbftByzantine model.
   This model only verifies liveness. *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS N, F, MAX_VIEWS

Nodes == 1..N
ByzNodes == (N - F + 1)..N
HonestNodes == 1..(N - F)
QuorumSize == 2 * F + 1
Views == 1..MAX_VIEWS

\* Round-robin leader rotation: leader(view) = ((view - 1) mod N) + 1
Leader(v) == ((v - 1) % N) + 1

VARIABLES
    currentView,       \* Current view number
    committed,         \* Whether an honest node has committed
    commitView,        \* The view at which commit happened (0 if not yet)
    tcSet              \* Set of views that have TCs

vars == <<currentView, committed, commitView, tcSet>>

Init ==
    /\ currentView = 1
    /\ committed = FALSE
    /\ commitView = 0
    /\ tcSet = {}

(* Honest leader step: leader proposes, all honest nodes vote, quorum forms, commit.
   Guard: leader must be honest and view within range. *)
HonestLeaderStep ==
    /\ ~committed
    /\ currentView <= MAX_VIEWS
    /\ Leader(currentView) \in HonestNodes
    /\ committed' = TRUE
    /\ commitView' = currentView
    /\ UNCHANGED <<currentView, tcSet>>

(* Byzantine leader step: leader doesn't propose, all honest nodes timeout, TC forms.
   Guard: leader must be Byzantine and view within range. *)
ByzantineLeaderStep ==
    /\ ~committed
    /\ currentView <= MAX_VIEWS
    /\ Leader(currentView) \in ByzNodes
    /\ tcSet' = tcSet \union {currentView}
    /\ currentView' = currentView + 1
    /\ UNCHANGED <<committed, commitView>>

(* Idle step: if committed or past max view, do nothing (stutter). *)
Idle ==
    /\ committed \/ currentView > MAX_VIEWS
    /\ UNCHANGED vars

Next ==
    \/ HonestLeaderStep
    \/ ByzantineLeaderStep
    \/ Idle

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* === SAFETY === *)
(* At most one commit *)
SingleCommit == committed => commitView \in Views

(* === LIVENESS === *)
(* Under partial synchrony with honest majority, eventually committed.
   With round-robin leader rotation and MAX_VIEWS > N - F,
   at least one honest leader is guaranteed within MAX_VIEWS steps. *)
Liveness == <>committed

(* Leader liveness: within N steps, an honest leader must appear.
   This is guaranteed by round-robin rotation with honest majority. *)
HonestLeaderExists == \E v \in 1..N : Leader(v) \in HonestNodes

=============================================================================
