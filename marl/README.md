# Octopus Safe FACMAC Service

This package provides a lightweight FACMAC-compatible external research service for Octopus.

It is not authoritative runtime truth, not a protocol checkpoint service, and not evidence that the Go runtime already embeds full Safe FACMAC or full MOISE+ semantics.

It is designed to work with the Go adaptive control plane already integrated in Octopus:

1. Octopus emits observations and adaptive traces.
2. Octopus can also emit multi-agent observations, one per consensus instance.
3. This service trains from JSONL traces.
4. The Go runtime may call `/infer` through `-adaptive-policy=facmac-http`; this is an external HTTP policy bridge rather than native in-process FACMAC execution.

## Multi-agent action semantics

The service now supports:

- stage-based trajectory samples via `candidate` / `governed` / `masked` / `applied` / `trace`
- decentralized per-instance actor outputs via `agent_actions`
- service-side trainer metadata and policy outputs over the global observation; the README does not treat critic-related internals as a stronger public contract than the code currently exposes
- top-level aggregate action fields inside each stage's `action` payload, as service-side schema support rather than a guarantee that the Go runtime will adopt every field unchanged

Each `agent_action` targets one Octopus consensus instance and can independently tune:

- `committee_size`
- `pacemaker_timeout_ms`
- `mempool_max_batch_txs`
- `mempool_proposal_interval_ms`

## Route contract boundaries

The FastAPI service currently enforces these boundaries:

- `/trace/ingest` accepts stage-based payloads only; the public request shape is `candidate` / `governed` / `masked` / `applied` / `trace`, and there is no legacy flat `action` compatibility route contract
- the example payload below is one concrete stage-based sample shape, not the only semantically valid combination of optional `governed`, `masked`, and `trace` fields
- request models use `extra="forbid"`, so unknown fields are rejected with 422
- `/train/offline`, `/checkpoint/save`, and `/checkpoint/load` accept workspace-relative paths only
- `/adaptive`, `/schema`, `/replay`, and `/organization` expose service-side snapshots, not authoritative Go runtime truth; `/schema` summarizes schema snapshot fields rather than acting as a complete ingest-validator spec
- `/inspect` exposes a service-side decision-flow explanation over candidate / governed / applied actions, not a complete serialized stage object equivalent to the ingest contract

## Run tests

```bash
PYTHONPATH="/mnt/d/Alex/Papers/Experiment/Octopus" python3 -m unittest discover -s marl/tests -v
```

## Run service

```bash
uvicorn marl.app:app --host 127.0.0.1 --port 18080
```

## AIoT scenario driver

The package also includes a simple heterogeneous/dynamic/adversarial AIoT context driver:

```python
from marl.scenario import AIoTScenarioDriver, ScenarioPublisher

driver = AIoTScenarioDriver(seed=7)
publisher = ScenarioPublisher("http://127.0.0.1:9000/adaptive/context")
publisher.publish(driver.next_context(step=0))
```

## Offline training

```bash
curl -X POST http://127.0.0.1:18080/train/offline \
  -H "Content-Type: application/json" \
  -d '{"trace_path": "tmp/octopus-trace.jsonl"}'
```

## Online replay and checkpoints

```bash
curl -X POST http://127.0.0.1:18080/trace/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "policy_name": "safe-facmac",
    "observation": {"validator_count": 8},
    "candidate": {"action": {"pacemaker_timeout_ms": 900}, "present": true},
    "governed": {"action": {"pacemaker_timeout_ms": 900}, "present": true, "mutated": false},
    "masked": {"action": {"pacemaker_timeout_ms": 900}, "present": true, "mutated": false},
    "applied": {"action": {"pacemaker_timeout_ms": 900}, "present": true},
    "trace": {"enabled": true, "write_failed": false, "close_failed": false, "dropped_samples": 0},
    "reward": 1.0
  }'

curl -X POST http://127.0.0.1:18080/train/online \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 32}'

curl -X POST http://127.0.0.1:18080/checkpoint/save \
  -H "Content-Type: application/json" \
  -d '{"path": "tmp/facmac-checkpoint.json"}'

curl -X POST http://127.0.0.1:18080/checkpoint/load \
  -H "Content-Type: application/json" \
  -d '{"path": "tmp/facmac-checkpoint.json"}'
```

## Curriculum and orchestrator

The package now includes:

- `marl.curriculum.CurriculumSchedule`: staged heterogeneous/dynamic/adversarial AIoT difficulty
- `marl.orchestrator.ExperimentOrchestrator`: closed-loop runner for:
  - pushing context to Octopus
  - reading `/adaptive`
  - ingesting trajectories into the MARL service
  - triggering online training
  - saving checkpoints
