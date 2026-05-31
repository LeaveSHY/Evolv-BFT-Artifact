from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from marl.schemas import Observation, runtime_trace_sample_from_dict
from marl.service import PolicyService, _validate_checkpoint_path, _validate_trace_path
from marl.sfac_bridge import SFACBridge
from marl.torch_sfac_bridge import TorchSFACBridge


class ObservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def coerce_root_payload(cls, value):
        if isinstance(value, dict):
            return {"payload": value}
        raise ValueError("observation request must be an object")

    @model_validator(mode="after")
    def validate_payload(self) -> "ObservationRequest":
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be an object")
        Observation.from_dict(self.payload)
        return self


class TracePathRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_trace_path(self) -> "TracePathRequest":
        _validate_trace_path(self.trace_path, field_name="trace_path")
        return self


class TrainOnlineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=1, ge=1)


class PathRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_path(self) -> "PathRequest":
        _validate_checkpoint_path(self.path, field_name="path")
        if len(PurePosixPath(self.path).parts) != 1:
            raise ValueError("path must be a single checkpoint filename inside the workspace")
        return self


class TrajectorySampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_name: str = Field(default="unknown", min_length=1)
    observation: Any
    candidate: Any = Field(default_factory=dict)
    governed: Any = Field(default_factory=dict)
    masked: Any = Field(default_factory=dict)
    applied: Any = Field(default_factory=dict)
    reward: float = 0.0
    timestamp: str | None = None
    next_observation: Any = None
    done: bool = False
    team_reward: float | None = None
    role_rewards: dict[str, float] = Field(default_factory=dict)
    governance_delta: bool = False
    guardrail_delta: bool = False
    schema_version: str | None = None
    trace: Any = Field(default_factory=dict)


service = PolicyService()
sfac = SFACBridge()
torch_sfac = TorchSFACBridge(m_instances=10, train_mode=True)
app = FastAPI(title="Evolv-BFT Unified MARL Service", version="0.3.0")

def _error_detail(exc: Exception) -> str:
    if isinstance(exc, (FileNotFoundError, PermissionError, OSError)):
        return "workspace file operation failed"
    return str(exc)


def _unprocessable(exc: Exception) -> HTTPException:
    return HTTPException(status_code=422, detail=_error_detail(exc))


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model_ready": service.model_snapshot() is not None}


@app.get("/organization")
def organization() -> dict:
    return service.organization_snapshot()


@app.get("/schema")
def schema() -> dict:
    return service.schema_snapshot()


@app.get("/replay")
def replay() -> dict:
    return service.replay_summary()


@app.get("/adaptive")
def adaptive() -> dict:
    return service.adaptive_snapshot()


@app.post("/inspect")
def inspect(observation: ObservationRequest) -> dict:
    obs = Observation.from_dict(observation.payload)
    return service.inspect_decision_flow(obs)


@app.post("/infer")
def infer(observation: ObservationRequest) -> dict:
    obs = Observation.from_dict(observation.payload)
    return service.decision_envelope(obs)


@app.post("/train/offline")
def train_offline(request: TracePathRequest) -> dict:
    try:
        return service.train_offline(request.trace_path)
    except (ValueError, FileNotFoundError, PermissionError, OSError) as exc:
        raise _unprocessable(exc) from exc


@app.post("/trace/ingest")
def trace_ingest(sample: TrajectorySampleRequest) -> dict:
    try:
        return service.ingest(runtime_trace_sample_from_dict(sample.model_dump(exclude_none=True)))
    except (TypeError, ValueError) as exc:
        raise _unprocessable(exc) from exc


@app.post("/train/online")
def train_online(request: TrainOnlineRequest) -> dict:
    try:
        return service.train_online(request.batch_size)
    except ValueError as exc:
        raise _unprocessable(exc) from exc


@app.post("/checkpoint/save")
def checkpoint_save(request: PathRequest) -> dict:
    try:
        return service.save_checkpoint(request.path)
    except (ValueError, PermissionError, OSError) as exc:
        raise _unprocessable(exc) from exc


@app.post("/checkpoint/load")
def checkpoint_load(request: PathRequest) -> dict:
    try:
        return service.load_checkpoint(request.path)
    except (ValueError, FileNotFoundError, PermissionError, OSError) as exc:
        raise _unprocessable(exc) from exc


# ═══════════════════════════════════════════════════════════════════════════════
# SFAC Bridge Endpoints (consolidated from experiments/sfac_server.py)
# ═══════════════════════════════════════════════════════════════════════════════


class SFACDecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epoch: int = 0
    num_instances: int = 4
    instances: list[dict] = Field(default_factory=list)
    global_state: list[float] = Field(default_factory=list)


class SFACFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_instance_rewards: list[float] = Field(default_factory=list)
    role_rewards: dict[str, float] = Field(default_factory=dict)
    done: bool = False


@app.get("/sfac/health")
def sfac_health() -> dict:
    return sfac.health()


@app.post("/sfac/decide")
def sfac_decide(request: SFACDecideRequest) -> dict:
    if not sfac.available:
        raise HTTPException(status_code=503, detail="SFAC controller not available")
    return sfac.decide(request.model_dump())


@app.post("/sfac/feedback")
def sfac_feedback(request: SFACFeedbackRequest) -> dict:
    if not sfac.available:
        raise HTTPException(status_code=503, detail="SFAC controller not available")
    return sfac.feedback(request.model_dump())


@app.post("/sfac/reset")
def sfac_reset() -> dict:
    return sfac.reset()


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch GPU-Accelerated SFAC Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


class TorchSFACDecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epoch: int = 0
    num_instances: int = 10
    instances: list[dict] = Field(default_factory=list)
    global_state: list[float] = Field(default_factory=list)


class TorchSFACFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_instance_rewards: list[float] = Field(default_factory=list)
    role_rewards: dict[str, float] = Field(default_factory=dict)
    done: bool = False


class TorchCheckpointRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)


@app.get("/torch/health")
def torch_health() -> dict:
    return torch_sfac.health()


@app.get("/torch/metrics")
def torch_metrics() -> dict:
    return torch_sfac.metrics()


@app.post("/torch/decide")
def torch_decide(request: TorchSFACDecideRequest) -> dict:
    return torch_sfac.decide(request.model_dump())


@app.post("/torch/feedback")
def torch_feedback(request: TorchSFACFeedbackRequest) -> dict:
    return torch_sfac.feedback(request.model_dump())


@app.post("/torch/reset")
def torch_reset() -> dict:
    return torch_sfac.reset()


@app.post("/torch/save")
def torch_save(request: TorchCheckpointRequest) -> dict:
    try:
        return torch_sfac.save(request.path)
    except (OSError, PermissionError) as exc:
        raise _unprocessable(exc) from exc


@app.post("/torch/load")
def torch_load(request: TorchCheckpointRequest) -> dict:
    try:
        return torch_sfac.load(request.path)
    except (FileNotFoundError, OSError) as exc:
        raise _unprocessable(exc) from exc


@app.get("/torch/export")
def torch_export() -> dict:
    return torch_sfac.export_numpy()
