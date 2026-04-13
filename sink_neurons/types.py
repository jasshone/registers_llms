from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WindowArtifactMetadata:
    dataset_name: str
    dataset_config: str
    dataset_split: str
    model_id: str
    window_length: int
    stride: int
    max_windows: int | None
    bos_token_id: int
    num_windows: int
    tokenized_stream_length: int


@dataclass(frozen=True)
class ScoreArtifactMetadata:
    model_id: str
    score_name: str
    window_artifact: str
    num_layers: int
    intermediate_size: int
    num_examples: int


@dataclass(frozen=True)
class SelectionArtifactMetadata:
    model_id: str
    score_artifact: str
    method: str
    count: int
    percentile: float | None = None
    topk: int | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class RelocationArtifactMetadata:
    model_id: str
    window_artifact: str
    selection_artifact: str
    num_layers: int
    num_examples: int
    summary_metric_name: str


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Expected dataclass instance, got {type(value)!r}")


def normalize_path(path: Path) -> str:
    return str(path.resolve())
