from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .types import dataclass_to_dict


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_pt(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    torch.save(payload, path)


def load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def save_artifact(path: Path, tensors: dict[str, Any], metadata: Any, config: dict[str, Any]) -> None:
    payload = {
        "metadata": dataclass_to_dict(metadata),
        "config": config,
        "tensors": tensors,
    }
    save_pt(path, payload)
    save_json(path.with_suffix(".json"), payload["metadata"] | {"config": config})
