from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase

from sink_neurons.artifacts import load_pt, save_artifact
from sink_neurons.types import normalize_path
from sink_neurons.windows import DATASET_CONFIG, DATASET_NAME, iter_token_ids


@dataclass(frozen=True)
class StreamBuildConfig:
    model_id: str
    split: str
    num_streams: int
    stream_length: int


def build_streams_tensor(
    tokenizer: PreTrainedTokenizerBase,
    *,
    split: str,
    num_streams: int,
    stream_length: int,
) -> tuple[torch.Tensor, int]:
    if num_streams <= 0:
        raise ValueError("num_streams must be positive")
    if stream_length < 2:
        raise ValueError("stream_length must be at least 2")
    if tokenizer.bos_token_id is None:
        raise ValueError("Tokenizer must define bos_token_id")

    needed_content_tokens = num_streams * (stream_length - 1)
    content: list[int] = []
    for token_id in iter_token_ids(tokenizer, split=split):
        content.append(int(token_id))
        if len(content) >= needed_content_tokens:
            break
    if len(content) < needed_content_tokens:
        raise ValueError("Not enough dataset tokens to build requested streams")

    streams: list[list[int]] = []
    cursor = 0
    for _ in range(num_streams):
        stream_content = content[cursor : cursor + stream_length - 1]
        cursor += stream_length - 1
        streams.append([int(tokenizer.bos_token_id), *stream_content])
    return torch.tensor(streams, dtype=torch.long), len(content)


def save_streams_artifact(
    output_path: Path,
    *,
    streams: torch.Tensor,
    config: StreamBuildConfig,
    tokenized_stream_length: int,
    bos_token_id: int,
) -> None:
    metadata = {
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": config.split,
        "model_id": config.model_id,
        "num_streams": config.num_streams,
        "stream_length": config.stream_length,
        "bos_token_id": bos_token_id,
        "tokenized_stream_length": tokenized_stream_length,
    }
    save_artifact(
        output_path,
        tensors={"streams": streams},
        metadata=metadata,
        config={
            "model_id": config.model_id,
            "split": config.split,
            "num_streams": config.num_streams,
            "stream_length": config.stream_length,
        },
    )


def load_streams_artifact(path: Path) -> tuple[torch.Tensor, dict]:
    payload = load_pt(path)
    if "streams" in payload["tensors"]:
        return payload["tensors"]["streams"], payload
    if "windows" in payload["tensors"]:
        return payload["tensors"]["windows"], payload
    raise KeyError(f"{normalize_path(path)} does not contain streams or windows")

