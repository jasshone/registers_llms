from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from .artifacts import load_pt, save_artifact
from .types import WindowArtifactMetadata, normalize_path


DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"


@dataclass(frozen=True)
class WindowBuildConfig:
    model_id: str
    split: str
    window_length: int
    stride: int
    max_windows: int | None


class TensorWindowDataset(Dataset[torch.Tensor]):
    def __init__(self, windows: torch.Tensor) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]


def iter_token_ids(tokenizer: PreTrainedTokenizerBase, split: str) -> Iterator[int]:
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    for item in dataset:
        text = item["text"]
        if not text:
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        for token_id in token_ids:
            yield int(token_id)


def build_windows_tensor(
    tokenizer: PreTrainedTokenizerBase,
    *,
    split: str,
    window_length: int,
    stride: int,
    max_windows: int | None = None,
    show_progress: bool = True,
) -> tuple[torch.Tensor, int]:
    if window_length < 2:
        raise ValueError("window_length must be at least 2 to allow BOS + text tokens")
    if stride <= 0:
        raise ValueError("stride must be positive")
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        raise ValueError("Tokenizer must define bos_token_id")

    content_length = window_length - 1
    windows: list[list[int]] = []
    token_buffer: list[int] = []
    tokenized_stream_length = 0

    progress = tqdm(
        desc="Building windows",
        disable=not show_progress,
        unit="tok",
        dynamic_ncols=True,
    )
    for token_id in iter_token_ids(tokenizer, split=split):
        token_buffer.append(token_id)
        tokenized_stream_length += 1
        progress.update(1)

        while len(token_buffer) >= content_length:
            windows.append([bos_token_id, *token_buffer[:content_length]])
            progress.set_postfix(windows=len(windows))
            if max_windows is not None and len(windows) >= max_windows:
                break

            drop = min(stride, len(token_buffer))
            token_buffer = token_buffer[drop:]
        if max_windows is not None and len(windows) >= max_windows:
            break
    progress.close()

    if not windows:
        raise ValueError("No windows were produced; check split/window length")
    return torch.tensor(windows, dtype=torch.long), tokenized_stream_length


def save_windows_artifact(
    output_path: Path,
    *,
    windows: torch.Tensor,
    config: WindowBuildConfig,
    tokenized_stream_length: int,
    bos_token_id: int,
) -> None:
    metadata = WindowArtifactMetadata(
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        dataset_split=config.split,
        model_id=config.model_id,
        window_length=config.window_length,
        stride=config.stride,
        max_windows=config.max_windows,
        bos_token_id=bos_token_id,
        num_windows=int(windows.shape[0]),
        tokenized_stream_length=tokenized_stream_length,
    )
    save_artifact(
        output_path,
        tensors={"windows": windows},
        metadata=metadata,
        config={
            "model_id": config.model_id,
            "split": config.split,
            "window_length": config.window_length,
            "stride": config.stride,
            "max_windows": config.max_windows,
        },
    )


def load_windows_artifact(path: Path) -> tuple[torch.Tensor, dict]:
    payload = load_pt(path)
    return payload["tensors"]["windows"], payload


def build_window_dataloader(windows: torch.Tensor, batch_size: int, shuffle: bool = False) -> DataLoader[torch.Tensor]:
    dataset = TensorWindowDataset(windows)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
