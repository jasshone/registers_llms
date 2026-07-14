from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator

import torch


@dataclass
class StageTiming:
    name: str
    seconds: float
    peak_cuda_memory_mb: float | None = None


@dataclass
class TimingRecorder:
    stages: list[StageTiming] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                peak_memory = None
            self.stages.append(
                StageTiming(
                    name=name,
                    seconds=perf_counter() - start,
                    peak_cuda_memory_mb=peak_memory,
                )
            )

    def as_dict(self) -> dict[str, object]:
        total = sum(stage.seconds for stage in self.stages)
        return {
            "total_seconds": total,
            "stages": [
                {
                    "name": stage.name,
                    "seconds": stage.seconds,
                    "peak_cuda_memory_mb": stage.peak_cuda_memory_mb,
                }
                for stage in self.stages
            ],
        }
