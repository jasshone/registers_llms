from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from .artifacts import save_artifact
from .types import normalize_path
from .windows import build_window_dataloader


@dataclass(frozen=True)
class BosDiagnosticsResult:
    mean_token_norms: torch.Tensor
    mean_bos_attention_by_layer: torch.Tensor
    mean_bos_attention_by_layer_head: torch.Tensor
    num_examples: int


@torch.no_grad()
def measure_bos_diagnostics(
    model: PreTrainedModel,
    windows: torch.Tensor,
    *,
    batch_size: int,
    show_progress: bool = True,
) -> BosDiagnosticsResult:
    dataloader = build_window_dataloader(windows, batch_size=batch_size)
    num_layers = model.config.num_hidden_layers
    seq_len = int(windows.shape[1])
    num_heads = model.config.num_attention_heads

    token_norm_sum = torch.zeros((num_layers, seq_len), dtype=torch.float64)
    bos_attn_layer_sum = torch.zeros(num_layers, dtype=torch.float64)
    bos_attn_layer_head_sum = torch.zeros((num_layers, num_heads), dtype=torch.float64)
    total_examples = 0

    for batch in tqdm(dataloader, desc="BOS diagnostics", disable=not show_progress, dynamic_ncols=True):
        batch = batch.to(model.device)
        outputs = model(
            input_ids=batch,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        bsz = int(batch.shape[0])
        total_examples += bsz

        for layer_idx, hidden in enumerate(outputs.hidden_states[1:]):
            # hidden: [batch, seq, hidden_size]
            norms = torch.linalg.vector_norm(hidden.float(), dim=-1).sum(dim=0).cpu()
            token_norm_sum[layer_idx] += norms

        for layer_idx, attn in enumerate(outputs.attentions):
            # attn: [batch, heads, query, key]
            bos_recv = attn[:, :, 1:, 0].float()
            bos_attn_layer_sum[layer_idx] += bos_recv.mean(dim=(1, 2)).sum().item()
            bos_attn_layer_head_sum[layer_idx] += bos_recv.mean(dim=2).sum(dim=0).cpu()

    mean_token_norms = (token_norm_sum / total_examples).float()
    mean_bos_attention_by_layer = (bos_attn_layer_sum / total_examples).float()
    mean_bos_attention_by_layer_head = (bos_attn_layer_head_sum / total_examples).float()
    return BosDiagnosticsResult(
        mean_token_norms=mean_token_norms,
        mean_bos_attention_by_layer=mean_bos_attention_by_layer,
        mean_bos_attention_by_layer_head=mean_bos_attention_by_layer_head,
        num_examples=total_examples,
    )


def save_bos_diagnostics_artifact(
    output_path: Path,
    *,
    result: BosDiagnosticsResult,
    model_id: str,
    window_artifact: Path,
    batch_size: int,
) -> None:
    metadata = {
        "model_id": model_id,
        "window_artifact": normalize_path(window_artifact),
        "num_layers": int(result.mean_token_norms.shape[0]),
        "sequence_length": int(result.mean_token_norms.shape[1]),
        "num_examples": result.num_examples,
    }
    save_artifact(
        output_path,
        tensors={
            "mean_token_norms": result.mean_token_norms,
            "mean_bos_attention_by_layer": result.mean_bos_attention_by_layer,
            "mean_bos_attention_by_layer_head": result.mean_bos_attention_by_layer_head,
        },
        metadata=metadata,
        config={"batch_size": batch_size},
    )
