from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from .artifacts import save_artifact
from .modeling import get_mlp_intermediate_size, get_transformer_layers
from .relocation import RelocationBaseline
from .scoring import BosScoreResult
from .types import normalize_path
from .windows import build_window_dataloader


@dataclass(frozen=True)
class BosDiagnosticsResult:
    mean_token_norms: torch.Tensor
    mean_bos_attention_by_layer: torch.Tensor
    mean_bos_attention_by_layer_head: torch.Tensor
    num_examples: int


@dataclass(frozen=True)
class BosBaselineMetricsResult:
    diagnostics: BosDiagnosticsResult
    scores: BosScoreResult
    relocation_baseline: RelocationBaseline


def _make_score_hook(
    layer_idx: int,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    bos_sum: torch.Tensor,
    nonbos_sum: torch.Tensor,
    nonbos_square_sum: torch.Tensor,
) -> Callable:
    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        gate_act = act_fn(output).float()
        nonbos = gate_act[:, 1:, :]
        bos_sum[layer_idx] += gate_act[:, 0, :].sum(dim=0).cpu()
        nonbos_sum[layer_idx] += nonbos.sum(dim=(0, 1)).cpu()
        nonbos_square_sum[layer_idx] += nonbos.square().sum(dim=(0, 1)).cpu()

    return hook


def _score_projection_and_activation(mlp: torch.nn.Module) -> tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "act_fn"):
        return mlp.gate_proj, mlp.act_fn
    if hasattr(mlp, "c_fc") and hasattr(mlp, "act"):
        return mlp.c_fc, mlp.act
    if hasattr(mlp, "dense_h_to_4h") and hasattr(mlp, "act"):
        return mlp.dense_h_to_4h, mlp.act
    if hasattr(mlp, "fc1") and hasattr(mlp, "activation_fn"):
        return mlp.fc1, mlp.activation_fn
    raise AttributeError("Unsupported MLP architecture for BOS scoring")


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


@torch.no_grad()
def measure_bos_baseline_metrics(
    model: PreTrainedModel,
    windows: torch.Tensor,
    *,
    batch_size: int,
    show_progress: bool = True,
) -> BosBaselineMetricsResult:
    dataloader = build_window_dataloader(windows, batch_size=batch_size)
    num_layers = model.config.num_hidden_layers
    seq_len = int(windows.shape[1])
    num_heads = model.config.num_attention_heads
    intermediate_size = get_mlp_intermediate_size(model)

    token_norm_sum = torch.zeros((num_layers, seq_len), dtype=torch.float64)
    bos_norm_before = torch.zeros(num_layers, dtype=torch.float64)
    bos_attn_layer_sum = torch.zeros(num_layers, dtype=torch.float64)
    bos_attn_layer_head_sum = torch.zeros((num_layers, num_heads), dtype=torch.float64)
    bos_score_sum = torch.zeros((num_layers, intermediate_size), dtype=torch.float64)
    nonbos_score_sum = torch.zeros((num_layers, intermediate_size), dtype=torch.float64)
    nonbos_score_square_sum = torch.zeros((num_layers, intermediate_size), dtype=torch.float64)
    total_examples = 0
    total_nonbos_positions = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []
    layers = get_transformer_layers(model)

    for layer_idx in range(num_layers):
        projection, act_fn = _score_projection_and_activation(layers[layer_idx].mlp)
        handle = projection.register_forward_hook(
            _make_score_hook(
                layer_idx,
                act_fn,
                bos_score_sum,
                nonbos_score_sum,
                nonbos_score_square_sum,
            )
        )
        handles.append(handle)

    try:
        for batch in tqdm(dataloader, desc="BOS baseline metrics", disable=not show_progress, dynamic_ncols=True):
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
            total_nonbos_positions += int(batch.shape[0] * (batch.shape[1] - 1))

            for layer_idx, hidden in enumerate(outputs.hidden_states[1:]):
                hidden_float = hidden.float()
                norms = torch.linalg.vector_norm(hidden_float, dim=-1)
                token_norm_sum[layer_idx] += norms.sum(dim=0).cpu()
                bos_norm_before[layer_idx] += norms[:, 0].sum().item()

            for layer_idx, attn in enumerate(outputs.attentions):
                bos_recv = attn[:, :, 1:, 0].float()
                bos_attn_layer_sum[layer_idx] += bos_recv.mean(dim=(1, 2)).sum().item()
                bos_attn_layer_head_sum[layer_idx] += bos_recv.mean(dim=2).sum(dim=0).cpu()
    finally:
        for handle in handles:
            handle.remove()

    mean_token_norms = (token_norm_sum / total_examples).float()
    mean_bos_attention_by_layer = (bos_attn_layer_sum / total_examples).float()
    mean_bos_attention_by_layer_head = (bos_attn_layer_head_sum / total_examples).float()
    scores = (bos_score_sum / total_examples).float()
    nonbos_mean = (nonbos_score_sum / total_nonbos_positions).float()
    nonbos_mean_square = (nonbos_score_square_sum / total_nonbos_positions).float()
    nonbos_var = (nonbos_mean_square - nonbos_mean.square()).clamp_min(0.0)
    nonbos_std = torch.sqrt(nonbos_var).clamp_min(1e-8)
    bos_minus_nonbos = scores - nonbos_mean
    z_scores = bos_minus_nonbos / nonbos_std

    return BosBaselineMetricsResult(
        diagnostics=BosDiagnosticsResult(
            mean_token_norms=mean_token_norms,
            mean_bos_attention_by_layer=mean_bos_attention_by_layer,
            mean_bos_attention_by_layer_head=mean_bos_attention_by_layer_head,
            num_examples=total_examples,
        ),
        scores=BosScoreResult(
            scores=scores,
            num_examples=total_examples,
            z_scores=z_scores,
            bos_minus_nonbos=bos_minus_nonbos,
            nonbos_std=nonbos_std,
        ),
        relocation_baseline=RelocationBaseline(
            bos_norm_before=(bos_norm_before / total_examples).float(),
            bos_attention_before=mean_bos_attention_by_layer,
            num_examples=total_examples,
        ),
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
