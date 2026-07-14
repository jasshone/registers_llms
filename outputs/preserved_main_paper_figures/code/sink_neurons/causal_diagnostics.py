from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator

import torch
from transformers import PreTrainedModel

from .intervention import build_intervened_inputs, patched_mlp_reroute_forwards, _split_mlp_forward
from .modeling import get_mlp_intermediate_size, get_transformer_layers, resolve_device
from .windows import build_window_dataloader


@dataclass(frozen=True)
class DiagnosticResult:
    mean_nll: float
    ppl: float
    bos_attention_mean: float
    bos_attention_max_layer: float
    bos_attention_argmax_layer: int
    dummy_attention_mean: float | None = None
    dummy_attention_max_layer: float | None = None
    dummy_attention_argmax_layer: int | None = None
    dummy_minus_bos_mean: float | None = None
    token_count: int = 0
    num_windows: int = 0


def layer_counts(selected: torch.Tensor) -> str:
    counts: dict[int, int] = {}
    for layer_idx, _neuron_idx in selected.tolist():
        counts[int(layer_idx)] = counts.get(int(layer_idx), 0) + 1
    return " ".join(f"{layer}:{counts[layer]}" for layer in sorted(counts))


def layer_matched_random_selection(
    model: PreTrainedModel,
    selected: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    width = get_mlp_intermediate_size(model)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    rows: list[tuple[int, int]] = []
    by_layer: dict[int, set[int]] = {}
    for layer_idx, neuron_idx in selected.cpu().tolist():
        by_layer.setdefault(int(layer_idx), set()).add(int(neuron_idx))

    for layer_idx in sorted(by_layer):
        excluded = by_layer[layer_idx]
        count = len(excluded)
        pool = torch.tensor([idx for idx in range(width) if idx not in excluded], dtype=torch.long)
        if pool.numel() < count:
            raise ValueError(f"layer {layer_idx} has only {pool.numel()} random-control neurons for {count} selected")
        chosen = pool[torch.randperm(pool.numel(), generator=generator)[:count]]
        rows.extend((layer_idx, int(neuron_idx)) for neuron_idx in chosen.tolist())
    return torch.tensor(rows, dtype=torch.long)


def _window_nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    valid = shift_labels.ne(-100)
    return (losses * valid).sum(dim=1), valid.sum(dim=1)


@contextmanager
def patched_mlp_zero_positions(
    model: PreTrainedModel,
    selected: torch.Tensor,
    *,
    positions: tuple[int, ...] = (0,),
) -> Iterator[None]:
    originals = {}
    grouped: dict[int, list[int]] = {}
    for layer_idx, neuron_idx in selected.tolist():
        grouped.setdefault(int(layer_idx), []).append(int(neuron_idx))

    try:
        layers = get_transformer_layers(model)
        for layer_idx, neurons in grouped.items():
            mlp = layers[layer_idx].mlp
            originals[layer_idx] = mlp.forward
            selected_neurons = torch.tensor(sorted(neurons), dtype=torch.long)

            def make_forward(layer_mlp: torch.nn.Module, selected_for_layer: torch.Tensor):
                def forward(x: torch.Tensor) -> torch.Tensor:
                    selected_on_device = selected_for_layer.to(x.device)
                    gate_act, project = _split_mlp_forward(layer_mlp, x)
                    gate_act = gate_act.clone()
                    for pos in positions:
                        resolved = pos if pos >= 0 else x.shape[1] + pos
                        if 0 <= resolved < x.shape[1]:
                            gate_act[:, resolved, selected_on_device] = 0.0
                    return project(gate_act)

                return forward

            mlp.forward = make_forward(mlp, selected_neurons)  # type: ignore[method-assign]
        yield
    finally:
        layers = get_transformer_layers(model)
        for layer_idx, original in originals.items():
            layers[layer_idx].mlp.forward = original  # type: ignore[method-assign]


def _attention_summary(
    attentions: tuple[torch.Tensor, ...],
    *,
    query_start: int,
    bos_idx: int,
    dummy_idx: int | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    bos_values = []
    dummy_values = []
    for attn in attentions:
        bos_values.append(attn[:, :, query_start:, bos_idx].float().mean().detach().cpu())
        if dummy_idx is not None:
            dummy_values.append(attn[:, :, query_start:, dummy_idx].float().mean().detach().cpu())
    bos = torch.stack(bos_values)
    dummy = torch.stack(dummy_values) if dummy_values else None
    return bos, dummy


@torch.no_grad()
def evaluate_diagnostic_mode(
    model: PreTrainedModel,
    windows: torch.Tensor,
    *,
    batch_size: int,
    mode: str,
    selected: torch.Tensor | None = None,
    relocation_scale: float = 1.0,
    show_progress: bool = False,
) -> DiagnosticResult:
    if mode not in {"clean", "zero_global", "zero_bos", "dummy_only", "relocate"}:
        raise ValueError(f"unsupported diagnostic mode: {mode}")
    if mode in {"zero_global", "zero_bos", "relocate"} and selected is None:
        raise ValueError(f"mode {mode} requires selected neurons")

    dataloader = build_window_dataloader(windows, batch_size=batch_size)
    loss_sums: list[torch.Tensor] = []
    token_counts: list[torch.Tensor] = []
    bos_attn_sum: torch.Tensor | None = None
    dummy_attn_sum: torch.Tensor | None = None
    total_examples = 0
    device = resolve_device(model)

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        tqdm = lambda iterable, **_kwargs: iterable  # type: ignore[assignment]

    for batch in tqdm(dataloader, desc=f"diagnostic {mode}", disable=not show_progress, dynamic_ncols=True):
        batch = batch.to(device)
        hook_context = nullcontext()
        query_start = 1
        bos_idx = 0
        dummy_idx: int | None = None
        labels = batch.clone()
        labels[:, 0] = -100

        if mode == "zero_global":
            from .intervention import patched_mlp_zero_forwards

            hook_context = patched_mlp_zero_forwards(model, selected)  # type: ignore[arg-type]
            model_kwargs = {"input_ids": batch}
        elif mode == "zero_bos":
            hook_context = patched_mlp_zero_positions(model, selected, positions=(0,))  # type: ignore[arg-type]
            model_kwargs = {"input_ids": batch}
        elif mode == "dummy_only":
            inputs_embeds, attention_mask = build_intervened_inputs(model, batch, dummy_init="zero", num_dummy_tokens=1)
            model_kwargs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
            labels = torch.full((batch.shape[0], batch.shape[1] + 1), -100, dtype=torch.long, device=device)
            labels[:, 2:] = batch[:, 1:]
            query_start = 2
            dummy_idx = 1
        elif mode == "relocate":
            hook_context = patched_mlp_reroute_forwards(
                model,
                selected,  # type: ignore[arg-type]
                num_dummy_tokens=1,
                relocation_scale=relocation_scale,
                relocation_fraction=1.0,
            )
            inputs_embeds, attention_mask = build_intervened_inputs(model, batch, dummy_init="zero", num_dummy_tokens=1)
            model_kwargs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
            labels = torch.full((batch.shape[0], batch.shape[1] + 1), -100, dtype=torch.long, device=device)
            labels[:, 2:] = batch[:, 1:]
            query_start = 2
            dummy_idx = 1
        else:
            model_kwargs = {"input_ids": batch}

        with hook_context:
            outputs = model(
                **model_kwargs,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        loss_sum, token_count = _window_nll_from_logits(outputs.logits, labels)
        loss_sums.append(loss_sum.detach().cpu())
        token_counts.append(token_count.detach().cpu())

        bos, dummy = _attention_summary(outputs.attentions, query_start=query_start, bos_idx=bos_idx, dummy_idx=dummy_idx)
        bsz = int(batch.shape[0])
        total_examples += bsz
        bos_attn_sum = bos * bsz if bos_attn_sum is None else bos_attn_sum + bos * bsz
        if dummy is not None:
            dummy_attn_sum = dummy * bsz if dummy_attn_sum is None else dummy_attn_sum + dummy * bsz

    loss = torch.cat(loss_sums)
    counts = torch.cat(token_counts)
    total_tokens = int(counts.sum().item())
    mean_nll = float(loss.sum().item() / total_tokens)
    bos_by_layer = (bos_attn_sum / total_examples).float()  # type: ignore[operator]
    bos_argmax = int(torch.argmax(bos_by_layer).item())
    dummy_mean = None
    dummy_max = None
    dummy_argmax = None
    dummy_minus_bos = None
    if dummy_attn_sum is not None:
        dummy_by_layer = (dummy_attn_sum / total_examples).float()
        dummy_mean = float(dummy_by_layer.mean().item())
        dummy_max = float(dummy_by_layer.max().item())
        dummy_argmax = int(torch.argmax(dummy_by_layer).item())
        dummy_minus_bos = float((dummy_by_layer - bos_by_layer).mean().item())

    return DiagnosticResult(
        mean_nll=mean_nll,
        ppl=math.exp(mean_nll),
        bos_attention_mean=float(bos_by_layer.mean().item()),
        bos_attention_max_layer=float(bos_by_layer.max().item()),
        bos_attention_argmax_layer=bos_argmax,
        dummy_attention_mean=dummy_mean,
        dummy_attention_max_layer=dummy_max,
        dummy_attention_argmax_layer=dummy_argmax,
        dummy_minus_bos_mean=dummy_minus_bos,
        token_count=total_tokens,
        num_windows=int(windows.shape[0]),
    )
