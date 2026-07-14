from __future__ import annotations

import contextlib
import importlib
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Literal

import torch
from torch import nn
from transformers import PreTrainedModel

from .modeling import resolve_device
from .windows import build_window_dataloader


TargetKind = Literal["sink", "early", "local", "random"]
HeadKind = Literal["sink_heavy", "low_sink"]
LayerBand = Literal["all", "early", "middle", "late"]
InterventionKind = Literal["attention_reroute", "ordinary_value_at_sink", "sink_value_at_ordinary"]


@dataclass(frozen=True)
class HeadRef:
    layer: int
    head: int
    sink_strength: float


@dataclass(frozen=True)
class HeadSelection:
    sink_position: int
    sink_heavy: tuple[HeadRef, ...]
    low_sink: tuple[HeadRef, ...]
    all_scores: tuple[HeadRef, ...]


@dataclass(frozen=True)
class InterventionSite:
    sequence_index: int
    query_position: int
    layer: int
    head: int
    head_kind: HeadKind
    target_kind: TargetKind
    target_position: int
    delta: float


@dataclass(frozen=True)
class BatchRerouteSpec:
    batch_index: int
    layer: int
    head: int
    query_position: int
    target_position: int
    delta: float


@dataclass(frozen=True)
class SiteResult:
    sequence_index: int
    query_position: int
    layer: int
    head: int
    head_kind: HeadKind
    target_kind: TargetKind
    target_position: int
    delta: float
    kl_clean_intervened: float
    delta_loss: float
    logit_l2: float
    delta_correct_prob: float
    skipped: bool


@dataclass(frozen=True)
class ValueSiteResult:
    sequence_index: int
    query_position: int
    layer: int
    head: int
    control_kind: TargetKind
    sink_position: int
    control_position: int
    intervention_kind: Literal["ordinary_value_at_sink", "sink_value_at_ordinary"]
    kl_clean_intervened: float
    delta_loss: float
    logit_l2: float
    delta_correct_prob: float
    skipped: bool


@dataclass(frozen=True)
class SequenceSummary:
    sequence_index: int
    delta: float
    head_kind: HeadKind
    layer_band: LayerBand
    control_kind: TargetKind
    mean_control_minus_sink_kl: float
    mean_control_minus_sink_loss: float
    mean_control_minus_sink_logit_l2: float
    mean_control_minus_sink_correct_prob: float
    sink_less_disruptive: bool
    num_pairs: int


@dataclass(frozen=True)
class BootstrapSummary:
    delta: float
    head_kind: HeadKind
    layer_band: LayerBand
    control_kind: TargetKind
    mean_paired_difference: float
    ci_low: float
    ci_high: float
    mean_control_minus_sink_loss: float
    mean_control_minus_sink_logit_l2: float
    mean_control_minus_sink_correct_prob: float
    fraction_sink_less_disruptive: float
    num_sequences: int


@dataclass(frozen=True)
class SecondaryResult:
    sequence_index: int
    target_kind: TargetKind
    delta: float
    mean_kl_clean_intervened: float
    mean_delta_loss: float
    mean_logit_l2: float
    mean_delta_correct_prob: float
    applied_sites: int
    skipped_sites: int
    query_positions: int


def layer_band(layer: int, num_layers: int) -> LayerBand:
    if num_layers <= 0:
        return "all"
    cut1 = max(1, num_layers // 3)
    cut2 = max(cut1 + 1, (2 * num_layers) // 3)
    if layer < cut1:
        return "early"
    if layer < cut2:
        return "middle"
    return "late"


def _as_special_set(tokenizer: object | None, extra_special_ids: Iterable[int] = ()) -> set[int]:
    special_ids = {int(x) for x in extra_special_ids}
    if tokenizer is None:
        return special_ids
    for attr in ("all_special_ids", "bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is None:
            continue
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            special_ids.update(int(x) for x in value if x is not None)
        else:
            special_ids.add(int(value))
    return special_ids


def visible_non_special_positions(
    input_ids: torch.Tensor,
    query_position: int,
    *,
    special_ids: set[int],
    exclude: set[int] | None = None,
) -> list[int]:
    exclude = exclude or set()
    upper = min(int(query_position), int(input_ids.numel()) - 1)
    positions = []
    for pos in range(0, upper + 1):
        if pos in exclude:
            continue
        if int(input_ids[pos].item()) in special_ids:
            continue
        positions.append(pos)
    return positions


def choose_target_positions(
    input_ids: torch.Tensor,
    query_position: int,
    *,
    sink_position: int,
    special_ids: set[int],
    rng: random.Random,
) -> dict[TargetKind, int] | None:
    if query_position <= sink_position:
        return None
    exclude = {sink_position}
    early = [pos for pos in range(1, min(5, query_position + 1)) if pos not in exclude and int(input_ids[pos]) not in special_ids]
    local = [
        pos
        for pos in range(max(0, query_position - 16), query_position)
        if pos not in exclude and int(input_ids[pos]) not in special_ids
    ]
    random_pool = visible_non_special_positions(input_ids, query_position, special_ids=special_ids, exclude=exclude)
    if not early or not local or not random_pool:
        return None
    return {
        "sink": int(sink_position),
        "early": int(rng.choice(early)),
        "local": int(rng.choice(local)),
        "random": int(rng.choice(random_pool)),
    }


def redirect_attention_mass(
    attn: torch.Tensor,
    *,
    target_position: int,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add fixed mass to target and proportionally rescale all other source positions.

    Returns the rewritten distribution and a boolean mask marking rows that were skipped
    because the requested target already had too much mass.
    """
    if delta < 0.0:
        raise ValueError("delta must be nonnegative")
    if target_position < 0 or target_position >= attn.shape[-1]:
        raise IndexError(f"target_position {target_position} outside attention width {attn.shape[-1]}")
    target = attn[..., target_position]
    valid = target + delta <= 1.0
    denom = (1.0 - target).clamp_min(1e-12)
    scale = ((1.0 - target - delta).clamp_min(0.0) / denom).to(attn.dtype)
    rewritten = attn * scale.unsqueeze(-1)
    rewritten[..., target_position] = target + delta
    rewritten = torch.where(valid.unsqueeze(-1), rewritten, attn)
    return rewritten.clamp_min(0.0), ~valid


def kl_clean_intervened(clean_logits: torch.Tensor, intervened_logits: torch.Tensor) -> torch.Tensor:
    clean_log_probs = clean_logits.float().log_softmax(dim=-1)
    intervened_log_probs = intervened_logits.float().log_softmax(dim=-1)
    clean_probs = clean_log_probs.exp()
    return (clean_probs * (clean_log_probs - intervened_log_probs)).sum(dim=-1)


def next_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits.float(), labels, reduction="none")


class _AttentionPatchState:
    def __init__(self) -> None:
        self.kind: InterventionKind = "attention_reroute"
        self.layer: int | None = None
        self.head: int | None = None
        self.query_position: int | None = None
        self.target_position: int | None = None
        self.control_position: int | None = None
        self.delta: float = 0.0
        self.skipped = False
        self.skipped_count = 0
        self.applied_count = 0
        self.reroute_specs: dict[tuple[int, int, int], tuple[int, float]] = {}
        self.batch_reroute_specs: list[BatchRerouteSpec] = []
        self.skipped_batch_indices: set[int] = set()


def _repeat_kv_if_needed(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    groups = int(getattr(module, "num_key_value_groups", 1))
    if groups == 1:
        return value
    try:
        from transformers.models.llama.modeling_llama import repeat_kv
    except Exception:
        return value.repeat_interleave(groups, dim=1)
    return repeat_kv(value, groups)


def _patched_eager_attention(state: _AttentionPatchState) -> Callable:
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = _repeat_kv_if_needed(module, key)
        value_states = _repeat_kv_if_needed(module, value)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        layer_idx = getattr(module, "layer_idx", None)
        applies = (
            state.layer is not None
            and state.head is not None
            and state.query_position is not None
            and state.target_position is not None
            and layer_idx is not None
            and int(layer_idx) == int(state.layer)
            and state.query_position < attn_weights.shape[-2]
            and state.head < attn_weights.shape[1]
            and state.target_position < attn_weights.shape[-1]
        )
        if state.kind == "attention_reroute" and state.batch_reroute_specs and layer_idx is not None:
            current_layer = int(layer_idx)
            changed = False
            for spec in state.batch_reroute_specs:
                if int(spec.layer) != current_layer:
                    continue
                if (
                    spec.batch_index >= attn_weights.shape[0]
                    or spec.head >= attn_weights.shape[1]
                    or spec.query_position >= attn_weights.shape[-2]
                    or spec.target_position >= attn_weights.shape[-1]
                ):
                    continue
                if not changed:
                    attn_weights = attn_weights.clone()
                    changed = True
                row = attn_weights[spec.batch_index : spec.batch_index + 1, spec.head, spec.query_position, :]
                rewritten, skipped = redirect_attention_mass(row, target_position=spec.target_position, delta=spec.delta)
                attn_weights[spec.batch_index : spec.batch_index + 1, spec.head, spec.query_position, :] = rewritten
                skipped_items = int(skipped.sum().item())
                if skipped_items:
                    state.skipped_batch_indices.add(int(spec.batch_index))
                state.skipped_count += skipped_items
                state.applied_count += int(skipped.numel()) - skipped_items
            state.skipped = state.skipped_count > 0
        elif state.kind == "attention_reroute" and state.reroute_specs and layer_idx is not None:
            current_layer = int(layer_idx)
            changed = False
            for (spec_layer, spec_head, spec_query), (spec_target, spec_delta) in state.reroute_specs.items():
                if spec_layer != current_layer:
                    continue
                if spec_head >= attn_weights.shape[1] or spec_query >= attn_weights.shape[-2] or spec_target >= attn_weights.shape[-1]:
                    continue
                if not changed:
                    attn_weights = attn_weights.clone()
                    changed = True
                row = attn_weights[:, spec_head, spec_query, :]
                rewritten, skipped = redirect_attention_mass(row, target_position=spec_target, delta=spec_delta)
                attn_weights[:, spec_head, spec_query, :] = rewritten
                skipped_items = int(skipped.sum().item())
                state.skipped_count += skipped_items
                state.applied_count += int(skipped.numel()) - skipped_items
            state.skipped = state.skipped_count > 0
        elif applies and state.kind == "attention_reroute":
            row = attn_weights[:, state.head, state.query_position, :]
            rewritten, skipped = redirect_attention_mass(row, target_position=state.target_position, delta=state.delta)
            attn_weights = attn_weights.clone()
            attn_weights[:, state.head, state.query_position, :] = rewritten
            skipped_items = int(skipped.sum().item())
            state.skipped_count += skipped_items
            state.applied_count += int(skipped.numel()) - skipped_items
            state.skipped = bool(skipped.any().item())
        elif applies and state.control_position is not None and state.control_position < value_states.shape[-2]:
            value_states = value_states.clone()
            if state.kind == "ordinary_value_at_sink":
                value_states[:, state.head, state.target_position, :] = value_states[:, state.head, state.control_position, :]
            elif state.kind == "sink_value_at_ordinary":
                value_states[:, state.head, state.control_position, :] = value_states[:, state.head, state.target_position, :]

        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return eager_attention_forward


@contextlib.contextmanager
def patched_eager_attention(state: _AttentionPatchState) -> Iterator[None]:
    module_names = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.gpt_neox.modeling_gpt_neox",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
    ]
    global_originals: list[tuple[object, object]] = []
    patched = _patched_eager_attention(state)
    try:
        for name in module_names:
            try:
                module = importlib.import_module(name)
            except Exception:
                continue
            original_global = getattr(module, "eager_attention_forward", None)
            if original_global is not None:
                global_originals.append((module, original_global))
                setattr(module, "eager_attention_forward", patched)
        yield
    finally:
        for module, original in global_originals:
            setattr(module, "eager_attention_forward", original)


@torch.no_grad()
def identify_sink_heads(
    model: PreTrainedModel,
    windows: torch.Tensor,
    *,
    batch_size: int,
    query_start: int = 16,
    sink_position: int | None = None,
    topk: int = 16,
    show_progress: bool = False,
) -> HeadSelection:
    dataloader = build_window_dataloader(windows, batch_size=batch_size)
    device = resolve_device(model)
    score_sum: torch.Tensor | None = None
    source_sum: torch.Tensor | None = None
    examples = 0
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        tqdm = lambda iterable, **_kwargs: iterable  # type: ignore[assignment]

    for batch in tqdm(dataloader, desc="identify sink heads", disable=not show_progress, dynamic_ncols=True):
        batch = batch.to(device)
        outputs = model(input_ids=batch, output_attentions=True, use_cache=False, return_dict=True)
        bsz = int(batch.shape[0])
        examples += bsz
        attn_stack = torch.stack([attn.detach().float().cpu() for attn in outputs.attentions])
        # [layers, batch, heads, query, source]
        eligible = attn_stack[:, :, :, query_start:, :]
        if sink_position is None:
            source_vals = eligible.mean(dim=(0, 1, 2, 3))
            source_sum = source_vals * bsz if source_sum is None else source_sum + source_vals * bsz
        source_idx = 0 if sink_position is None else int(sink_position)
        scores = eligible[..., source_idx].mean(dim=(1, 3))
        score_sum = scores * bsz if score_sum is None else score_sum + scores * bsz

    if score_sum is None:
        raise ValueError("no attention scores were produced")
    if sink_position is None:
        if source_sum is None:
            raise ValueError("no source scores were produced")
        sink_position = int(torch.argmax(source_sum / examples).item())
        if sink_position != 0:
            return identify_sink_heads(
                model,
                windows,
                batch_size=batch_size,
                query_start=query_start,
                sink_position=sink_position,
                topk=topk,
                show_progress=show_progress,
            )

    mean_scores = (score_sum / examples).float()
    all_scores = [
        HeadRef(layer=layer, head=head, sink_strength=float(mean_scores[layer, head].item()))
        for layer in range(mean_scores.shape[0])
        for head in range(mean_scores.shape[1])
    ]
    high = sorted(all_scores, key=lambda item: item.sink_strength, reverse=True)[:topk]
    high_keys = {(item.layer, item.head) for item in high}
    low: list[HeadRef] = []
    for layer in sorted({item.layer for item in high}):
        count = sum(1 for item in high if item.layer == layer)
        candidates = [
            item
            for item in all_scores
            if item.layer == layer and (item.layer, item.head) not in high_keys and item not in low
        ]
        low.extend(sorted(candidates, key=lambda item: item.sink_strength)[:count])
    if len(low) < len(high):
        used = high_keys | {(item.layer, item.head) for item in low}
        fill = [item for item in sorted(all_scores, key=lambda item: item.sink_strength) if (item.layer, item.head) not in used]
        low.extend(fill[: len(high) - len(low)])
    return HeadSelection(
        sink_position=int(sink_position),
        sink_heavy=tuple(high),
        low_sink=tuple(low[: len(high)]),
        all_scores=tuple(all_scores),
    )


@torch.no_grad()
def run_attention_site(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    site: InterventionSite,
    target_token_id: int,
) -> SiteResult:
    device = resolve_device(model)
    input_ids = input_ids.to(device).unsqueeze(0)
    clean = model(input_ids=input_ids, use_cache=False, return_dict=True)
    clean_logits = clean.logits[:, site.query_position, :]
    clean_loss = next_token_loss(clean_logits, torch.tensor([target_token_id], device=device))
    clean_prob = clean_logits.float().softmax(dim=-1)[:, target_token_id]

    state = _AttentionPatchState()
    state.kind = "attention_reroute"
    state.layer = site.layer
    state.head = site.head
    state.query_position = site.query_position
    state.target_position = site.target_position
    state.delta = site.delta
    with patched_eager_attention(state):
        intervened = model(input_ids=input_ids, use_cache=False, return_dict=True)
    intervened_logits = intervened.logits[:, site.query_position, :]
    intervened_loss = next_token_loss(intervened_logits, torch.tensor([target_token_id], device=device))
    intervened_prob = intervened_logits.float().softmax(dim=-1)[:, target_token_id]
    return SiteResult(
        sequence_index=site.sequence_index,
        query_position=site.query_position,
        layer=site.layer,
        head=site.head,
        head_kind=site.head_kind,
        target_kind=site.target_kind,
        target_position=site.target_position,
        delta=site.delta,
        kl_clean_intervened=float(kl_clean_intervened(clean_logits, intervened_logits).item()),
        delta_loss=float((intervened_loss - clean_loss).item()),
        logit_l2=float(torch.linalg.vector_norm((intervened_logits - clean_logits).float(), ord=2).item()),
        delta_correct_prob=float((intervened_prob - clean_prob).item()),
        skipped=state.skipped,
    )



@torch.no_grad()
def run_attention_sites_batch(
    model: PreTrainedModel,
    input_ids_batch: list[torch.Tensor],
    *,
    sites: list[InterventionSite],
    target_token_ids: list[int],
) -> list[SiteResult]:
    if not (len(input_ids_batch) == len(sites) == len(target_token_ids)):
        raise ValueError("input_ids_batch, sites, and target_token_ids must have equal length")
    if not sites:
        return []
    lengths = {int(ids.numel()) for ids in input_ids_batch}
    if len(lengths) != 1:
        raise ValueError("batched attention-site execution requires equal-length input windows")
    device = resolve_device(model)
    batch = torch.stack([ids.to(device) for ids in input_ids_batch], dim=0)
    labels = torch.tensor(target_token_ids, device=device, dtype=torch.long)
    query_positions = torch.tensor([site.query_position for site in sites], device=device, dtype=torch.long)
    batch_indices = torch.arange(len(sites), device=device)

    clean = model(input_ids=batch, use_cache=False, return_dict=True)
    clean_logits = clean.logits[batch_indices, query_positions, :]
    clean_loss = next_token_loss(clean_logits, labels)
    clean_prob = clean_logits.float().softmax(dim=-1).gather(-1, labels[:, None]).squeeze(-1)

    state = _AttentionPatchState()
    state.kind = "attention_reroute"
    state.batch_reroute_specs = [
        BatchRerouteSpec(
            batch_index=batch_idx,
            layer=site.layer,
            head=site.head,
            query_position=site.query_position,
            target_position=site.target_position,
            delta=site.delta,
        )
        for batch_idx, site in enumerate(sites)
    ]
    with patched_eager_attention(state):
        intervened = model(input_ids=batch, use_cache=False, return_dict=True)
    intervened_logits = intervened.logits[batch_indices, query_positions, :]
    intervened_loss = next_token_loss(intervened_logits, labels)
    intervened_prob = intervened_logits.float().softmax(dim=-1).gather(-1, labels[:, None]).squeeze(-1)
    kls = kl_clean_intervened(clean_logits, intervened_logits)
    logit_l2 = torch.linalg.vector_norm((intervened_logits - clean_logits).float(), ord=2, dim=-1)

    results: list[SiteResult] = []
    skipped_flags = [idx in state.skipped_batch_indices for idx in range(len(sites))]
    for idx, site in enumerate(sites):
        results.append(
            SiteResult(
                sequence_index=site.sequence_index,
                query_position=site.query_position,
                layer=site.layer,
                head=site.head,
                head_kind=site.head_kind,
                target_kind=site.target_kind,
                target_position=site.target_position,
                delta=site.delta,
                kl_clean_intervened=float(kls[idx].item()),
                delta_loss=float((intervened_loss[idx] - clean_loss[idx]).item()),
                logit_l2=float(logit_l2[idx].item()),
                delta_correct_prob=float((intervened_prob[idx] - clean_prob[idx]).item()),
                skipped=skipped_flags[idx],
            )
        )
    return results


@torch.no_grad()
def run_value_site(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    site: InterventionSite,
    sink_position: int,
    control_position: int,
    intervention_kind: Literal["ordinary_value_at_sink", "sink_value_at_ordinary"],
    target_token_id: int,
) -> ValueSiteResult:
    device = resolve_device(model)
    input_ids = input_ids.to(device).unsqueeze(0)
    clean = model(input_ids=input_ids, use_cache=False, return_dict=True)
    clean_logits = clean.logits[:, site.query_position, :]
    clean_loss = next_token_loss(clean_logits, torch.tensor([target_token_id], device=device))
    clean_prob = clean_logits.float().softmax(dim=-1)[:, target_token_id]

    state = _AttentionPatchState()
    state.kind = intervention_kind
    state.layer = site.layer
    state.head = site.head
    state.query_position = site.query_position
    state.target_position = sink_position
    state.control_position = control_position
    with patched_eager_attention(state):
        intervened = model(input_ids=input_ids, use_cache=False, return_dict=True)
    intervened_logits = intervened.logits[:, site.query_position, :]
    intervened_loss = next_token_loss(intervened_logits, torch.tensor([target_token_id], device=device))
    intervened_prob = intervened_logits.float().softmax(dim=-1)[:, target_token_id]
    return ValueSiteResult(
        sequence_index=site.sequence_index,
        query_position=site.query_position,
        layer=site.layer,
        head=site.head,
        control_kind=site.target_kind,
        sink_position=sink_position,
        control_position=control_position,
        intervention_kind=intervention_kind,
        kl_clean_intervened=float(kl_clean_intervened(clean_logits, intervened_logits).item()),
        delta_loss=float((intervened_loss - clean_loss).item()),
        logit_l2=float(torch.linalg.vector_norm((intervened_logits - clean_logits).float(), ord=2).item()),
        delta_correct_prob=float((intervened_prob - clean_prob).item()),
        skipped=state.skipped,
    )


@torch.no_grad()
def run_secondary_sequence(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    sequence_index: int,
    heads: tuple[HeadRef, ...],
    query_targets: dict[int, dict[TargetKind, int]],
    target_kind: TargetKind,
    delta: float,
) -> SecondaryResult:
    device = resolve_device(model)
    input_ids = input_ids.to(device).unsqueeze(0)
    query_positions = sorted(query_targets)
    if not query_positions:
        raise ValueError("query_targets must contain at least one query position")
    labels = input_ids[:, [pos + 1 for pos in query_positions]]

    clean = model(input_ids=input_ids, use_cache=False, return_dict=True)
    clean_logits = clean.logits[:, query_positions, :]
    clean_loss = next_token_loss(clean_logits.reshape(-1, clean_logits.shape[-1]), labels.reshape(-1)).view(1, -1)
    clean_prob = clean_logits.float().softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    state = _AttentionPatchState()
    state.kind = "attention_reroute"
    state.reroute_specs = {
        (head.layer, head.head, query_pos): (query_targets[query_pos][target_kind], float(delta))
        for head in heads
        for query_pos in query_positions
        if target_kind in query_targets[query_pos]
    }
    with patched_eager_attention(state):
        intervened = model(input_ids=input_ids, use_cache=False, return_dict=True)
    intervened_logits = intervened.logits[:, query_positions, :]
    intervened_loss = next_token_loss(intervened_logits.reshape(-1, intervened_logits.shape[-1]), labels.reshape(-1)).view(1, -1)
    intervened_prob = intervened_logits.float().softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    return SecondaryResult(
        sequence_index=sequence_index,
        target_kind=target_kind,
        delta=float(delta),
        mean_kl_clean_intervened=float(kl_clean_intervened(clean_logits, intervened_logits).mean().item()),
        mean_delta_loss=float((intervened_loss - clean_loss).mean().item()),
        mean_logit_l2=float(torch.linalg.vector_norm((intervened_logits - clean_logits).float(), ord=2, dim=-1).mean().item()),
        mean_delta_correct_prob=float((intervened_prob - clean_prob).mean().item()),
        applied_sites=int(state.applied_count),
        skipped_sites=int(state.skipped_count),
        query_positions=len(query_positions),
    )


def summarize_sequence_pairs(results: Iterable[SiteResult]) -> list[SequenceSummary]:
    rows = [row for row in results if not row.skipped]
    if not rows:
        return []
    num_layers = max(row.layer for row in rows) + 1
    grouped: dict[tuple[int, float, HeadKind, int, int, int], dict[TargetKind, SiteResult]] = {}
    for row in rows:
        key = (row.sequence_index, row.delta, row.head_kind, row.layer, row.head, row.query_position)
        grouped.setdefault(key, {})[row.target_kind] = row
    by_sequence: dict[tuple[int, float, HeadKind, LayerBand, TargetKind], list[tuple[float, float, float, float]]] = {}
    for (seq, delta, head_kind, layer, _head, _query), vals in grouped.items():
        sink = vals.get("sink")
        if sink is None:
            continue
        bands = ("all", layer_band(layer, num_layers))
        for control in ("early", "local", "random"):
            control_row = vals.get(control)
            if control_row is None:
                continue
            diffs = (
                control_row.kl_clean_intervened - sink.kl_clean_intervened,
                control_row.delta_loss - sink.delta_loss,
                control_row.logit_l2 - sink.logit_l2,
                control_row.delta_correct_prob - sink.delta_correct_prob,
            )
            for band in bands:
                by_sequence.setdefault((seq, delta, head_kind, band, control), []).append(diffs)
    summaries = []
    for (seq, delta, head_kind, band, control), diffs in sorted(by_sequence.items()):
        n = len(diffs)
        mean_kl = float(sum(item[0] for item in diffs) / n)
        summaries.append(
            SequenceSummary(
                sequence_index=seq,
                delta=delta,
                head_kind=head_kind,
                layer_band=band,
                control_kind=control,
                mean_control_minus_sink_kl=mean_kl,
                mean_control_minus_sink_loss=float(sum(item[1] for item in diffs) / n),
                mean_control_minus_sink_logit_l2=float(sum(item[2] for item in diffs) / n),
                mean_control_minus_sink_correct_prob=float(sum(item[3] for item in diffs) / n),
                sink_less_disruptive=mean_kl > 0.0,
                num_pairs=n,
            )
        )
    return summaries


def bootstrap_sequence_summaries(
    summaries: Iterable[SequenceSummary],
    *,
    seed: int = 0,
    samples: int = 2000,
) -> list[BootstrapSummary]:
    grouped: dict[tuple[float, HeadKind, LayerBand, TargetKind], list[SequenceSummary]] = {}
    for row in summaries:
        grouped.setdefault((row.delta, row.head_kind, row.layer_band, row.control_kind), []).append(row)
    rng = random.Random(seed)
    out = []
    for (delta, head_kind, band, control), rows in sorted(grouped.items()):
        vals = [row.mean_control_minus_sink_kl for row in rows]
        losses = [row.mean_control_minus_sink_loss for row in rows]
        logit_l2s = [row.mean_control_minus_sink_logit_l2 for row in rows]
        correct_probs = [row.mean_control_minus_sink_correct_prob for row in rows]
        n = len(vals)
        if n == 0:
            continue
        means = []
        for _ in range(samples):
            draw = [vals[rng.randrange(n)] for _ in range(n)]
            means.append(sum(draw) / n)
        means.sort()
        lo = means[max(0, int(0.025 * samples) - 1)]
        hi = means[min(samples - 1, int(0.975 * samples))]
        out.append(
            BootstrapSummary(
                delta=delta,
                head_kind=head_kind,
                layer_band=band,
                control_kind=control,
                mean_paired_difference=float(sum(vals) / n),
                ci_low=float(lo),
                ci_high=float(hi),
                mean_control_minus_sink_loss=float(sum(losses) / n),
                mean_control_minus_sink_logit_l2=float(sum(logit_l2s) / n),
                mean_control_minus_sink_correct_prob=float(sum(correct_probs) / n),
                fraction_sink_less_disruptive=float(sum(v > 0.0 for v in vals) / n),
                num_sequences=n,
            )
        )
    return out

