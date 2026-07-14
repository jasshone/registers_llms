from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/workspace/registers_llms/.hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/workspace/registers_llms/.hf_cache/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/registers_llms/.hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/registers_llms/.hf_cache/hub")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sink_neurons.artifacts import load_pt
from sink_neurons.env import configure_runtime
from sink_neurons.intervention import build_intervened_inputs
from sink_neurons.modeling import load_model


FINAL_OUT = ROOT / "outputs" / "final_split_bias_transfer"
OUT = ROOT / "outputs" / "pythia_dummy_to_sink_pre_intervention"
WINDOWS_PER_SPLIT = 64


@dataclass(frozen=True)
class ModelCfg:
    key: str
    label: str
    model_id: str
    params: float


PYTHIA_MODELS = [
    ModelCfg("pythia_70m", "70M", "EleutherAI/pythia-70m", 70e6),
    ModelCfg("pythia_160m", "160M", "EleutherAI/pythia-160m", 160e6),
    ModelCfg("pythia_410m", "410M", "EleutherAI/pythia-410m", 410e6),
    ModelCfg("pythia_1b", "1B", "EleutherAI/pythia-1b", 1e9),
    ModelCfg("pythia_1_4b", "1.4B", "EleutherAI/pythia-1.4b", 1.4e9),
    ModelCfg("pythia_2_8b", "2.8B", "EleutherAI/pythia-2.8b", 2.8e9),
    ModelCfg("pythia_6_9b", "6.9B", "EleutherAI/pythia-6.9b", 6.9e9),
    ModelCfg("pythia_12b", "12B", "EleutherAI/pythia-12b", 12e9),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pre-intervention zero-dummy hidden-state cosine similarity to sink-token hidden states for Pythia models."
    )
    parser.add_argument("--models", nargs="*", default=[cfg.key for cfg in PYTHIA_MODELS])
    parser.add_argument("--windows", type=int, default=16, help="Validation windows per model to use for recomputation.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--csv-only", action="store_true", help="Skip recomputation and only redraw from the CSV.")
    return parser.parse_args()


def setup_style() -> None:
    font_dir = Path(__file__).resolve().parent / "fonts"
    font_path = font_dir / "HelveticaNeueLight.otf"
    bold_path = font_dir / "HelveticaNeueBold.otf"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        if bold_path.exists():
            fm.fontManager.addfont(str(bold_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = font_name
        plt.rcParams["font.weight"] = 200
        plt.rcParams["axes.labelweight"] = 200
        plt.rcParams["axes.titleweight"] = 200

    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "DejaVu Sans"
    plt.rcParams["mathtext.it"] = "DejaVu Sans:italic"
    plt.rcParams["mathtext.cal"] = "cmsy10"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(axis="y", which="major", labelsize=15, width=1.4, length=5)
    ax.tick_params(axis="x", which="major", labelsize=15, width=1.4, length=5, pad=8)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.5)
    ax.yaxis.grid(True, color="gray", linewidth=1.0, alpha=0.16, zorder=0)


def cfg_by_key(key: str) -> ModelCfg:
    for cfg in PYTHIA_MODELS:
        if cfg.key == key:
            return cfg
    raise KeyError(f"Unknown Pythia model key: {key}")


def artifact_tensor(payload: dict[str, Any], *names: str) -> torch.Tensor:
    for name in names:
        value = payload.get(name)
        if isinstance(value, torch.Tensor):
            return value
        tensors = payload.get("tensors")
        if isinstance(tensors, dict) and isinstance(tensors.get(name), torch.Tensor):
            return tensors[name]
    raise ValueError(f"None of these tensors are present: {names}")


def load_validation_inputs(model_key: str, windows: int) -> tuple[torch.Tensor, torch.Tensor]:
    model_dir = FINAL_OUT / model_key
    windows_path = model_dir / "windows_192x1024.pt"
    mask_path = model_dir / "val_sink_mask.pt"
    all_windows = artifact_tensor(load_pt(windows_path), "windows").long()
    sink_mask = artifact_tensor(load_pt(mask_path), "sink_mask", "mask").bool()
    val = all_windows[WINDOWS_PER_SPLIT : 2 * WINDOWS_PER_SPLIT]
    if val.shape[:2] != sink_mask.shape:
        raise ValueError(f"Window/mask shape mismatch for {model_key}: {tuple(val.shape)} vs {tuple(sink_mask.shape)}")
    return val[:windows], sink_mask[:windows]


def mean_dummy_to_sink_cosine(clean_h: torch.Tensor, dummy_h: torch.Tensor, sink_mask: torch.Tensor) -> tuple[float, int, int]:
    dummy_vec = dummy_h[:, 1, :].float().cpu()
    clean = clean_h.float().cpu()
    mask = sink_mask.cpu().bool()
    example_cosines: list[torch.Tensor] = []
    sink_tokens = 0
    for i in range(mask.shape[0]):
        if not bool(mask[i].any()):
            continue
        sink_vec = clean[i, mask[i], :].mean(dim=0)
        example_cosines.append(torch.nn.functional.cosine_similarity(dummy_vec[i], sink_vec, dim=0))
        sink_tokens += int(mask[i].sum().item())
    if not example_cosines:
        return float("nan"), 0, 0
    return float(torch.stack(example_cosines).mean().item()), len(example_cosines), sink_tokens


@torch.no_grad()
def compute_model_rows(cfg: ModelCfg, windows: int, batch_size: int) -> list[dict[str, Any]]:
    input_ids, sink_mask = load_validation_inputs(cfg.key, windows)
    model = load_model(cfg.model_id, eager_attention=True)
    rows: list[dict[str, Any]] = []
    sums: list[float] | None = None
    example_counts: list[int] | None = None
    token_counts: list[int] | None = None

    for start in range(0, input_ids.shape[0], batch_size):
        end = min(start + batch_size, input_ids.shape[0])
        batch_ids = input_ids[start:end].to(model.device)
        batch_mask = sink_mask[start:end]
        clean_out = model(
            input_ids=batch_ids,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        embeds, attention_mask = build_intervened_inputs(
            model,
            batch_ids,
            dummy_init="zero",
            num_dummy_tokens=1,
            dummy_position="after_bos",
        )
        dummy_out = model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )

        if sums is None:
            depth = len(clean_out.hidden_states)
            sums = [0.0] * depth
            example_counts = [0] * depth
            token_counts = [0] * depth

        assert sums is not None and example_counts is not None and token_counts is not None
        for hs_idx, (clean_h, dummy_h) in enumerate(zip(clean_out.hidden_states, dummy_out.hidden_states)):
            value, examples, sink_tokens = mean_dummy_to_sink_cosine(clean_h.detach(), dummy_h.detach(), batch_mask)
            if examples:
                sums[hs_idx] += value * examples
                example_counts[hs_idx] += examples
                token_counts[hs_idx] += sink_tokens

        del clean_out, dummy_out, embeds, attention_mask

    assert sums is not None and example_counts is not None and token_counts is not None
    for hs_idx, total in enumerate(sums):
        examples = example_counts[hs_idx]
        rows.append(
            {
                "model_key": cfg.key,
                "display_label": cfg.label,
                "model_id": cfg.model_id,
                "params": cfg.params,
                "hidden_state_index": hs_idx,
                "block_index": hs_idx - 1,
                "is_embedding": hs_idx == 0,
                "dummy_to_sink_cosine_before_intervention": total / examples if examples else float("nan"),
                "sink_examples": examples,
                "sink_tokens": token_counts[hs_idx],
                "windows": int(input_ids.shape[0]),
                "split": "val",
                "dummy_init": "zero",
                "dummy_position": "after_bos",
                "sink_mask_source": str((FINAL_OUT / cfg.key / "val_sink_mask.pt").relative_to(ROOT)),
                "window_source": str((FINAL_OUT / cfg.key / "windows_192x1024.pt").relative_to(ROOT)),
            }
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    models = [cfg.key for cfg in PYTHIA_MODELS if any(r["model_key"] == cfg.key for r in rows)]
    colors = dict(zip(models, plt.cm.viridis(np.linspace(0.04, 0.88, max(2, len(models))))))

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.9), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.24, top=0.84)

    for model_key in models:
        model_rows = sorted(
            [r for r in rows if r["model_key"] == model_key],
            key=lambda row: int(row["hidden_state_index"]),
        )
        hs_idx = np.array([int(r["hidden_state_index"]) for r in model_rows], dtype=float)
        max_idx = float(hs_idx.max()) if hs_idx.size else 1.0
        x = 100.0 * hs_idx / max_idx if max_idx else hs_idx
        y = np.array([float(r["dummy_to_sink_cosine_before_intervention"]) for r in model_rows])
        label = model_rows[0]["display_label"]
        ax.plot(
            x,
            y,
            color=colors[model_key],
            linewidth=2.2,
            marker="o",
            markersize=3.4,
            label=label,
            zorder=4,
        )

    ax.axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.55, zorder=2)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("Percent Through Model", fontsize=21, fontweight=200, labelpad=10)
    ax.set_ylabel("Cosine Similarity", fontsize=21, fontweight=200)
    ax.set_title("Dummy to Sink Before Intervention", fontsize=25, pad=10, fontweight=200)
    ax.set_xlim(0.0, 100.0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    clean_axis(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.33),
        ncol=min(4, len(models)),
        frameon=True,
        fancybox=False,
        edgecolor="#9ca3af",
        facecolor="white",
        framealpha=0.96,
        fontsize=14,
        handlelength=1.7,
        columnspacing=1.1,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "pythia_dummy_to_sink_pre_intervention.png", dpi=300)
    fig.savefig(out_dir / "pythia_dummy_to_sink_pre_intervention.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.out.is_absolute():
        args.out = ROOT / args.out
    setup_style()
    csv_path = args.out / "pythia_dummy_to_sink_pre_intervention.csv"

    if args.csv_only:
        rows: list[dict[str, Any]] = read_rows(csv_path)
    else:
        configure_runtime()
        rows = []
        for model_key in args.models:
            cfg = cfg_by_key(model_key)
            print(f"[compute] {cfg.key}", flush=True)
            rows.extend(compute_model_rows(cfg, args.windows, args.batch_size))
        write_rows(csv_path, rows)

    plot(rows, args.out)
    print(csv_path.relative_to(ROOT))
    print((args.out / "pythia_dummy_to_sink_pre_intervention.png").relative_to(ROOT))
    print((args.out / "pythia_dummy_to_sink_pre_intervention.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
