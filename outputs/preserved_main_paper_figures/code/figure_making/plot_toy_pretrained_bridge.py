
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / 'outputs' / 'toy_pretrained_bridge' / 'wikitext_pythia_tokenizer_10k_scale10_test512'
CSV = RUN_DIR / 'toy_relocation_bridge_metrics.csv'
OUT = RUN_DIR / 'figures'


def setup_style() -> None:
    font_dir = Path(__file__).resolve().parent / 'fonts'
    font_path = font_dir / 'HelveticaNeueLight.otf'
    bold_path = font_dir / 'HelveticaNeueBold.otf'
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        if bold_path.exists():
            fm.fontManager.addfont(str(bold_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams['font.family'] = font_name
        plt.rcParams['font.weight'] = 200
        plt.rcParams['axes.labelweight'] = 200
        plt.rcParams['axes.titleweight'] = 200
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


def clean_axis(ax: plt.Axes, *, y_grid: bool = True) -> None:
    ax.grid(False)
    ax.tick_params(axis='y', which='major', labelsize=13, width=1.3, length=5)
    ax.tick_params(axis='x', which='major', labelsize=13, width=1.3, length=5, pad=7)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(1.4)
    if y_grid:
        ax.yaxis.grid(True, color='gray', linewidth=1.0, alpha=0.15, zorder=0)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        cur: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or value == '':
                cur[key] = value
                continue
            try:
                cur[key] = float(value)
            except ValueError:
                cur[key] = value
        out.append(cur)
    return out


def arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def main() -> None:
    setup_style()
    rows = read_rows(CSV)
    x = arr(rows, 'step') / 1000.0
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.22, wspace=0.32)

    colors = {
        'sink': '#3b528b',
        'selected': '#21918c',
        'random': '#d1495b',
        'dummy': '#8a8f98',
        'score': '#5ec962',
    }

    ax = axes[0]
    ax.plot(x, arr(rows, 'clean_max_head_bos_attention'), color=colors['sink'], linewidth=2.5, marker='o', markersize=4.3, label='BOS attention')
    ax.set_title('Toy Sink Emergence', fontsize=18, pad=8, fontweight=200)
    ax.set_xlabel('Step (k)', fontsize=14, labelpad=6)
    ax.set_ylabel('Max-head BOS attention', fontsize=13, labelpad=6)
    ax.set_ylim(0.0, max(arr(rows, 'clean_max_head_bos_attention')) * 1.18)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    clean_axis(ax)
    ax2 = ax.twinx()
    ax2.plot(x, arr(rows, 'selected_score_mass'), color=colors['score'], linewidth=2.2, marker='s', markersize=3.8, label='Score mass')
    ax2.set_ylabel('Selected score mass', fontsize=13, labelpad=7)
    ax2.tick_params(axis='y', labelsize=13, width=1.3, length=5)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_linewidth(1.3)
    ax2.grid(False)

    ax = axes[1]
    ax.axhline(0.0, color='#6b7280', linewidth=1.0, alpha=0.65, zorder=1)
    ax.plot(x, arr(rows, 'selected_dummy_minus_bos_mean'), color=colors['selected'], linewidth=2.5, marker='o', markersize=4.3, label='Selected')
    ax.plot(x, arr(rows, 'random_dummy_minus_bos_mean'), color=colors['random'], linewidth=2.2, marker='o', markersize=3.8, label='Random')
    ax.plot(x, arr(rows, 'dummy_only_dummy_minus_bos_mean'), color=colors['dummy'], linewidth=2.0, linestyle='--', marker='o', markersize=3.5, label='Dummy only')
    ax.set_title('Relocation Sufficiency', fontsize=18, pad=8, fontweight=200)
    ax.set_xlabel('Step (k)', fontsize=14, labelpad=6)
    ax.set_ylabel('Dummy - BOS attention', fontsize=13, labelpad=6)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    clean_axis(ax)
    ax.legend(loc='upper right', frameon=False, fontsize=11, handlelength=1.5)

    ax = axes[2]
    ax.axhline(1.0, color='#6b7280', linewidth=1.0, alpha=0.65, zorder=1)
    ax.plot(x, arr(rows, 'zero_max_head_bos_retention'), color=colors['selected'], linewidth=2.5, marker='o', markersize=4.3, label='Selected')
    ax.plot(x, arr(rows, 'random_zero_max_head_bos_retention'), color=colors['random'], linewidth=2.2, marker='o', markersize=3.8, label='Random')
    ax.set_title('Ablation Necessity', fontsize=18, pad=8, fontweight=200)
    ax.set_xlabel('Step (k)', fontsize=14, labelpad=6)
    ax.set_ylabel('BOS attention retained', fontsize=13, labelpad=6)
    ax.set_ylim(0.64, 1.05)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    clean_axis(ax)
    ax.legend(loc='lower left', frameon=False, fontsize=11, handlelength=1.5)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'toy_pretrained_relocation_bridge.png', dpi=300)
    fig.savefig(OUT / 'toy_pretrained_relocation_bridge.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
