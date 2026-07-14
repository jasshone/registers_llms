
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / 'outputs' / 'realtext_toy_shifted_bos_gradients' / 'wikitext103_train_expanded_3seed'


def setup_style() -> None:
    font_dir = Path(__file__).resolve().parent / 'fonts'
    font_path = font_dir / 'HelveticaNeueLight.otf'
    bold_path = font_dir / 'HelveticaNeueBold.otf'
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        if bold_path.exists():
            fm.fontManager.addfont(str(bold_path))
        plt.rcParams['font.family'] = fm.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams['font.weight'] = 200
        plt.rcParams['axes.labelweight'] = 200
        plt.rcParams['axes.titleweight'] = 200
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, val in list(row.items()):
            try:
                row[key] = float(val)
            except (TypeError, ValueError):
                pass
    return rows


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=11, width=1.2, length=4)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(1.2)


def variant_pos(name: str) -> int | None:
    if name.startswith('single_'):
        return int(name.split('_', 1)[1])
    return None


def err(row: dict[str, Any], key: str) -> float:
    return float(row.get(key + '_std', 0.0) or 0.0)


def val(row: dict[str, Any], key: str) -> float:
    return float(row.get(key + '_mean', float('nan')))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, default=DEFAULT_SRC)
    ap.add_argument('--stem', default='traintext_bos_position_gradients')
    args = ap.parse_args()
    src = args.src if args.src.is_absolute() else ROOT / args.src
    rows = read_rows(src / 'aggregate_gradient_summary.csv')
    singles = sorted([r for r in rows if str(r['variant']).startswith('single_')], key=lambda r: variant_pos(str(r['variant'])) or 0)
    controls = [r for r in rows if not str(r['variant']).startswith('single_')]
    if not rows:
        raise SystemExit('no rows')

    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.80, wspace=0.38)

    x = np.array([variant_pos(str(r['variant'])) for r in singles], dtype=float)
    pos0_attn = np.array([val(r, 'pos0_attention_mean') for r in singles])
    bos_attn = np.array([val(r, 'bos_attention_sum_mean') for r in singles])
    pos0_err = np.array([err(r, 'pos0_attention_mean') for r in singles])
    bos_err = np.array([err(r, 'bos_attention_sum_mean') for r in singles])
    axes[0].errorbar(x, pos0_attn, yerr=pos0_err, marker='o', linewidth=2.2, markersize=5, color='#b22222', label='absolute pos 0')
    axes[0].errorbar(x, bos_attn, yerr=bos_err, marker='o', linewidth=2.2, markersize=5, color='#1f5fbf', label='BOS position')
    axes[0].set_title('Attention Target', fontsize=18, pad=9)
    axes[0].set_xlabel('BOS training position', fontsize=13)
    axes[0].set_ylabel('Mean attention', fontsize=13)
    axes[0].set_xscale('symlog', linthresh=2)
    axes[0].set_xticks([0, 2, 4, 8, 16, 32, 64])
    axes[0].set_xticklabels(['0', '2', '4', '8', '16', '32', '64'])
    axes[0].legend(frameon=True, fancybox=False, edgecolor='#9ca3af', fontsize=10)
    axes[0].yaxis.grid(True, color='#6b7280', linewidth=1, alpha=0.14)
    clean_axis(axes[0])

    pos0_grad = np.array([val(r, 'pos0_qk_pressure_mean') for r in singles])
    bos_grad = np.array([val(r, 'bos_qk_pressure_max_position_mean') for r in singles])
    axes[1].errorbar(x, pos0_grad * 1e8, yerr=np.array([err(r, 'pos0_qk_pressure_mean') for r in singles]) * 1e8, marker='o', linewidth=2.2, markersize=5, color='#b22222', label='absolute pos 0')
    axes[1].errorbar(x, bos_grad * 1e8, yerr=np.array([err(r, 'bos_qk_pressure_max_position_mean') for r in singles]) * 1e8, marker='o', linewidth=2.2, markersize=5, color='#1f5fbf', label='BOS position')
    axes[1].axhline(0.0, color='#374151', linewidth=1.0, alpha=0.8)
    axes[1].set_title('QK Gradient Pressure', fontsize=18, pad=9)
    axes[1].set_xlabel('BOS training position', fontsize=13)
    axes[1].set_ylabel('pressure x 1e8', fontsize=13)
    axes[1].set_xscale('symlog', linthresh=2)
    axes[1].set_xticks([0, 2, 4, 8, 16, 32, 64])
    axes[1].set_xticklabels(['0', '2', '4', '8', '16', '32', '64'])
    axes[1].yaxis.grid(True, color='#6b7280', linewidth=1, alpha=0.14)
    clean_axis(axes[1])

    labels = [str(r['variant']).replace('duplicate_', 'dup ').replace('single_', 'pos ') for r in controls]
    if controls:
        xpos = np.arange(len(controls))
        width = 0.36
        axes[2].bar(xpos - width/2, [val(r, 'pos0_attention_mean') for r in controls], width=width, color='#b22222', label='absolute pos 0')
        axes[2].bar(xpos + width/2, [val(r, 'bos_attention_sum_mean') for r in controls], width=width, color='#1f5fbf', label='BOS total')
        axes[2].set_xticks(xpos)
        axes[2].set_xticklabels(labels, rotation=20, ha='right')
    axes[2].set_title('BOS Presence Controls', fontsize=18, pad=9)
    axes[2].set_ylabel('Mean attention', fontsize=13)
    axes[2].yaxis.grid(True, color='#6b7280', linewidth=1, alpha=0.14)
    clean_axis(axes[2])

    fig.suptitle('Train-Text BOS Position Controls', fontsize=22, y=0.94, fontweight=200)
    out = src / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{args.stem}.png', dpi=300)
    fig.savefig(out / f'{args.stem}.pdf')
    plt.close(fig)
    print((out / f'{args.stem}.png').relative_to(ROOT))
    print((out / f'{args.stem}.pdf').relative_to(ROOT))


if __name__ == '__main__':
    main()
