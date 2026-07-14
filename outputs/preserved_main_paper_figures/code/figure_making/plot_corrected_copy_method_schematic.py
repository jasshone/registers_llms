from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "sink_hijacking_attack" / "figures"


def setup_style() -> None:
    font_dir = Path(__file__).resolve().parent / "fonts"
    font_path = font_dir / "HelveticaNeueLight.otf"
    bold_path = font_dir / "HelveticaNeueBold.otf"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        if bold_path.exists():
            fm.fontManager.addfont(str(bold_path))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.weight"] = 300
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def rounded(ax: plt.Axes, xy: tuple[float, float], w: float, h: float, *, fc: str, ec: str = "#27313f", lw: float = 1.4, radius: float = 0.04) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = "#2f3a47", lw: float = 1.8, rad: float = 0.0) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)


def label(ax: plt.Axes, x: float, y: float, text: str, *, size: int = 12, weight: int = 200, color: str = "#17202a", ha: str = "center") -> None:
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, fontweight=weight, color=color, linespacing=1.18)


def token_row(ax: plt.Axes, y: float, *, with_dummy: bool) -> None:
    x0 = 0.08
    widths = [0.075, 0.085, 0.085, 0.085, 0.12]
    names = ["BOS", "tok 1", "tok 2", "...", "query"]
    colors = ["#edf7f5", "#f6f7f9", "#f6f7f9", "#f6f7f9", "#f6f7f9"]
    x = x0
    for idx, (w, name, color) in enumerate(zip(widths, names, colors)):
        rounded(ax, (x, y), w, 0.06, fc=color, ec="#8b98a7", lw=1.0, radius=0.012)
        label(ax, x + w / 2, y + 0.03, name, size=9)
        x += w + 0.012
        if idx == 0 and with_dummy:
            for d in range(3):
                rounded(ax, (x, y), 0.055, 0.06, fc="#fff7ed", ec="#c56b30", lw=1.1, radius=0.012)
                label(ax, x + 0.0275, y + 0.03, f"slot\n{d+1}", size=8, color="#7c2d12")
                x += 0.064


def activation_column(ax: plt.Axes, x: float, y: float, *, title: str, fill: str, active_rows: tuple[int, ...]) -> None:
    rounded(ax, (x, y), 0.17, 0.235, fc="#ffffff", ec="#8b98a7", lw=1.2, radius=0.02)
    label(ax, x + 0.085, y + 0.212, title, size=10, color="#26313d")
    for row in range(4):
        yy = y + 0.032 + row * 0.038
        ax.add_patch(Rectangle((x + 0.026, yy), 0.118, 0.021, facecolor="#edf0f4", edgecolor="none"))
        if row in active_rows:
            ax.add_patch(Rectangle((x + 0.026, yy), 0.085 + 0.008 * row, 0.021, facecolor=fill, edgecolor="none", alpha=0.9))


def main() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.subplots_adjust(left=0.035, right=0.965, top=0.925, bottom=0.075)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    label(ax, 0.5, 0.955, "Corrected Copy Intervention", size=25, color="#17202a")
    label(ax, 0.5, 0.912, "Copy selected MLP activations into inserted slots while preserving the source tokens", size=13, color="#5f6b7a")

    # Step 1.
    rounded(ax, (0.045, 0.665), 0.265, 0.18, fc="#ffffff", ec="#c9d1dc", lw=1.3)
    label(ax, 0.177, 0.815, "1. Insert slots", size=14, color="#0f766e")
    label(ax, 0.177, 0.775, "Zero dummy/register slots\nare placed after BOS", size=11, color="#344253")
    token_row(ax, 0.69, with_dummy=True)

    # Step 2.
    rounded(ax, (0.365, 0.665), 0.27, 0.18, fc="#ffffff", ec="#c9d1dc", lw=1.3)
    label(ax, 0.5, 0.815, "2. Read selected neurons", size=14, color="#0f766e")
    label(ax, 0.5, 0.775, "For each selected MLP neuron,\ncompute max activation over\nnon-dummy positions", size=11, color="#344253")
    activation_column(ax, 0.413, 0.68, title="non-dummy activations", fill="#d1495b", active_rows=(1, 2, 3))

    # Step 3.
    rounded(ax, (0.69, 0.665), 0.265, 0.18, fc="#ffffff", ec="#c9d1dc", lw=1.3)
    label(ax, 0.822, 0.815, "3. Copy, do not move", size=14, color="#0f766e")
    label(ax, 0.822, 0.775, "Assigned dummy slot gets\nscale * max activation", size=11, color="#344253")
    token_row(ax, 0.69, with_dummy=True)
    for x in (0.17, 0.234, 0.298):
        ax.add_patch(Rectangle((x, 0.692), 0.035, 0.056, facecolor="#d1495b", alpha=0.24, edgecolor="none"))

    arrow(ax, (0.315, 0.755), (0.36, 0.755))
    arrow(ax, (0.64, 0.755), (0.685, 0.755))

    # Preserve original vs old relocation.
    rounded(ax, (0.08, 0.365), 0.385, 0.18, fc="#eefaf7", ec="#9bd4ca", lw=1.3)
    label(ax, 0.272, 0.505, "Corrected copy-mode", size=15, color="#064e48")
    label(ax, 0.272, 0.458, "Original selected-neuron activations stay\nat their source token positions", size=11, color="#344253")
    label(ax, 0.272, 0.405, "This removes the source-ablation confound", size=11, color="#064e48")

    rounded(ax, (0.535, 0.365), 0.385, 0.18, fc="#fff7ed", ec="#f1b27c", lw=1.3)
    label(ax, 0.727, 0.505, "Older move-and-ablate", size=15, color="#7c2d12")
    label(ax, 0.727, 0.458, "Source activations may be reduced\nor removed while dummy slots gain mass", size=11, color="#344253")
    label(ax, 0.727, 0.405, "Use only as a confound check", size=11, color="#7c2d12")

    arrow(ax, (0.5, 0.665), (0.295, 0.548), color="#0f766e", rad=0.16)
    arrow(ax, (0.5, 0.665), (0.705, 0.548), color="#7c2d12", rad=-0.16)

    # Controls and metrics.
    rounded(ax, (0.08, 0.115), 0.84, 0.16, fc="#ffffff", ec="#c9d1dc", lw=1.3)
    label(ax, 0.19, 0.235, "Controls", size=14, color="#26313d")
    label(ax, 0.19, 0.185, "dummy-only slots\nlayer-matched random copy", size=10.5, color="#344253")
    label(ax, 0.49, 0.235, "Readouts", size=14, color="#26313d")
    label(ax, 0.49, 0.185, "inserted-slot attention\nPPL ratio, margin, flips", size=10.5, color="#344253")
    label(ax, 0.79, 0.235, "Supported wording", size=14, color="#26313d")
    label(ax, 0.79, 0.185, "attention denial plus\nfallback-prior amplification", size=10.5, color="#344253")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "corrected_copy_method_schematic.png", dpi=300)
    fig.savefig(OUT / "corrected_copy_method_schematic.pdf")
    plt.close(fig)
    print((OUT / "corrected_copy_method_schematic.png").relative_to(ROOT))
    print((OUT / "corrected_copy_method_schematic.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
