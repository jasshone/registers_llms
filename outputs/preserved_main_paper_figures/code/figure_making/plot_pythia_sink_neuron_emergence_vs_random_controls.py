from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "pythia_random_control_checkpoint_timing" / "pythia_selected_vs_matched_random_control_separate_scale.png"
OUT = ROOT / "outputs" / "pythia_random_control_checkpoint_timing"
OUT_PNG = OUT / "pythia_sink_neuron_emergence_vs_random_controls.png"
OUT_PDF = OUT / "pythia_sink_neuron_emergence_vs_random_controls.pdf"


def save_padded_image(src: Path, out_png: Path, out_pdf: Path, pad: int = 90) -> None:
    img = Image.open(src).convert("RGB")
    canvas = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad), "white")
    canvas.paste(img, (pad, pad))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    fig, ax = plt.subplots(figsize=(canvas.width / 300, canvas.height / 300), dpi=300)
    ax.imshow(canvas)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def main() -> None:
    save_padded_image(SRC, OUT_PNG, OUT_PDF)
    print(OUT_PNG.relative_to(ROOT))
    print(OUT_PDF.relative_to(ROOT))


if __name__ == "__main__":
    main()
