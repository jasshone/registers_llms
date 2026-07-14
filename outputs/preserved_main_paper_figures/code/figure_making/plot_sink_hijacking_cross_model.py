from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "sink_hijacking_attack" / "cross_model_focused_attack_summary.csv"
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
        plt.rcParams["font.weight"] = 200
        plt.rcParams["axes.labelweight"] = 200
        plt.rcParams["axes.titleweight"] = 200
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(axis="y", which="major", labelsize=15, width=1.4, length=5)
    ax.tick_params(axis="x", which="major", labelsize=15, width=1.4, length=0, pad=8)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.5)
    ax.yaxis.grid(True, color="gray", linewidth=1.0, alpha=0.16, zorder=0)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in list(r.items()):
            try: r[k]=float(v)
            except (TypeError,ValueError): pass
    return rows


def main() -> None:
    setup_style()
    rows=[r for r in read_rows(SRC) if r['position']=='after_bos']
    labels=[r['label'] for r in rows]
    x=np.arange(len(labels))
    width=0.34
    selected="#d1495b"
    random="#21918c"
    fig,axes=plt.subplots(1,3,figsize=(16.0,5.5),constrained_layout=False)
    fig.subplots_adjust(left=0.09,right=0.97,bottom=0.25,top=0.82,wspace=0.34)
    panels=[
        (axes[0],'Dummy Sink Mass','Mean Attention',[r['window_dummy_attention_selected'] for r in rows],[r['window_dummy_attention_random_mean'] for r in rows],None),
        (axes[1],'Perplexity Change','PPL Ratio',[r['ppl_ratio_selected'] for r in rows],[r['ppl_ratio_random_mean'] for r in rows],1.0),
        (axes[2],'Local-Fact Margin','Margin Change',[r['margin_delta_selected'] for r in rows],[r['margin_delta_random_mean'] for r in rows],0.0),
    ]
    for ax,title,ylabel,sel_vals,rnd_vals,baseline in panels:
        ax.bar(x-width/2,rnd_vals,width,color=random,edgecolor='white',linewidth=0.9,label='Random control',zorder=4)
        ax.bar(x+width/2,sel_vals,width,color=selected,edgecolor='white',linewidth=0.9,label='Selected neurons',zorder=4)
        if baseline is not None:
            ax.axhline(baseline,color='#6b7280',linewidth=1.0,alpha=0.7,zorder=2)
        ax.set_title(title,fontsize=23,pad=10,fontweight=200)
        ax.set_ylabel(ylabel,fontsize=19,fontweight=200)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        clean_axis(ax)
    handles,leg_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,leg_labels,loc='lower center',bbox_to_anchor=(0.5,0.04),ncol=2,frameon=True,fancybox=False,edgecolor='#9ca3af',facecolor='white',framealpha=0.96,fontsize=14)
    fig.suptitle('Focused Sink Hijacking Across Pythia Models',fontsize=26,fontweight=200,y=0.965)
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/'sink_hijacking_cross_model.png',dpi=300)
    fig.savefig(OUT/'sink_hijacking_cross_model.pdf')
    plt.close(fig)
    print((OUT/'sink_hijacking_cross_model.png').relative_to(ROOT))
    print((OUT/'sink_hijacking_cross_model.pdf').relative_to(ROOT))

if __name__=='__main__':
    main()
