
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from run_realtext_toy_sink_training import (
    RealTextTinyGPT,
    RealTextToyConfig,
    attention_and_qk_summary,
    encode_text,
    read_tiny_shakespeare,
    read_wikitext,
    split_tokens,
    write_csv,
)


def make_batch_at_bos_position(
    tokens: torch.Tensor,
    cfg: RealTextToyConfig,
    batch_size: int,
    device: torch.device,
    *,
    bos_position: int,
    fixed: bool = False,
) -> torch.Tensor:
    if not (0 <= bos_position < cfg.seq_len):
        raise ValueError(f"bos_position must be in [0, {cfg.seq_len - 1}], got {bos_position}")
    content_len = cfg.seq_len - 1
    max_start = int(tokens.numel()) - content_len
    if max_start <= 0:
        raise ValueError("token stream too short for requested sequence length")
    if fixed:
        starts = [0] if batch_size == 1 else torch.linspace(0, max_start - 1, steps=batch_size).long().tolist()
    else:
        starts = torch.randint(0, max_start, (batch_size,)).tolist()
    rows = []
    for start in starts:
        content = tokens[int(start) : int(start) + content_len]
        row = torch.empty(cfg.seq_len, dtype=torch.long)
        row[:bos_position] = content[:bos_position]
        row[bos_position] = cfg.bos_token_id
        row[bos_position + 1 :] = content[bos_position:]
        rows.append(row)
    return torch.stack(rows, dim=0).to(device)


def _mean_attention_to_key(attentions: list[torch.Tensor], key_pos: int, *, query_start: int | None = None) -> tuple[float, float, str]:
    by_layer = []
    by_head = []
    for attn in attentions:
        # attn: batch, head, query, key. Only future queries can attend to key_pos.
        start = key_pos + 1 if query_start is None else max(query_start, key_pos + 1)
        if start >= attn.shape[-2]:
            layer_vals = torch.full((attn.shape[1],), float("nan"), device=attn.device)
        else:
            layer_vals = attn[:, :, start:, key_pos].float().mean(dim=(0, 2))
        by_head.append(layer_vals.detach().cpu())
        by_layer.append(layer_vals.mean().detach().cpu())
    by_layer_t = torch.stack(by_layer)
    by_head_t = torch.stack(by_head)
    finite = by_head_t[torch.isfinite(by_head_t)]
    max_head = float(finite.max()) if finite.numel() else float("nan")
    return float(torch.nanmean(by_layer_t)), max_head, " ".join(f"{float(x):.6g}" for x in by_layer_t.tolist())


def _qk_margin_to_key(qk_logits: list[torch.Tensor], key_pos: int) -> tuple[float, float]:
    margins = []
    for logits in qk_logits:
        vals = []
        layer_logits = logits.float()
        for qpos in range(max(1, key_pos + 1), layer_logits.shape[-2]):
            key_logit = layer_logits[:, :, qpos, key_pos]
            other_parts = []
            if key_pos > 0:
                other_parts.append(layer_logits[:, :, qpos, :key_pos])
            if key_pos + 1 <= qpos:
                other_parts.append(layer_logits[:, :, qpos, key_pos + 1 : qpos + 1])
            if not other_parts:
                continue
            other = torch.cat(other_parts, dim=-1).max(dim=-1).values
            vals.append(key_logit - other)
        if vals:
            margins.append(torch.stack(vals, dim=-1).mean(dim=(0, 2)).detach().cpu())
    if not margins:
        return float("nan"), float("nan")
    by_head = torch.stack(margins)
    return float(by_head.mean()), float(by_head.max())


def train_position(args: argparse.Namespace, bos_position: int, base_splits: dict[str, torch.Tensor], cfg: RealTextToyConfig, device: torch.device, run_root: Path) -> dict[str, Any]:
    torch.manual_seed(args.seed + bos_position * 1009)
    random.seed(args.seed + bos_position * 1009)
    model = RealTextTinyGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    eval_batch = make_batch_at_bos_position(base_splits[args.eval_split], cfg, args.eval_batch_size, device, bos_position=bos_position, fixed=True)
    rows = []
    out_dir = run_root / f"bos_pos_{bos_position}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                out = model(eval_batch, output_attentions=True, output_qk_logits=True)
                loss = F.cross_entropy(out["logits"][:, :-1, :].contiguous().view(-1, cfg.vocab_size), eval_batch[:, 1:].contiguous().view(-1))
            pos0_mean, pos0_max, pos0_layers = _mean_attention_to_key(out["attentions"], 0)
            bos_mean, bos_max, bos_layers = _mean_attention_to_key(out["attentions"], bos_position)
            next_mean, next_max, next_layers = _mean_attention_to_key(out["attentions"], min(bos_position + 1, cfg.seq_len - 1)) if bos_position + 1 < cfg.seq_len - 1 else (float("nan"), float("nan"), "")
            bos_qk_mean, bos_qk_max = _qk_margin_to_key(out["qk_logits"], bos_position)
            pos0_qk_mean, pos0_qk_max = _qk_margin_to_key(out["qk_logits"], 0)
            row = {
                "bos_position": bos_position,
                "step": step,
                "loss": float(loss.cpu()),
                "ppl": float(torch.exp(loss).cpu()),
                "pos0_attention_mean": pos0_mean,
                "pos0_attention_max_head": pos0_max,
                "bos_position_attention_mean": bos_mean,
                "bos_position_attention_max_head": bos_max,
                "next_position_attention_mean": next_mean,
                "next_position_attention_max_head": next_max,
                "bos_minus_pos0_attention": bos_mean - pos0_mean,
                "bos_minus_next_attention": bos_mean - next_mean if math.isfinite(next_mean) else float("nan"),
                "bos_qk_margin_mean": bos_qk_mean,
                "bos_qk_margin_max_head": bos_qk_max,
                "pos0_qk_margin_mean": pos0_qk_mean,
                "pos0_qk_margin_max_head": pos0_qk_max,
                "pos0_attention_by_layer": pos0_layers,
                "bos_position_attention_by_layer": bos_layers,
                "next_position_attention_by_layer": next_layers,
            }
            rows.append(row)
            write_csv(out_dir / "metrics.csv", rows)
            if args.save_checkpoints:
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step, "bos_position": bos_position}, out_dir / f"checkpoint_step{step}.pt")
            print(
                f"[pos={bos_position}] step={step} loss={row['loss']:.4f} pos0={pos0_mean:.4f} bospos={bos_mean:.4f} next={next_mean:.4f} bos-pos0={row['bos_minus_pos0_attention']:+.4f}",
                flush=True,
            )
            model.train()
        if step == args.steps:
            break
        batch = make_batch_at_bos_position(base_splits["train"], cfg, args.batch_size, device, bos_position=bos_position, fixed=False)
        out = model(batch, labels=batch)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return rows[-1]


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Toy Training BOS Position Sweep",
        "",
        "Each model is trained from scratch with the BOS token placed at a fixed sequence position. Metrics compare learned attention to absolute position 0 against attention to the trained BOS position.",
        "",
        "| BOS train pos | final loss | final PPL | attn to pos0 | attn to BOS pos | attn to next pos | BOS-pos0 | BOS-next |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {int(r['bos_position'])} | {r['loss']:.4f} | {r['ppl']:.3g} | {r['pos0_attention_mean']:.4f} | "
            f"{r['bos_position_attention_mean']:.4f} | {r['next_position_attention_mean']:.4f} | "
            f"{r['bos_minus_pos0_attention']:+.4f} | {r['bos_minus_next_attention']:+.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train toy LMs with BOS fixed at nonzero sequence positions.")
    p.add_argument("--positions", nargs="*", type=int, default=[0, 2, 4, 8, 16, 32, 64])
    p.add_argument("--dataset", choices=["tiny_shakespeare", "wikitext"], default="wikitext")
    p.add_argument("--encoding", choices=["byte", "hf_tokenizer"], default="hf_tokenizer")
    p.add_argument("--tokenizer-model-id", default="EleutherAI/pythia-70m")
    p.add_argument("--data-path", type=Path, default=Path("outputs/realtext_toy_data/tiny_shakespeare.txt"))
    p.add_argument("--token-windows", type=Path, default=None, help="Optional cached token window .pt file; uses windows[:, 1:] as corpus tokens.")
    p.add_argument("--cached-vocab-size", type=int, default=50277)
    p.add_argument("--cached-bos-token-id", type=int, default=0)
    p.add_argument("--wikitext-train-split", default="train")
    p.add_argument("--max-chars", type=int, default=10_000_000)
    p.add_argument("--out", type=Path, default=Path("outputs/realtext_toy_bos_train_position/wikitext_pythia_tokenizer_10k"))
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--eval-split", choices=["val", "test"], default="val")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-mlp", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.token_windows is not None:
        payload = torch.load(args.token_windows, map_location="cpu")
        windows = payload["windows"] if isinstance(payload, dict) else payload
        if windows.ndim != 2 or windows.shape[1] < 2:
            raise ValueError(f"expected 2D token windows with length >=2, got {tuple(windows.shape)}")
        tokens = windows[:, 1:].contiguous().view(-1).long()
        if args.max_chars is not None:
            tokens = tokens[: args.max_chars]
        vocab_size = args.cached_vocab_size
        bos_token_id = args.cached_bos_token_id
        encoding_source = f"cached_token_windows:{args.token_windows}"
    else:
        if args.dataset == "tiny_shakespeare":
            text = read_tiny_shakespeare(args.data_path)
        else:
            text = read_wikitext(args.wikitext_train_split, args.max_chars)
        if args.max_chars is not None:
            text = text[: args.max_chars]
        tokens, vocab_size, bos_token_id, encoding_source = encode_text(text, encoding=args.encoding, tokenizer_model_id=args.tokenizer_model_id)
    splits = split_tokens(tokens, train_frac=0.9, val_frac=0.05)
    cfg = RealTextToyConfig(vocab_size=vocab_size, bos_token_id=bos_token_id, seq_len=args.seq_len, n_layer=args.n_layer, n_head=args.n_head, d_model=args.d_model, d_mlp=args.d_mlp)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({"args": vars(args), "model": asdict(cfg), "encoding_source": encoding_source, "num_tokens": int(tokens.numel()), "split_tokens": {k: int(v.numel()) for k, v in splits.items()}}, indent=2, default=str) + "\n")
    final_rows = []
    for pos in args.positions:
        final_rows.append(train_position(args, pos, splits, cfg, device, args.out))
        write_csv(args.out / "bos_train_position_summary.csv", final_rows)
        write_summary_md(args.out / "bos_train_position_summary.md", final_rows)
    print(args.out / "bos_train_position_summary.md")


if __name__ == "__main__":
    main()
