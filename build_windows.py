from __future__ import annotations

import argparse
from pathlib import Path

from sink_neurons.env import configure_runtime
from sink_neurons.modeling import DEFAULT_MODEL_ID, load_tokenizer
from sink_neurons.windows import WindowBuildConfig, build_windows_tensor, resolve_start_token, save_windows_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed-length WikiText windows with explicit BOS.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/windows/wikitext103_raw_windows.pt"))
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_runtime()
    args = parse_args()
    tokenizer = load_tokenizer(args.model_id)
    start_token = resolve_start_token(tokenizer)
    windows, stream_length = build_windows_tensor(
        tokenizer,
        split=args.split,
        window_length=args.window_length,
        stride=args.stride,
        max_windows=args.max_windows,
        show_progress=not args.no_progress,
    )
    config = WindowBuildConfig(
        model_id=args.model_id,
        split=args.split,
        window_length=args.window_length,
        stride=args.stride,
        max_windows=args.max_windows,
    )
    save_windows_artifact(
        args.output,
        windows=windows,
        config=config,
        tokenized_stream_length=stream_length,
        bos_token_id=start_token.token_id,
        start_token_source=start_token.source,
    )
    print(f"Saved {windows.shape[0]} windows to {args.output}")


if __name__ == "__main__":
    main()
