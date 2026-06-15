from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B"


def get_hf_token(explicit_token: str | None = None) -> str | None:
    return explicit_token or os.environ.get("HF_TOKEN")


def _force_eager_attention(model: PreTrainedModel) -> None:
    # Some Transformers/Accelerate codepaths consult either the model config,
    # nested text config, or the model attribute directly.
    if hasattr(model, "config"):
        setattr(model.config, "_attn_implementation", "eager")
        if hasattr(model.config, "attn_implementation"):
            setattr(model.config, "attn_implementation", "eager")
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            setattr(text_config, "_attn_implementation", "eager")
            if hasattr(text_config, "attn_implementation"):
                setattr(text_config, "attn_implementation", "eager")
    setattr(model, "_attn_implementation", "eager")


def load_tokenizer(model_id: str = DEFAULT_MODEL_ID, token: str | None = None) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=get_hf_token(token), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    token: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    eager_attention: bool = False,
) -> PreTrainedModel:
    kwargs: dict[str, Any] = {
        "token": get_hf_token(token),
        "device_map": device_map,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if eager_attention:
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    if eager_attention:
        _force_eager_attention(model)
    return model


def resolve_device(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


def get_embed_tokens(model: PreTrainedModel) -> torch.nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "embed_tokens"):
        return model.model.language_model.embed_tokens
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "embed_tokens"):
        return model.model.decoder.embed_tokens
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "embed_in"):
        return model.gpt_neox.embed_in
    raise AttributeError("Unsupported model architecture: cannot find token embedding module")


def get_transformer_layers(model: PreTrainedModel) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise AttributeError("Unsupported model architecture: cannot find transformer layers")


def get_mlp_intermediate_size(model: PreTrainedModel) -> int:
    if hasattr(model.config, "intermediate_size"):
        return int(model.config.intermediate_size)
    if getattr(model.config, "n_inner", None) is not None:
        return int(model.config.n_inner)
    if hasattr(model.config, "n_embd"):
        return int(4 * model.config.n_embd)
    raise AttributeError("Unsupported model architecture: cannot infer MLP intermediate size")


def model_output_dir(root: Path, kind: str) -> Path:
    path = root / kind
    path.mkdir(parents=True, exist_ok=True)
    return path
