"""
Verify that save/load preserves weights exactly (bitwise) for a given checkpoint.

This is a targeted regression test for the "Save/Load: FAILED (Weight mismatch!)" message
from test_dataset_generalization.py.

Usage:
  python verify_checkpoint_roundtrip.py --checkpoint dataset_gen_checkpoints/my_model_stopped_step_1230.pt
"""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import Dict

import torch

from omni_model_v2 import OmniConfigV2, OmniFusionV2


def _load_state(path: str) -> Dict[str, torch.Tensor]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint format: {type(state)}")
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def _deduce_cfg(state: Dict[str, torch.Tensor]) -> OmniConfigV2:
    d_model = int(state["patch_embed.weight"].shape[0])
    in_channels = int(state["patch_embed.weight"].shape[1])
    patch_size = int(state["patch_embed.weight"].shape[2])
    vocab_size = int(state["text_embed.weight"].shape[0])

    layer_indices = []
    for k in state.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                layer_indices.append(int(parts[1]))
    n_layers = max(layer_indices) + 1

    head_dim = 64
    if d_model % head_dim != 0:
        raise ValueError(f"d_model={d_model} not divisible by head_dim=64")
    n_heads = d_model // head_dim
    qk_norm = any(k.endswith(".attn.q_norm.weight") for k in state.keys())
    text_pooling = "attn" if any(k.startswith("text_attn_pool.") for k in state.keys()) else "mean"

    return OmniConfigV2(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        patch_size=patch_size,
        in_channels=in_channels,
        qk_norm=qk_norm,
        attention_logit_cap=50.0,
        text_pooling=text_pooling,
        grad_checkpointing=False,
        lazy_logits=False,
        device="cpu",
        dtype="float32",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    args = ap.parse_args()

    state = _load_state(args.checkpoint)
    cfg = _deduce_cfg(state)
    model = OmniFusionV2(cfg).eval()
    model.load_state_dict(state, strict=True)
    if hasattr(model, "zero_padding_embedding"):
        model.zero_padding_embedding()

    # Save a clean, decompiled state_dict (what the test_dataset_generalization save/load test expects).
    state_to_save = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "roundtrip.pt")
        torch.save(state_to_save, path)

        model2 = OmniFusionV2(cfg).eval()
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        model2.load_state_dict(loaded, strict=True)
        if hasattr(model2, "zero_padding_embedding"):
            model2.zero_padding_embedding()

        sd2 = {k: v.detach().cpu() for k, v in model2.state_dict().items()}

    if set(state_to_save.keys()) != set(sd2.keys()):
        raise SystemExit("FAIL: key set changed across save/load.")

    mismatched = []
    for k in state_to_save.keys():
        if not torch.equal(state_to_save[k], sd2[k]):
            mismatched.append(k)
            if len(mismatched) >= 5:
                break

    if mismatched:
        raise SystemExit(f"FAIL: tensors differ after roundtrip. Example keys: {mismatched}")

    print("OK: checkpoint roundtrip is bitwise identical (state_dict tensors match).")


if __name__ == "__main__":
    main()
