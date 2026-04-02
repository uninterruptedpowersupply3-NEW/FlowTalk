"""
Fast scan: how prompt sensitivity evolves across a series of checkpoints.

This is meant to answer questions like:
"Why do 4 epochs vs 6 epochs look the same? Did conditioning saturate early?"

It avoids VAE decode and full image sampling. It only measures:
- v_pred delta at a fixed (noisy) t value
- block0 mean text-attention mass (IMAGE queries -> TEXT keys)

Usage:
  python scan_prompt_sensitivity_checkpoints.py --pattern "dataset_gen_checkpoints/ImageOnlyBETA_step_*.pt"
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from data_manager import TiktokenTokenizer
from omni_model_v2 import OmniConfigV2, OmniFusionV2


def _load_state(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
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
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    return {k: v.to(device) for k, v in state.items()}


def _deduce_cfg(state: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> OmniConfigV2:
    d_model = int(state["patch_embed.weight"].shape[0])
    in_channels = int(state["patch_embed.weight"].shape[1])
    patch_size = int(state["patch_embed.weight"].shape[2])
    vocab_size = int(state["text_embed.weight"].shape[0])

    layer_indices: List[int] = []
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
        device=str(device),
        dtype="bfloat16" if dtype == torch.bfloat16 else ("float16" if dtype == torch.float16 else "float32"),
    )


def _encode(tok: TiktokenTokenizer, text: str, device: torch.device) -> torch.Tensor:
    ids = tok.encode(text, add_pad=False, add_eot=True)
    if isinstance(ids, torch.Tensor):
        ids = ids.view(-1).long()
    else:
        ids = torch.as_tensor(ids, dtype=torch.long).view(-1)
    return ids.to(device)


@torch.no_grad()
def _vpred_img_tokens(model: OmniFusionV2, prompt_ids: torch.Tensor, x_lat: torch.Tensor, t_val: float) -> torch.Tensor:
    device = x_lat.device
    t = torch.tensor([t_val], device=device, dtype=x_lat.dtype)
    out = model.forward([prompt_ids], [x_lat], t, causal_text=True)
    mask = out["modality_mask"]
    return out["image"][mask == 1.0]


def _cos_mse(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    cos = float(F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item())
    mse = float(F.mse_loss(a_f, b_f).item())
    return cos, mse


def _block0_text_mass(model: OmniFusionV2, prompt_ids: torch.Tensor, x_lat: torch.Tensor, t_val: float) -> float:
    attn0 = model.blocks[0].attn
    captured: Dict[str, torch.Tensor] = {}
    captured_kwargs: Dict[str, object] = {}

    def pre_hook(_mod, args, kwargs):
        captured["x_in"] = args[0].detach()
        captured["rope_func"] = args[1]
        captured["positions"] = args[2].detach()
        captured_kwargs.update(kwargs)

    h = attn0.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        _ = _vpred_img_tokens(model, prompt_ids, x_lat, t_val)
    finally:
        h.remove()

    x_in = captured["x_in"]
    rope_func = captured["rope_func"]
    positions = captured["positions"]
    mod_mask = captured_kwargs.get("mod_mask", None)
    if mod_mask is None:
        return float("nan")

    T = int(x_in.shape[0])
    q = attn0.q_proj(x_in).view(T, attn0.n_heads, attn0.head_dim)
    k = attn0.k_proj(x_in).view(T, attn0.n_kv_heads, attn0.head_dim)
    q = attn0.q_norm(q)
    k = attn0.k_norm(k)
    q, k = rope_func(q, k, positions)
    if attn0.n_kv_heads != attn0.n_heads:
        n_rep = attn0.n_heads // attn0.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)

    is_img = (mod_mask == 1.0)
    is_txt = (mod_mask == 0.0)
    if not bool(is_img.any()) or not bool(is_txt.any()):
        return float("nan")

    q_img = q[is_img]
    k_all = k
    scale = 1.0 / math.sqrt(float(attn0.head_dim))
    scores = torch.einsum("ihd,jhd->ihj", q_img, k_all) * scale
    if getattr(attn0, "logit_cap", 0.0) and float(attn0.logit_cap) > 0:
        cap = float(attn0.logit_cap)
        scores = cap * torch.tanh(scores / cap)
    probs = torch.softmax(scores, dim=-1)
    mass = probs[..., is_txt].sum(dim=-1).mean()
    return float(mass.item())


def _extract_step(path: str) -> int:
    m = re.search(r"step_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t", type=float, default=0.1)
    ap.add_argument("--prompt-a", type=str, default="blue")
    ap.add_argument("--prompt-b", type=str, default="ocean beach water waves blue")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tok = TiktokenTokenizer()
    ids_a = _encode(tok, args.prompt_a, device=device)
    ids_b = _encode(tok, args.prompt_b, device=device)

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No checkpoints matched: {args.pattern}")
    paths = sorted(paths, key=_extract_step)

    print("step,cos_vpred,mse_vpred,block0_text_mass")

    for p in paths:
        state = _load_state(p, device=device)
        cfg = _deduce_cfg(state, device=device, dtype=dtype)
        model = OmniFusionV2(cfg).to(device).eval()
        if dtype == torch.bfloat16:
            model = model.bfloat16()
        elif dtype == torch.float16:
            model = model.half()

        incompatible = model.load_state_dict(state, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if missing or unexpected:
            raise SystemExit(f"Load mismatch for {p}: missing={len(missing)} unexpected={len(unexpected)}")
        if hasattr(model, "zero_padding_embedding"):
            model.zero_padding_embedding()

        torch.manual_seed(args.seed)
        x_lat = torch.randn(cfg.in_channels, 32, 32, device=device, dtype=dtype)

        va = _vpred_img_tokens(model, ids_a, x_lat, float(args.t))
        vb = _vpred_img_tokens(model, ids_b, x_lat, float(args.t))
        cos, mse = _cos_mse(va, vb)
        mass = _block0_text_mass(model, ids_a, x_lat, float(args.t))

        step = _extract_step(p)
        print(f"{step},{cos:.6f},{mse:.6f},{mass:.4f}")

        # Free VRAM between checkpoints.
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None


if __name__ == "__main__":
    main()
