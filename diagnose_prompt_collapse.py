"""
Standalone diagnostics for "prompt ignored / same image for different prompts" failures.

This script is intentionally evaluation-only:
- It does NOT train.
- It does NOT require a VAE or any Hugging Face downloads.
- It measures *mechanical* prompt influence (tensor deltas, attention mass), not CLIP-style semantics.

Usage:
  python diagnose_prompt_collapse.py --checkpoint dataset_gen_checkpoints/ImageOnlyBETA_step_10000.pt

Optional:
  python diagnose_prompt_collapse.py --checkpoint dataset_gen_checkpoints/ImageOnlyBETA_step_10000.pt --alpha-ntp 0.05

What it checks (with hard numbers):
1) Tokenization differences for prompts.
2) Prompt sensitivity in v_pred (packed image head outputs) at multiple t values.
3) Prompt sensitivity in full generation latents (same seed, different prompts; CFG enabled).
4) Attention selectivity proxy: average attention probability mass from IMAGE queries to TEXT keys
   in the first attention block.
5) (Optional) Explains why "scaled txt loss" often sits near ~alpha_ntp * ln(vocab) early in training.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from data_manager import TiktokenTokenizer
from omni_model_v2 import OmniConfigV2, OmniFusionV2


def _chatml(user_text: str, assistant_text: str) -> str:
    return (
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_text}<|im_end|>"
    )


@dataclass(frozen=True)
class ModelShape:
    d_model: int
    n_layers: int
    n_heads: int
    head_dim: int
    vocab_size: int
    patch_size: int
    in_channels: int
    qk_norm: bool
    text_pooling: str


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")


def _load_state_dict(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
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

    # Strip torch.compile prefix if present.
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    return {k: v.to(device) for k, v in state.items()}


def _deduce_shape(state: Dict[str, torch.Tensor]) -> ModelShape:
    if "patch_embed.weight" in state:
        d_model = int(state["patch_embed.weight"].shape[0])
        in_channels = int(state["patch_embed.weight"].shape[1])
        patch_size = int(state["patch_embed.weight"].shape[2])
    elif "text_embed.weight" in state:
        d_model = int(state["text_embed.weight"].shape[1])
        in_channels = 16
        patch_size = 2
    else:
        raise ValueError("Could not deduce d_model (missing patch_embed.weight and text_embed.weight).")

    vocab_size = int(state["text_embed.weight"].shape[0]) if "text_embed.weight" in state else 100352

    layer_indices: List[int] = []
    for k in state.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                layer_indices.append(int(parts[1]))
    n_layers = (max(layer_indices) + 1) if layer_indices else 0
    if n_layers <= 0:
        raise ValueError("Could not deduce n_layers (no blocks.* keys).")

    # This repo assumes head_dim=64; infer n_heads from d_model when possible.
    head_dim = 64
    if d_model % head_dim != 0:
        raise ValueError(f"d_model={d_model} not divisible by head_dim=64; cannot infer n_heads.")
    n_heads = d_model // head_dim

    qk_norm = any(k.endswith(".attn.q_norm.weight") for k in state.keys())
    text_pooling = "attn" if any(k.startswith("text_attn_pool.") for k in state.keys()) else "mean"

    return ModelShape(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        patch_size=patch_size,
        in_channels=in_channels,
        qk_norm=qk_norm,
        text_pooling=text_pooling,
    )


def _build_model(shape: ModelShape, device: torch.device, dtype: torch.dtype) -> OmniFusionV2:
    cfg = OmniConfigV2(
        d_model=shape.d_model,
        n_layers=shape.n_layers,
        n_heads=shape.n_heads,
        head_dim=shape.head_dim,
        vocab_size=shape.vocab_size,
        patch_size=shape.patch_size,
        in_channels=shape.in_channels,
        qk_norm=shape.qk_norm,
        attention_logit_cap=50.0,
        text_pooling=shape.text_pooling,
        grad_checkpointing=False,
        lazy_logits=False,
        device=str(device),
        dtype="bfloat16" if dtype == torch.bfloat16 else ("float16" if dtype == torch.float16 else "float32"),
    )
    model = OmniFusionV2(cfg).to(device).eval()
    if dtype == torch.bfloat16:
        model = model.bfloat16()
    elif dtype == torch.float16:
        model = model.half()
    return model


def _encode(tok: TiktokenTokenizer, text: str, device: torch.device) -> torch.Tensor:
    ids = tok.encode(text, add_pad=False, add_eot=True)
    if isinstance(ids, torch.Tensor):
        ids = ids.view(-1).long()
    else:
        ids = torch.as_tensor(ids, dtype=torch.long).view(-1)
    return ids.to(device)


@torch.no_grad()
def _vpred_packed(model: OmniFusionV2, prompt_ids: torch.Tensor, x_lat: torch.Tensor, t_val: float) -> torch.Tensor:
    device = x_lat.device
    dtype = x_lat.dtype
    t = torch.tensor([t_val], device=device, dtype=dtype)
    out = model.forward([prompt_ids], [x_lat], t, causal_text=True)
    mask = out["modality_mask"]
    img = out["image"][mask == 1.0]
    if img.numel() == 0:
        raise RuntimeError("No image tokens found in packed output.")
    return img


def _cos_mse(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    cos = float(F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item())
    mse = float(F.mse_loss(a_f, b_f).item())
    return cos, mse


def _capture_block0_attn_stats(
    model: OmniFusionV2,
    prompt_ids: torch.Tensor,
    x_lat: torch.Tensor,
    t_val: float,
) -> Dict[str, float]:
    """
    Proxy metrics for whether IMAGE queries are attending to TEXT keys in block 0.

    We capture the *inputs* to PackedSelfAttention in block 0, then recompute Q/K
    and a full softmax attention distribution on the captured tensors.

    Returns:
      mean_text_mass: average probability mass assigned to TEXT keys from IMAGE queries
      mean_entropy: mean Shannon entropy over all keys (averaged across image queries and heads)
    """
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
        _ = _vpred_packed(model, prompt_ids, x_lat, t_val)
    finally:
        h.remove()

    x_in = captured["x_in"]
    positions = captured["positions"]
    mod_mask = captured_kwargs.get("mod_mask", None)
    rope_func = captured.get("rope_func", None)
    if rope_func is None:
        rope_func = getattr(model, "rope", None)
    if rope_func is None:
        raise RuntimeError("Could not locate rope function on the attention call or model.")

    if mod_mask is None:
        raise RuntimeError("Could not capture mod_mask from block 0 attention call.")

    total_tokens = int(x_in.shape[0])
    q = attn0.q_proj(x_in).view(total_tokens, attn0.n_heads, attn0.head_dim)
    k = attn0.k_proj(x_in).view(total_tokens, attn0.n_kv_heads, attn0.head_dim)

    q = attn0.q_norm(q)
    k = attn0.k_norm(k)
    q, k = rope_func(q, k, positions)

    if attn0.n_kv_heads != attn0.n_heads:
        n_rep = attn0.n_heads // attn0.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)

    is_img = (mod_mask == 1.0)
    is_txt = (mod_mask == 0.0)
    if not bool(is_img.any()) or not bool(is_txt.any()):
        return {"mean_text_mass": float("nan"), "mean_entropy": float("nan")}

    q_img = q[is_img]  # [T_img, H, D]
    k_all = k  # [T, H, D]

    scale = 1.0 / math.sqrt(float(attn0.head_dim))
    # scores: [T_img, H, T]
    scores = torch.einsum("ihd,jhd->ihj", q_img, k_all) * scale
    if getattr(attn0, "logit_cap", 0.0) and float(attn0.logit_cap) > 0:
        cap = float(attn0.logit_cap)
        scores = cap * torch.tanh(scores / cap)

    probs = torch.softmax(scores, dim=-1)  # [T_img, H, T]
    text_mass = probs[..., is_txt].sum(dim=-1)  # [T_img, H]
    mean_text_mass = float(text_mass.mean().item())

    ent = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)  # [T_img, H]
    mean_entropy = float(ent.mean().item())
    return {"mean_text_mass": mean_text_mass, "mean_entropy": mean_entropy}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg-scale", type=float, default=4.5)
    ap.add_argument("--alpha-ntp", type=float, default=None, help="If set, prints alpha_ntp*ln(vocab) baseline.")
    ap.add_argument(
        "--wrap-chatml",
        action="store_true",
        help=(
            "Wrap each prompt into a ChatML 1-turn template (user + assistant). "
            "This is useful because latent-cache captions are often stored in ChatML format."
        ),
    )
    ap.add_argument(
        "--chatml-user",
        type=str,
        default="Provide booru-style tags for this image.",
        help="User message to use when --wrap-chatml is enabled.",
    )
    ap.add_argument("--prompts", type=str, nargs="*", default=[
        "blue",
        "red",
        "ocean beach water waves blue",
        "forest trees leaves green",
        "anime girl red hair yellow eyes",
        "anime girl blue hair blue eyes",
    ])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    state = _load_state_dict(args.checkpoint, device=device)
    shape = _deduce_shape(state)
    model = _build_model(shape, device=device, dtype=dtype)

    incompatible = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        raise SystemExit(
            f"Checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)} "
            f"(example missing={missing[:5]} unexpected={unexpected[:5]})"
        )
    if hasattr(model, "zero_padding_embedding"):
        model.zero_padding_embedding()

    tok = TiktokenTokenizer()

    print("=== Model ===")
    print(f"checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"device: {device.type} | dtype: {dtype}")
    print(
        f"d_model={shape.d_model} n_layers={shape.n_layers} n_heads={shape.n_heads} head_dim={shape.head_dim} "
        f"vocab={shape.vocab_size} patch={shape.patch_size} in_ch={shape.in_channels} qk_norm={shape.qk_norm}"
    )

    if args.alpha_ntp is not None:
        baseline = float(args.alpha_ntp) * math.log(float(shape.vocab_size))
        print("\n=== Scaled Text Loss Baseline (Random Logits) ===")
        print(f"alpha_ntp={args.alpha_ntp} | ln(vocab)={math.log(float(shape.vocab_size)):.4f} | alpha*ln(vocab)={baseline:.4f}")

    print("\n=== Tokenization ===")
    encoded: List[Tuple[str, torch.Tensor]] = []
    for p in args.prompts:
        p_effective = _chatml(str(args.chatml_user), p) if bool(args.wrap_chatml) else p
        ids = _encode(tok, p_effective, device=device)
        encoded.append((p, ids))
        if bool(args.wrap_chatml):
            print(f"prompt='{p}' (ChatML-wrapped) | n_tokens={int(ids.numel())} | first_ids={ids[:6].tolist()}")
        else:
            print(f"prompt='{p}' | n_tokens={int(ids.numel())} | first_ids={ids[:6].tolist()}")

    # Fixed latent (256x256 -> 32x32 latent for FLUX VAE downsample=8).
    torch.manual_seed(args.seed)
    x_lat = torch.randn(shape.in_channels, 32, 32, device=device, dtype=dtype)

    print("\n=== v_pred Prompt Sensitivity (Packed Image Tokens) ===")
    for t_val in (0.1, 0.5, 0.9):
        ref = _vpred_packed(model, encoded[0][1], x_lat, t_val)
        for p, ids in encoded[1:]:
            cur = _vpred_packed(model, ids, x_lat, t_val)
            cos, mse = _cos_mse(ref, cur)
            print(f"t={t_val:.1f} | vs '{encoded[0][0]}' -> '{p}' | cos={cos:.6f} mse={mse:.6f}")

    print("\n=== Block0 Attention Proxy (IMAGE queries -> TEXT keys) ===")
    for p, ids in encoded[:3]:
        stats = _capture_block0_attn_stats(model, ids, x_lat, t_val=0.1)
        print(f"prompt='{p}' | mean_text_mass={stats['mean_text_mass']:.4f} | mean_entropy={stats['mean_entropy']:.3f}")

    print("\n=== Generation Latent Sensitivity (Same Seed, CFG Enabled) ===")
    gen_latents: List[Tuple[str, torch.Tensor]] = []
    for p, ids in encoded[:3]:
        torch.manual_seed(args.seed)
        lat = model.generate([ids], height=256, width=256, steps=int(args.steps), cfg_scale=float(args.cfg_scale))[0]
        gen_latents.append((p, lat.detach()))
        print(f"generated prompt='{p}' | latent_mean={float(lat.float().mean().item()):.4f} | latent_std={float(lat.float().std().item()):.4f}")

    ref_p, ref_lat = gen_latents[0]
    for p, lat in gen_latents[1:]:
        cos, mse = _cos_mse(ref_lat, lat)
        print(f"seed={args.seed} steps={args.steps} cfg={args.cfg_scale} | '{ref_p}' vs '{p}' | cos={cos:.6f} mse={mse:.6f}")


if __name__ == "__main__":
    main()
