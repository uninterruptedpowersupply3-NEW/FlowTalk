"""
Verify which text-conditioning pathway is actually driving image generation.

This script is designed to *prove*, using concrete measurements on a real checkpoint,
whether prompt sensitivity comes from:
1) token-level text -> image attention (image queries attending to text keys/values), or
2) pooled-text conditioning injected into image AdaLN conditioning via `text_pool_proj`.

It measures (for two prompts, same x_t and t):
- v_pred difference on IMAGE tokens (cosine + MSE)
- v_pred difference with pooled-text injection ablated (text_pool_proj -> zeros)
- block0 IMAGE->TEXT attention entropy (how selective attention is over the text tokens)
- pooled-text vector similarity (mean-pooled text embedding and projected text_cond)

Usage:
  C:\\Users\\chatr\\Documents\\Tech\\VLLM\\venv\\Scripts\\python.exe verify_conditioning_pathways.py ^
    --checkpoint .\\dataset_gen_checkpoints\\ImageOnlyBETA_step_10000.pt ^
    --prompt-a "1girl, blue hair" ^
    --prompt-b "1girl, red hair"
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_manager import TiktokenTokenizer
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
        raise TypeError(f"Unexpected checkpoint type: {type(state)}")
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def _deduce_cfg(state: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> OmniConfigV2:
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
        raise ValueError(f"d_model={d_model} not divisible by head_dim={head_dim}")
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
    t = torch.tensor([t_val], device=x_lat.device, dtype=x_lat.dtype)
    out = model.forward([prompt_ids], [x_lat], t, causal_text=True)
    mask = out["modality_mask"]
    return out["image"][mask == 1.0]


def _cos_mse(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    cos = float(F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item())
    mse = float(F.mse_loss(a_f, b_f).item())
    return cos, mse


class _ZeroProj(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.zeros_like(x)


def _entropy(p: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return -(p * (p + eps).log()).sum(dim=-1)


@torch.no_grad()
def _block0_img_to_txt_entropy(model: OmniFusionV2, prompt_ids: torch.Tensor, x_lat: torch.Tensor, t_val: float) -> Tuple[int, float, float]:
    """
    Returns (n_text_tokens, mean_entropy, entropy_ratio_vs_uniform_max).

    Important: excludes tail block-padding tokens by slicing to cu_seqlens[-1].
    """
    attn0 = model.blocks[0].attn
    captured: Dict[str, torch.Tensor] = {}

    def pre_hook(_mod, args, kwargs):
        # args: x, rope_func, positions, cu_seqlens, max_seqlen
        captured["x_in"] = args[0].detach()
        captured["rope_func"] = args[1]
        captured["positions"] = args[2].detach()
        captured["cu_seqlens"] = args[3].detach()
        captured["mod_mask"] = kwargs.get("mod_mask", None)

    h = attn0.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        _ = _vpred_img_tokens(model, prompt_ids, x_lat, t_val)
    finally:
        h.remove()

    x_in = captured["x_in"]
    rope_func = captured["rope_func"]
    positions = captured["positions"]
    cu = captured["cu_seqlens"]
    mod_mask = captured["mod_mask"]
    if mod_mask is None:
        raise RuntimeError("Failed to capture mod_mask from block0 attention pre-hook.")

    real_len = int(cu[-1].item())
    x_in = x_in[:real_len]
    positions = positions[:real_len]
    mod_mask = mod_mask[:real_len]

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
        raise RuntimeError(f"Unexpected mod_mask split: img_any={bool(is_img.any())} txt_any={bool(is_txt.any())}")

    q_img = q[is_img]  # [Ti,H,D]
    scale = 1.0 / math.sqrt(float(attn0.head_dim))
    scores = torch.einsum("ihd,jhd->ihj", q_img, k) * scale
    if getattr(attn0, "logit_cap", 0.0) and float(attn0.logit_cap) > 0:
        cap = float(attn0.logit_cap)
        scores = cap * torch.tanh(scores / cap)
    probs = torch.softmax(scores, dim=-1)  # [Ti,H,T]

    txt_probs = probs[..., is_txt]  # [Ti,H,T_txt]
    txt_probs = txt_probs / txt_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    ent = float(_entropy(txt_probs).mean().item())
    n_txt = int(txt_probs.shape[-1])
    max_ent = math.log(n_txt) if n_txt > 0 else float("nan")
    ratio = float(ent / max_ent) if max_ent > 0 else float("nan")
    return n_txt, ent, ratio


@torch.no_grad()
def _pooled_text_vectors(model: OmniFusionV2, tok: TiktokenTokenizer, prompt_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    emb = model.text_embed(prompt_ids)
    pad = tok.pad_token
    mask = (prompt_ids != pad).to(emb.dtype).unsqueeze(-1)
    denom = mask.sum().clamp(min=1.0)
    pooled = (emb * mask).sum(dim=0, keepdim=True) / denom
    text_cond = model.text_pool_proj(pooled)
    return pooled, text_cond


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--prompt-a", type=str, required=True)
    ap.add_argument("--prompt-b", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t", type=float, default=0.1)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--w", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    state = _load_state(args.checkpoint)
    cfg = _deduce_cfg(state, device=device, dtype=dtype)
    model = OmniFusionV2(cfg).to(device).eval()
    if dtype == torch.bfloat16:
        model = model.bfloat16()
    elif dtype == torch.float16:
        model = model.half()

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"Checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    if hasattr(model, "zero_padding_embedding"):
        model.zero_padding_embedding()

    tok = TiktokenTokenizer(max_length=2048)
    ids_a = _encode(tok, args.prompt_a, device=device)
    ids_b = _encode(tok, args.prompt_b, device=device)

    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    x_lat = torch.randn(cfg.in_channels, int(args.h), int(args.w), device=device, dtype=dtype)

    # Baseline: prompt sensitivity with current checkpoint.
    va = _vpred_img_tokens(model, ids_a, x_lat, float(args.t))
    vb = _vpred_img_tokens(model, ids_b, x_lat, float(args.t))
    cos0, mse0 = _cos_mse(va, vb)

    # Ablation: remove pooled-text injection to see if token-level attention drives conditioning.
    orig_pool = model.text_pool_proj
    model.text_pool_proj = _ZeroProj().to(device=device, dtype=model.text_embed.weight.dtype)
    try:
        va2 = _vpred_img_tokens(model, ids_a, x_lat, float(args.t))
        vb2 = _vpred_img_tokens(model, ids_b, x_lat, float(args.t))
    finally:
        model.text_pool_proj = orig_pool

    cos1, mse1 = _cos_mse(va2, vb2)

    # Attention entropy: how selective IMAGE->TEXT attention is in block0.
    n_txt_a, ent_a, ent_ratio_a = _block0_img_to_txt_entropy(model, ids_a, x_lat, float(args.t))
    n_txt_b, ent_b, ent_ratio_b = _block0_img_to_txt_entropy(model, ids_b, x_lat, float(args.t))
    if n_txt_a != n_txt_b:
        raise SystemExit(f"Text token count differs between prompts (unexpected): {n_txt_a} vs {n_txt_b}")

    pooled_a, cond_a = _pooled_text_vectors(model, tok, ids_a)
    pooled_b, cond_b = _pooled_text_vectors(model, tok, ids_b)
    pooled_cos = float(F.cosine_similarity(pooled_a.float(), pooled_b.float()).item())
    cond_cos = float(F.cosine_similarity(cond_a.float(), cond_b.float()).item())

    print("\n=== Conditioning Pathway Verification ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device} | dtype={dtype}")
    print(f"x_lat: C={cfg.in_channels} H={args.h} W={args.w} | t={args.t} | seed={args.seed}")
    print(f"Prompt A: {args.prompt_a!r}")
    print(f"Prompt B: {args.prompt_b!r}")

    print("\n[1] v_pred difference on IMAGE tokens (baseline)")
    print(f"  cosine: {cos0:.6f}")
    print(f"  mse:    {mse0:.6f}")

    print("\n[2] v_pred difference with pooled-text injection ablated (text_pool_proj -> 0)")
    print(f"  cosine: {cos1:.6f}")
    print(f"  mse:    {mse1:.6f}")

    if mse0 > 0:
        frac = mse1 / mse0
        print(f"  mse_fraction_remaining: {frac:.3f} (lower => pooled conditioning dominates)")

    print("\n[3] block0 IMAGE->TEXT attention entropy (lower => more selective over text tokens)")
    print(f"  n_text_tokens: {n_txt_a}")
    print(f"  mean_entropy(A): {ent_a:.4f} | ratio_vs_uniform_max: {ent_ratio_a:.3f}")
    print(f"  mean_entropy(B): {ent_b:.4f} | ratio_vs_uniform_max: {ent_ratio_b:.3f}")

    print("\n[4] pooled-text similarity")
    print(f"  cosine(pooled_mean_text_emb): {pooled_cos:.6f}")
    print(f"  cosine(text_cond=text_pool_proj(pooled)): {cond_cos:.6f}")

    # Hard evidence threshold: if ablation kills most prompt sensitivity, token-level text attention
    # is not the primary conditioning pathway in this checkpoint.
    if mse0 > 0 and mse1 / mse0 < 0.3:
        print("\n[RESULT] Pooled text conditioning is the dominant prompt-sensitivity pathway in this checkpoint.")
    else:
        print("\n[RESULT] Token-level text attention contributes substantially to prompt sensitivity in this checkpoint.")


if __name__ == "__main__":
    main()
