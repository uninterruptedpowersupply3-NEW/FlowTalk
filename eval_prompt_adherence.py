"""
Eval-only prompt adherence check for OmniFusionV2 checkpoints.

What it proves:
- Determinism: same seed + same prompt => (nearly) identical output.
- Prompt sensitivity: same seed + different prompts => measurably different outputs.

It does NOT claim semantic correctness (CLIP-style alignment). It only checks that
text tokens actually influence the image pathway.

Usage (recommended for cache-only datasets):
  python eval_prompt_adherence.py --checkpoint dataset_gen_checkpoints/ImageOnlyBETA.pt
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from omni_model_v2 import OmniConfigV2, OmniFusionV2
from data_manager import TiktokenTokenizer

# Reuse the exact scale/shift constants used throughout this repo.
from test_dataset_generalization import (  # noqa: E402
    VAE_SCALE_FACTOR,
    VAE_SHIFT_FACTOR,
    encode_prompt_tokens,
)
from vae_module import FluxVAE  # noqa: E402


@dataclass(frozen=True)
class Metrics:
    max_abs_latent_diff: float
    mean_abs_vpred_diff: float


def _load_checkpoint_to_model(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = None
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

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

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}). Example: {missing[:3]}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}). Example: {unexpected[:3]}")

    model.to(device)


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    SSIM for [1,3,H,W] float tensors in [0,1].
    Matches the implementation style used in test_dataset_generalization.py.
    """
    if img1.dim() != 4 or img2.dim() != 4:
        raise ValueError("SSIM expects [B,C,H,W].")
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {tuple(img1.shape)} vs {tuple(img2.shape)}")

    sigma = 1.5
    gauss = torch.tensor(
        [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma**2)) for x in range(window_size)],
        device=img1.device,
        dtype=img1.dtype,
    )
    gauss = gauss / gauss.sum()
    w1d = gauss.unsqueeze(1)
    w2d = w1d @ w1d.t()
    window = w2d.float().unsqueeze(0).unsqueeze(0).expand(img1.size(1), 1, window_size, window_size).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


def _unpatchify_velocity(
    img_tokens: torch.Tensor, *, in_channels: int, patch_size: int, h_lat: int, w_lat: int
) -> torch.Tensor:
    """
    img_tokens: [L, C*p*p] where L=(h_lat/p)*(w_lat/p)
    Returns v_pred [C, h_lat, w_lat]
    """
    p = patch_size
    gh, gw = h_lat // p, w_lat // p
    L = gh * gw
    expected_dim = in_channels * p * p

    if img_tokens.dim() == 3:
        img_tokens = img_tokens.view(-1, img_tokens.shape[-1])

    if img_tokens.shape != (L, expected_dim):
        raise AssertionError(f"img_tokens shape {tuple(img_tokens.shape)} expected {(L, expected_dim)}")

    fold_input = img_tokens.transpose(0, 1).unsqueeze(0)  # [1, C*p*p, L]
    v_pred = (
        F.fold(
            fold_input,
            output_size=(h_lat, w_lat),
            kernel_size=p,
            stride=p,
        )
        .squeeze(0)
        .contiguous()
    )
    return v_pred


@torch.no_grad()
def _vpred_at_t(
    model: OmniFusionV2,
    prompt_ids: torch.Tensor,
    x_latents: torch.Tensor,
    t_val: float,
    *,
    dtype: torch.dtype,
    patch_size: int,
    in_channels: int,
) -> torch.Tensor:
    device = x_latents.device
    h_lat, w_lat = int(x_latents.shape[-2]), int(x_latents.shape[-1])
    t_batch = torch.full((1,), t_val, device=device, dtype=dtype)
    amp_ctx = torch.amp.autocast("cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()
    with amp_ctx:
        out = model.forward([prompt_ids], [x_latents], t_batch, causal_text=True)
        pred_v_packed = out["image"]
        mod_mask = out["modality_mask"]
        img_tokens = pred_v_packed[mod_mask == 1.0]
        return _unpatchify_velocity(img_tokens, in_channels=in_channels, patch_size=patch_size, h_lat=h_lat, w_lat=w_lat)


@torch.no_grad()
def _euler_generate(
    model: OmniFusionV2,
    prompt_ids: torch.Tensor,
    noise: torch.Tensor,
    steps: int,
    *,
    dtype: torch.dtype,
    patch_size: int,
    in_channels: int,
) -> torch.Tensor:
    device = noise.device
    h_lat, w_lat = int(noise.shape[-2]), int(noise.shape[-1])

    latents_gen = noise.clone()
    dt = 1.0 / float(steps)
    amp_ctx = torch.amp.autocast("cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()

    for step in range(steps):
        t_curr = float(step) * dt
        t_batch = torch.full((1,), t_curr, device=device, dtype=dtype)
        with amp_ctx:
            out = model.forward([prompt_ids], [latents_gen], t_batch, causal_text=True)
            pred_v_packed = out["image"]
            mod_mask = out["modality_mask"]
            img_tokens = pred_v_packed[mod_mask == 1.0]
            v_pred = _unpatchify_velocity(
                img_tokens, in_channels=in_channels, patch_size=patch_size, h_lat=h_lat, w_lat=w_lat
            )
        latents_gen = latents_gen + v_pred * dt

    return latents_gen


@torch.no_grad()
def _decode_to_image01(vae: FluxVAE, latents_scaled: torch.Tensor) -> torch.Tensor:
    """
    latents_scaled: [C,H,W] in the same latent space used by training (post extra affine transform).
    Returns image [1,3,Himg,Wimg] in [0,1].
    """
    if latents_scaled.dim() != 3:
        raise ValueError("Expected [C,H,W] latents.")
    device = next(vae.parameters()).device

    # Invert the extra affine transform used by this repo before calling FluxVAE.decode().
    # test_dataset_generalization.py uses: raw = scaled * VAE_SCALE_FACTOR + VAE_SHIFT_FACTOR
    # and then passes raw into FluxVAE.decode(), which performs: (raw / scale) + shift
    # yielding the original VAE latent space.
    gen_raw = (latents_scaled.unsqueeze(0).float() * float(VAE_SCALE_FACTOR)) + float(VAE_SHIFT_FACTOR)  # [1,C,H,W]

    rgb = vae.decode(gen_raw.to(device=device, dtype=vae.dtype))  # [1,3,H,W] in [-1,1]
    rgb01 = (rgb.clamp(-1, 1) + 1.0) * 0.5
    return rgb01


def _save_img(rgb01: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = (rgb01[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def _pairwise_scores(images: List[torch.Tensor]) -> Tuple[float, float]:
    ssims: List[float] = []
    mses: List[float] = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            a = images[i]
            b = images[j]
            ssims.append(_ssim(a, b))
            mses.append(float(F.mse_loss(a, b).item()))
    return (sum(ssims) / max(1, len(ssims))), (sum(mses) / max(1, len(mses)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=os.path.join("dataset_gen_checkpoints", "ImageOnlyBETA.pt"))
    ap.add_argument("--out-dir", type=str, default=os.path.join("prompt_adherence_outputs", "ImageOnlyBETA"))
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--n-heads", type=int, default=12)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--h-lat", type=int, default=16)
    ap.add_argument("--w-lat", type=int, default=16)
    ap.add_argument("--t-probe", type=float, default=0.5, help="t in [0,1) to probe v_pred sensitivity")
    ap.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default=None,
        help="Optional prompts list. If omitted, uses the built-in sanity prompts.",
    )
    ap.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional newline-delimited prompt file. Blank lines are ignored.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device.type == "cuda" else torch.float32)

    tok = TiktokenTokenizer()

    cfg = OmniConfigV2(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        vocab_size=100352,
        device=str(device),
        dtype="bfloat16" if dtype == torch.bfloat16 else ("float16" if dtype == torch.float16 else "float32"),
        qk_norm=True,
        attention_logit_cap=50.0,
        grad_checkpointing=False,
        lazy_logits=False,
        drop_path_rate=0.0,
    )
    model = OmniFusionV2(cfg).to(device).eval()
    if dtype == torch.bfloat16:
        model = model.bfloat16()
    elif dtype == torch.float16:
        model = model.half()

    _load_checkpoint_to_model(model, args.checkpoint, device)
    model.eval()

    # VAE for image decoding.
    vae = FluxVAE(dtype=torch.float32).to(device).eval()

    prompts: List[str] | None = None
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f.readlines() if ln.strip()]
    elif args.prompts:
        prompts = [p.strip() for p in args.prompts if p.strip()]

    if not prompts:
        prompts = [
            # short prompts (user reported weak conditioning for very short prompts)
            "blue square",
            "red circle",
            "green triangle",
            # longer prompts
            "a blue square on a black background",
            "a red circle centered on a white background",
            "a green triangle on dark background with soft edges",
        ]

    prompt_ids_list: List[torch.Tensor] = [encode_prompt_tokens(tok, p, add_eot=True).to(device) for p in prompts]

    # Fixed noise shared across prompts.
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    noise = torch.randn(cfg.in_channels, args.h_lat, args.w_lat, device=device, dtype=dtype)

    # 1) Determinism check: same prompt twice => same latents.
    lat_a1 = _euler_generate(
        model,
        prompt_ids_list[0],
        noise,
        steps=args.steps,
        dtype=dtype,
        patch_size=cfg.patch_size,
        in_channels=cfg.in_channels,
    )
    lat_a2 = _euler_generate(
        model,
        prompt_ids_list[0],
        noise,
        steps=args.steps,
        dtype=dtype,
        patch_size=cfg.patch_size,
        in_channels=cfg.in_channels,
    )
    max_abs_latent_diff = float((lat_a1 - lat_a2).abs().max().item())

    # 2) v_pred sensitivity at a fixed t (same x_t).
    vpreds: List[torch.Tensor] = [
        _vpred_at_t(
            model,
            pid,
            noise,
            float(args.t_probe),
            dtype=dtype,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
        )
        for pid in prompt_ids_list
    ]
    base_v = vpreds[0]
    mean_abs_vpred_diff = float(torch.stack([(v - base_v).abs().mean() for v in vpreds[1:]]).mean().item())

    # 3) Full images per prompt (same noise).
    decoded_images: List[torch.Tensor] = []
    for i, (p, pid) in enumerate(zip(prompts, prompt_ids_list)):
        lat = _euler_generate(
            model,
            pid,
            noise,
            steps=args.steps,
            dtype=dtype,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
        )
        img01 = _decode_to_image01(vae, lat)  # [1,3,H,W]
        decoded_images.append(img01.cpu())
        out_path = os.path.join(args.out_dir, f"{i:02d}.png")
        _save_img(img01, out_path)
        print(f"[SAVE] {out_path} | prompt='{p}'")

    mean_pairwise_ssim, mean_pairwise_mse = _pairwise_scores(decoded_images)

    print("\n=== Prompt Adherence Metrics ===")
    print(f"Checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"Device: {device} | dtype: {dtype}")
    print(f"Latents: C={cfg.in_channels} H={args.h_lat} W={args.w_lat} | steps={args.steps} | seed={args.seed}")
    print(f"Determinism: max_abs_latent_diff(same prompt, same noise) = {max_abs_latent_diff:.6e}")
    print(
        f"v_pred sensitivity @ t={float(args.t_probe):.3f}: "
        f"mean_abs_vpred_diff(vs first prompt) = {mean_abs_vpred_diff:.6e}"
    )
    print(f"Image diversity (same seed, different prompts): mean_pairwise_ssim = {mean_pairwise_ssim:.4f}")
    print(f"Image diversity (same seed, different prompts): mean_pairwise_mse  = {mean_pairwise_mse:.4f}")

    # Hard-fail only on determinism. Sensitivity thresholds are workload-dependent; print only.
    if max_abs_latent_diff > 1e-5:
        raise SystemExit(
            "Non-deterministic sampling detected "
            f"(max_abs_latent_diff={max_abs_latent_diff:.3e}). "
            "This prevents attributing differences to prompts reliably."
        )


if __name__ == "__main__":
    main()
