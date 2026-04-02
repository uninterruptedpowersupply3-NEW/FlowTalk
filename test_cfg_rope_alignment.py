"""
CFG RoPE Alignment Regression Test
=================================

This repo uses 3D Axial RoPE where image tokens get a *temporal* coordinate that depends on
the running `temporal_pos` in `OmniFusionV2.pack_inputs()`.

If classifier-free guidance (CFG) uses an unconditional prompt that is *shorter* than the
conditional prompt (e.g. a single EOT token), then the image tokens in the unconditional
half receive different temporal RoPE coordinates than the conditional half. Subtracting
those two velocity fields is not mathematically valid, because they were evaluated at
different position encodings.

This test:
1) Statically checks that the CFG unconditional path does NOT create a 1-token prompt.
2) Empirically proves that padding the unconditional prompt to the conditional length
   makes image-token temporal positions match.
"""

from __future__ import annotations

import pathlib

import torch

from omni_model_v2 import OmniConfigV2, OmniFusionV2


EOT_TOKEN_ID = 100257
PAD_TOKEN_ID = 100258


def _extract_image_temporal_positions(
    packed_pos: torch.Tensor, mod_mask: torch.Tensor, cu_seqlens: torch.Tensor, sample_idx: int
) -> torch.Tensor:
    start = int(cu_seqlens[sample_idx].item())
    end = int(cu_seqlens[sample_idx + 1].item())
    pos_slice = packed_pos[start:end]
    mask_slice = mod_mask[start:end]
    img_pos = pos_slice[mask_slice == 1.0]
    if img_pos.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.unique(img_pos[:, 0].to(torch.long))


def test_static_codepaths() -> None:
    text = pathlib.Path("omni_model_v2.py").read_text(encoding="utf-8", errors="ignore")
    # Ensure the CFG unconditional path pads to conditional length (PAD everywhere, EOT at the end).
    assert "uncond = txt.new_full((pad_len,), 100258)" in text and "uncond[-1] = 100257" in text, (
        "CFG unconditional prompt padding not detected in omni_model_v2.py. "
        "Expected PAD-fill + EOT-at-end logic to keep image-token RoPE temporal positions aligned."
    )


def test_empirical_alignment() -> None:
    # Small model for a fast pack_inputs() call. We only need embeddings + patch embedding.
    cfg = OmniConfigV2(
        d_model=128,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        head_dim=32,
        device="cpu",
        dtype="float32",
        grad_checkpointing=False,
        lazy_logits=False,
    )
    model = OmniFusionV2(cfg).eval()

    # Conditional prompt (length includes EOT at the end).
    cond = torch.tensor([42, 43, 44, 45, 46, 47, 48, 49, 50, EOT_TOKEN_ID], dtype=torch.long)
    # "Buggy" unconditional: single EOT token.
    uncond_short = torch.tensor([EOT_TOKEN_ID], dtype=torch.long)
    # Fixed unconditional: PAD everywhere, EOT at end, same length as conditional.
    uncond_padded = cond.new_full(cond.shape, PAD_TOKEN_ID)
    uncond_padded[-1] = EOT_TOKEN_ID

    # Dummy latent image (model-space VAE latents, shape [C,H,W]).
    img = torch.randn(16, 8, 8, dtype=torch.float32)

    # Pack conditional.
    _, _, pos_c, mask_c, cu_c, _, _ = model.pack_inputs([cond], [img], timesteps=torch.zeros(1), pad=False)
    t_img_c = _extract_image_temporal_positions(pos_c, mask_c, cu_c, 0)
    assert t_img_c.numel() > 0, "No image tokens detected in packed output."

    # Pack unconditional short (should NOT match temporal positions).
    _, _, pos_u, mask_u, cu_u, _, _ = model.pack_inputs([uncond_short], [img], timesteps=torch.zeros(1), pad=False)
    t_img_u_short = _extract_image_temporal_positions(pos_u, mask_u, cu_u, 0)
    assert t_img_u_short.numel() > 0, "No image tokens detected in packed unconditional output."
    assert not torch.equal(t_img_c, t_img_u_short), (
        "Expected conditional/unconditional image temporal positions to differ when unconditional text length differs."
    )

    # Pack unconditional padded (should match temporal positions exactly).
    _, _, pos_up, mask_up, cu_up, _, _ = model.pack_inputs([uncond_padded], [img], timesteps=torch.zeros(1), pad=False)
    t_img_u_padded = _extract_image_temporal_positions(pos_up, mask_up, cu_up, 0)
    assert torch.equal(t_img_c, t_img_u_padded), (
        f"Unconditional padded prompt did not align image temporal positions. cond={t_img_c.tolist()} "
        f"uncond={t_img_u_padded.tolist()}"
    )


def main() -> None:
    test_static_codepaths()
    test_empirical_alignment()
    print("PASS: CFG unconditional padding keeps image-token RoPE temporal positions aligned.")


if __name__ == "__main__":
    main()
