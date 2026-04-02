"""
Regression Test: empty_prompt_tokens() Must Preserve CFG RoPE Alignment
=====================================================================

Training uses CFG dropout (replace conditional text with an "empty" prompt) so the model learns
an unconditional prediction. In this repo, image-token *temporal* RoPE positions depend on the
running text length inside OmniFusionV2.pack_inputs().

If the unconditional prompt is shorter than the conditional prompt, then conditional and
unconditional velocity fields are evaluated under different RoPE positions, and CFG subtraction
is no longer mathematically valid.

This test verifies that:
1) empty_prompt_tokens(..., like=cond_ids) returns a prompt with identical length to cond_ids.
2) That prompt aligns image temporal positions in pack_inputs().
"""

from __future__ import annotations

import torch

from data_manager import TiktokenTokenizer
from omni_model_v2 import OmniConfigV2, OmniFusionV2
from test_dataset_generalization import empty_prompt_tokens


def _extract_img_temporal(packed_pos: torch.Tensor, mod_mask: torch.Tensor) -> torch.Tensor:
    img_pos = packed_pos[mod_mask == 1.0]
    if img_pos.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.unique(img_pos[:, 0].to(torch.long))


def main() -> None:
    tok = TiktokenTokenizer()

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

    cond = tok.encode("blue", add_pad=False, add_eot=True).view(-1).long()
    uncond = empty_prompt_tokens(tok, like=cond)

    assert int(uncond.numel()) == int(cond.numel()), f"uncond len {int(uncond.numel())} != cond len {int(cond.numel())}"
    assert int(uncond[-1].item()) == int(tok.eot_token), "uncond must end with EOT"

    # Dummy latent image (model-space latents).
    img = torch.randn(16, 8, 8, dtype=torch.float32)

    # Pack separately and ensure image temporal positions match.
    _, _, pos_c, mask_c, _, _, _ = model.pack_inputs([cond], [img], timesteps=torch.zeros(1), pad=False)
    _, _, pos_u, mask_u, _, _, _ = model.pack_inputs([uncond], [img], timesteps=torch.zeros(1), pad=False)

    t_c = _extract_img_temporal(pos_c, mask_c)
    t_u = _extract_img_temporal(pos_u, mask_u)

    assert t_c.numel() > 0 and t_u.numel() > 0, "No image tokens detected in packed output."
    assert torch.equal(t_c, t_u), f"Image temporal positions differ: cond={t_c.tolist()} uncond={t_u.tolist()}"

    print("PASS: empty_prompt_tokens(..., like=cond) preserves CFG RoPE alignment.")


if __name__ == "__main__":
    main()

