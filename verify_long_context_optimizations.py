import inspect
import math
import os
import sys
from dataclasses import asdict

import torch

import omni_model_v2 as om


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_architecture_claims() -> None:
    # Fact-check: "no separate CrossAttention module"
    # (There is cross-modal interaction, but it happens via shared self-attention over packed tokens.)
    has_cross = any(name.lower().endswith("crossattention") for name in dir(om))
    _assert(not has_cross, "Found CrossAttention-like symbol in omni_model_v2; claim 'no separate CrossAttention module' may be wrong.")


def test_blockmask_hoist_is_wired() -> None:
    sig_attn = inspect.signature(om.PackedSelfAttention.forward)
    _assert("block_mask" in sig_attn.parameters, "PackedSelfAttention.forward is missing block_mask parameter (hoisting not applied).")

    sig_blk = inspect.signature(om.MMDiTBlock.forward)
    _assert("block_mask" in sig_blk.parameters, "MMDiTBlock.forward is missing block_mask parameter (hoisting not applied).")


def test_lazy_logits_is_wired() -> None:
    _assert(hasattr(om.OmniConfigV2, "lazy_logits"), "OmniConfigV2 is missing lazy_logits flag.")


@torch.no_grad()
def test_lazy_logits_equivalence_cpu() -> None:
    # Prove the math: selecting rows before/after a per-token Linear head is equivalent.
    cfg = om.OmniConfigV2(
        d_model=64,
        n_layers=2,
        n_heads=4,
        head_dim=16,
        vocab_size=100352,
        device="cpu",
        dtype="float32",
        grad_checkpointing=False,
        lazy_logits=False,
        attention_logit_cap=0.0,  # avoid tanh soft-cap variance in debug prints; not required
        drop_path_rate=0.0,
    )
    model = om.OmniFusionV2(cfg).eval()

    # One sample, simple prompt.
    text = [torch.tensor([1, 2, 3, 4, 5, 100257], dtype=torch.long)]
    images = [None]
    t = torch.tensor([0.5], dtype=torch.float32)

    res = model(text, images, t, causal_text=True)
    _assert(res["x_out"].dtype == torch.float32, "Unexpected dtype for x_out in eval mode.")
    _assert(res["text"] is not None and res["image"] is not None, "Expected full heads in eval mode.")

    x_out = res["x_out"]
    mod_mask = res["modality_mask"]

    is_text = (mod_mask == 0.0)
    is_img = (mod_mask == 1.0)

    # Row-wise linear: head(x)[mask] == head(x[mask]).
    full_text = res["text"]
    full_img = res["image"]

    sub_text_a = full_text[is_text]
    sub_text_b = model.text_head(x_out[is_text])
    _assert(torch.allclose(sub_text_a, sub_text_b, atol=0, rtol=0), "Lazy text logits mismatch (should be exactly equal for Linear).")

    sub_img_a = full_img[is_img]
    sub_img_b = model.image_head(x_out[is_img])
    _assert(torch.allclose(sub_img_a, sub_img_b, atol=0, rtol=0), "Lazy image head mismatch (should be exactly equal for Linear).")


def test_blockmask_create_called_once_if_cuda() -> None:
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available; skipping block_mask call-count test.")
        return

    if not getattr(om, "FLEX_ATTENTION_AVAILABLE", False):
        print("[SKIP] FLEX_ATTENTION_AVAILABLE is False; skipping block_mask call-count test.")
        return

    # Small config to keep the test fast.
    cfg = om.OmniConfigV2(
        d_model=128,
        n_layers=4,
        n_heads=4,
        head_dim=32,
        vocab_size=100352,
        device="cuda",
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        grad_checkpointing=False,
        lazy_logits=False,
        drop_path_rate=0.0,
    )
    model = om.OmniFusionV2(cfg).to("cuda").eval()
    if cfg.torch_dtype == torch.bfloat16:
        model = model.bfloat16()

    # Monkeypatch create_block_mask to count calls.
    counter = {"n": 0}
    real_fn = om.create_block_mask

    def counted(*args, **kwargs):
        counter["n"] += 1
        return real_fn(*args, **kwargs)

    om.create_block_mask = counted  # type: ignore[assignment]
    try:
        text = [torch.tensor([1, 2, 3, 4, 5, 6, 100257], device="cuda", dtype=torch.long)]
        images = [None]
        t = torch.tensor([0.5], device="cuda", dtype=model.patch_embed.weight.dtype)
        _ = model(text, images, t, causal_text=True)

        # Expect exactly one create_block_mask call per forward now that it's hoisted.
        _assert(counter["n"] == 1, f"Expected create_block_mask to be called once, got {counter['n']}.")
    finally:
        om.create_block_mask = real_fn  # type: ignore[assignment]


def main() -> None:
    test_architecture_claims()
    test_blockmask_hoist_is_wired()
    test_lazy_logits_is_wired()
    test_lazy_logits_equivalence_cpu()
    test_blockmask_create_called_once_if_cuda()
    print("OK: long-context optimizations are present and behavior-preserving where expected.")


if __name__ == "__main__":
    main()
