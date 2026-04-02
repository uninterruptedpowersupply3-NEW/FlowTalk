import torch
import torch.nn.functional as F


def _force_sdpa_path(om):
    # Force the SDPA fallback path so we can capture the constructed attn_mask.
    om.FLEX_ATTENTION_AVAILABLE = False
    om.FLASH_ATTN_AVAILABLE = False
    om.XFORMERS_AVAILABLE = False


def main():
    import omni_model_v2 as om

    _force_sdpa_path(om)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This verification is intended to run on CUDA (to match the training/inference path).")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Small deterministic attention module.
    cfg = om.OmniConfigV2(
        d_model=32,
        n_heads=4,
        n_kv_heads=4,
        head_dim=8,
        qk_norm=False,
        attention_logit_cap=0.0,  # ensure we hit the SDPA(attn_mask=...) branch (not manual soft-cap)
    )
    attn = om.PackedSelfAttention(cfg).to(device=device, dtype=dtype).eval()

    # Sequence: 3 text tokens then 3 image tokens.
    L = 6
    x = torch.randn(L, cfg.d_model, device=device, dtype=dtype)
    positions = torch.zeros(L, 3, device=device, dtype=torch.long)
    positions[:, 0] = torch.arange(L, device=device, dtype=torch.long)  # monotonically increasing "t" position

    # B=1 packed boundaries.
    cu_seqlens = torch.tensor([0, L], device=device, dtype=torch.int32)
    max_seqlen = torch.tensor(L, device=device, dtype=torch.int32)

    mod_mask = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=device, dtype=dtype)

    # Monkeypatch SDPA to capture attn_mask.
    captured = {}
    orig_sdpa = F.scaled_dot_product_attention

    def _wrapped_sdpa(q, k, v, attn_mask=None, *args, **kwargs):
        captured["attn_mask"] = attn_mask
        return orig_sdpa(q, k, v, attn_mask=attn_mask, *args, **kwargs)

    F.scaled_dot_product_attention = _wrapped_sdpa
    try:
        _ = attn(
            x,
            rope_func=lambda q, k, pos: (q, k),  # no-op RoPE for mask verification
            positions=positions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            doc_ids=None,
            causal=True,
            mod_mask=mod_mask,
            kv_cache=None,
            layer_idx=0,
            block_mask=None,
        )
    finally:
        F.scaled_dot_product_attention = orig_sdpa

    attn_mask = captured.get("attn_mask")
    assert attn_mask is not None, "Did not capture an SDPA attn_mask; SDPA path may not have been used."
    assert attn_mask.shape == (L, L), f"Expected attn_mask shape {(L, L)}, got {tuple(attn_mask.shape)}"
    assert attn_mask.dtype == dtype, f"Expected attn_mask dtype {dtype}, got {attn_mask.dtype}"

    # Expectations:
    # - Text->Text future masked: (i<j) for i,j in {0,1,2}
    # - Text->Image masked: i in {0,1,2}, j in {3,4,5}
    # - Image->Text NOT masked
    # - Image->Image NOT masked
    text = [0, 1, 2]
    img = [3, 4, 5]

    def _is_neginf(v):
        return torch.isinf(v) & (v < 0)

    for i in text:
        for j in text:
            if j > i:
                assert _is_neginf(attn_mask[i, j]), f"Expected Text->future-Text masked at ({i},{j})"
            else:
                assert not _is_neginf(attn_mask[i, j]), f"Unexpected -inf at Text->Text ({i},{j})"

    for i in text:
        for j in img:
            assert _is_neginf(attn_mask[i, j]), f"Expected Text->Image masked at ({i},{j})"

    for i in img:
        for j in text:
            assert not _is_neginf(attn_mask[i, j]), f"Unexpected Image->Text mask at ({i},{j})"

    for i in img:
        for j in img:
            assert not _is_neginf(attn_mask[i, j]), f"Unexpected Image->Image mask at ({i},{j})"

    print("OK: SDPA fallback mask blocks Text->Image and enforces causal Text->Text.")


if __name__ == "__main__":
    main()

