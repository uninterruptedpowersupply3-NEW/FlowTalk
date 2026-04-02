import torch


def _max_abs(a: torch.Tensor) -> float:
    return float(a.detach().abs().max().item()) if a.numel() else 0.0


@torch.no_grad()
def main() -> None:
    from omni_model_v2 import OmniConfigV2, OmniFusionV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cpu" else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    # Small model for fast verification (must keep vocab_size large enough for special IDs like PAD=100258).
    cfg = OmniConfigV2(
        d_model=64,
        n_layers=2,
        n_heads=4,
        head_dim=16,
        vocab_size=100352,
        patch_size=2,
        in_channels=16,
        device=str(device),
        dtype="float32" if dtype == torch.float32 else ("bfloat16" if dtype == torch.bfloat16 else "float16"),
        qk_norm=False,
        attention_logit_cap=0.0,
        drop_path_rate=0.0,
        grad_checkpointing=False,
        lazy_logits=False,
        text_pooling="mean",
        pooled_text_cond_scale=1.0,
        pooled_text_drop_prob=0.0,
    )

    model = OmniFusionV2(cfg).to(device=device).eval()
    if dtype == torch.bfloat16:
        model = model.bfloat16()
    elif dtype == torch.float16:
        model = model.half()

    # One sample: includes PAD to ensure masking works.
    PAD = 100258
    EOT = 100257
    txt = torch.tensor([11, 22, PAD, 33, EOT], device=device, dtype=torch.long)
    img = torch.randn(cfg.in_channels, 16, 16, device=device, dtype=model.patch_embed.weight.dtype)
    t = torch.tensor([0.5], device=device, dtype=model.patch_embed.weight.dtype)

    # Baseline with pooled scale=1.0
    x1, c1, pos1, m1, cu1, doc1, shapes1 = model.pack_inputs([txt], [img], t, pad=False)
    is_img1 = (m1 == 1.0)
    assert bool(is_img1.any()), "Expected at least one image token"

    # Disable pooled injection and re-pack
    model.config.pooled_text_cond_scale = 0.0
    x0, c0, pos0, m0, cu0, doc0, shapes0 = model.pack_inputs([txt], [img], t, pad=False)
    is_img0 = (m0 == 1.0)
    assert torch.equal(is_img1, is_img0), "Modality mask changed when toggling pooled_text_cond_scale"

    # Expected delta is exactly the pooled text_cond vector, broadcast over image tokens.
    txt_emb = model.text_embed(txt).to(model.patch_embed.weight.dtype)
    model.config.pooled_text_cond_scale = 1.0
    text_cond = model._compute_pooled_text_cond(txt, txt_emb, model.patch_embed.weight.dtype)
    assert text_cond is not None, "Expected text_cond with pooled_text_cond_scale=1.0"

    delta = c1[is_img1] - c0[is_img1]
    expected = text_cond.expand_as(delta)
    err = _max_abs(delta - expected)
    tol = 0.0 if delta.dtype == torch.float32 else 5e-2
    assert err <= tol, f"Pooled injection delta mismatch (max_abs_err={err} tol={tol} dtype={delta.dtype})"

    # Dropout=1.0 should always disable pooled conditioning during training.
    model.train()
    model.config.pooled_text_drop_prob = 1.0
    model.config.pooled_text_cond_scale = 1.0
    _, c_drop, _, m_drop, _, _, _ = model.pack_inputs([txt], [img], t, pad=False)
    err_drop = _max_abs(c_drop[m_drop == 1.0] - c0[is_img0])
    assert err_drop <= tol, f"Expected pooled conditioning dropped, but image c changed (max_abs_err={err_drop} tol={tol})"

    # Attention pooling mode should instantiate the pooling module and produce finite conditioning.
    cfg_attn = OmniConfigV2(
        d_model=64,
        n_layers=2,
        n_heads=4,
        head_dim=16,
        vocab_size=100352,
        patch_size=2,
        in_channels=16,
        device=str(device),
        dtype=cfg.dtype,
        qk_norm=False,
        attention_logit_cap=0.0,
        drop_path_rate=0.0,
        grad_checkpointing=False,
        lazy_logits=False,
        text_pooling="attn",
        pooled_text_cond_scale=1.0,
        pooled_text_drop_prob=0.0,
    )
    model_attn = OmniFusionV2(cfg_attn).to(device=device).eval()
    if dtype == torch.bfloat16:
        model_attn = model_attn.bfloat16()
    elif dtype == torch.float16:
        model_attn = model_attn.half()

    assert getattr(model_attn, "text_attn_pool", None) is not None, "Expected text_attn_pool to exist in attn pooling mode"
    txt_emb2 = model_attn.text_embed(txt).to(model_attn.patch_embed.weight.dtype)
    tc2 = model_attn._compute_pooled_text_cond(txt, txt_emb2, model_attn.patch_embed.weight.dtype)
    assert tc2 is not None and torch.isfinite(tc2).all(), "Attention pooled text_cond is not finite"

    print("OK: pooled text conditioning controls (scale, dropout, pooling mode) behave as expected.")


if __name__ == "__main__":
    main()
