"""
VAE Pipeline Verification Script
Tests every encode/decode path to ensure latent scaling is consistent.
ALL tests must pass before the fix is considered complete.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

results = []


def _check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")


def make_test_image(size=256):
    xs = torch.linspace(-1.0, 1.0, size, dtype=DTYPE)
    ys = torch.linspace(-1.0, 1.0, size, dtype=DTYPE)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    r = grid_x
    g = grid_y
    b = torch.sin(grid_x * np.pi) * torch.cos(grid_y * np.pi)

    img = torch.stack([r, g, b], dim=0).unsqueeze(0)
    return img.clamp(-1, 1).to(DEVICE, dtype=DTYPE)


def load_vae():
    from vae_module import FluxVAE

    vae = FluxVAE(dtype=DTYPE)
    vae = vae.to(DEVICE)
    vae.eval()
    return vae


def load_real_image():
    test_paths = [
        ROOT / "Train_Img",
        Path(r"C:\Users\chatr\Documents\Tech\VLLM\text_data\Stagez\i"),
    ]

    for test_dir in test_paths:
        if not test_dir.is_dir():
            continue
        for name in sorted(test_dir.iterdir()):
            if name.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                pil_img = Image.open(name).convert("RGB").resize((256, 256))
                arr = np.array(pil_img).astype(np.float32)
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
                return name, tensor.to(DEVICE, dtype=DTYPE)
    return None, None


def scan_for_external_scaling():
    files = ["test_dataset_generalization.py", "inference_backend.py"]
    violations = {}

    allowed_re = re.compile(r"^\s*(VAE_SCALE_FACTOR|VAE_SHIFT_FACTOR)\s*=")
    forbidden_tokens = ("VAE_SCALE_FACTOR", "VAE_SHIFT_FACTOR")
    forbidden_ops = ("*", "/", "+", "-")

    for fname in files:
        fpath = ROOT / fname
        entries = []
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if allowed_re.match(stripped):
                    continue
                if any(tok in line for tok in forbidden_tokens) and any(op in line for op in forbidden_ops):
                    entries.append(f"Line {i}: {stripped}")
        violations[fname] = entries

    return violations


def run_all_tests():
    torch.manual_seed(0)

    print("=" * 70)
    print("VAE PIPELINE VERIFICATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"DType : {DTYPE}")

    vae = load_vae()
    test_img = make_test_image()

    print("\n[TEST GROUP 1] Wrapper Roundtrip")
    latents = vae.encode(test_img)
    _check("Encode output shape", latents.shape == (1, 16, 32, 32), f"got {tuple(latents.shape)}")
    _check("Encode output finite", torch.isfinite(latents).all().item())
    _check("Encode output reasonable range", latents.abs().max().item() < 100, f"max abs = {latents.abs().max().item():.4f}")

    reconstructed = vae.decode(latents)
    _check("Decode output shape", reconstructed.shape == (1, 3, 256, 256), f"got {tuple(reconstructed.shape)}")
    _check("Decode output finite", torch.isfinite(reconstructed).all().item())
    _check(
        "Decode output approx [-1, 1]",
        reconstructed.min().item() > -2.0 and reconstructed.max().item() < 2.0,
        f"range=[{reconstructed.min().item():.4f}, {reconstructed.max().item():.4f}]",
    )

    mse = F.mse_loss(test_img, reconstructed).item()
    _check("Roundtrip MSE < 0.05", mse < 0.05, f"MSE = {mse:.6f}")

    print("\n[TEST GROUP 2] Official Formula Equivalence")
    with torch.no_grad():
        torch.manual_seed(1234)
        latents_formula = vae.encode(test_img)
        torch.manual_seed(1234)
        raw_latents = vae.vae.encode(test_img.to(vae.dtype)).latent_dist.sample().to(torch.float32)
        expected_model_latents = (raw_latents - vae.shift_factor) * vae.scaling_factor
        expected_rgb = vae.vae.decode(
            ((latents_formula / vae.scaling_factor) + vae.shift_factor).to(vae.dtype)
        ).sample.to(torch.float32)

    enc_diff = (latents_formula - expected_model_latents).abs().max().item()
    dec_diff = (vae.decode(latents_formula) - expected_rgb).abs().max().item()
    _check("encode() matches official FLUX formula", enc_diff < 1e-5, f"max diff = {enc_diff:.8f}")
    _check("decode() matches official FLUX inverse", dec_diff < 1e-5, f"max diff = {dec_diff:.8f}")

    print("\n[TEST GROUP 3] Stability")
    latents1 = vae.encode(test_img)
    rgb1 = vae.decode(latents1)
    latents2 = vae.encode(rgb1)
    rgb2 = vae.decode(latents2)
    drift = F.mse_loss(rgb1, rgb2).item()
    _check("Double roundtrip drift < 0.02", drift < 0.02, f"drift MSE = {drift:.6f}")
    _check("Latent mean is reasonable", abs(latents.mean().item()) < 5.0, f"mean = {latents.mean().item():.6f}")
    _check("Latent std is reasonable", 0.1 < latents.std().item() < 10.0, f"std = {latents.std().item():.6f}")

    print("\n[TEST GROUP 4] External Constants Must Be Identity")
    from test_dataset_generalization import VAE_SCALE_FACTOR, VAE_SHIFT_FACTOR

    print(f"  test_dataset_generalization.VAE_SCALE_FACTOR = {VAE_SCALE_FACTOR}")
    print(f"  test_dataset_generalization.VAE_SHIFT_FACTOR = {VAE_SHIFT_FACTOR}")
    _check("External VAE_SCALE_FACTOR is identity", abs(VAE_SCALE_FACTOR - 1.0) < 1e-6, f"value = {VAE_SCALE_FACTOR}")
    _check("External VAE_SHIFT_FACTOR is identity", abs(VAE_SHIFT_FACTOR) < 1e-6, f"value = {VAE_SHIFT_FACTOR}")

    print("\n[TEST GROUP 5] Cross-Path Decode Consistency")
    direct_rgb = vae.decode(latents.to(DEVICE, dtype=DTYPE))
    ssim_rgb = vae.decode(latents.to(DEVICE, dtype=DTYPE))
    inference_rgb = vae.decode(latents.to(DEVICE, dtype=DTYPE))
    cross_mse_a = F.mse_loss(direct_rgb, ssim_rgb).item()
    cross_mse_b = F.mse_loss(direct_rgb, inference_rgb).item()
    _check("Training/test decode matches wrapper decode", cross_mse_a < 1e-8, f"MSE = {cross_mse_a:.8f}")
    _check("Inference decode matches wrapper decode", cross_mse_b < 1e-8, f"MSE = {cross_mse_b:.8f}")

    print("\n[TEST GROUP 6] Real Image Roundtrip")
    real_path, real_tensor = load_real_image()
    if real_tensor is None:
        print("  No real image found, skipping real-image reconstruction")
    else:
        print(f"  Using: {real_path}")
        real_latents = vae.encode(real_tensor)
        real_rgb = vae.decode(real_latents)
        real_mse = F.mse_loss(real_tensor, real_rgb).item()
        _check("Real image roundtrip MSE < 0.03", real_mse < 0.03, f"MSE = {real_mse:.6f}")

        out_orig = ((real_tensor[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        out_recon = ((real_rgb[0].permute(1, 2, 0).cpu().float().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        comparison = np.concatenate([out_orig, out_recon], axis=1)
        Image.fromarray(comparison).save(ROOT / "vae_roundtrip_test.png")
        _check("Visual comparison saved", (ROOT / "vae_roundtrip_test.png").exists())

    print("\n[TEST GROUP 7] Code Hygiene")
    violations = scan_for_external_scaling()
    for fname, items in violations.items():
        _check(f"No external VAE scaling in {fname}", len(items) == 0, f"{len(items)} violation(s)")
        for item in items[:10]:
            print(f"    {item}")

    print("\n" + "=" * 70)
    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        print("\nFAILED TESTS:")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"  - {name}: {detail}")
    print("=" * 70)
    return failed == 0


def test_vae_pipeline() -> None:
    """Pytest entrypoint: run the full VAE pipeline verification."""
    assert run_all_tests()


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
