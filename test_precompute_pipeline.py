"""
Precompute Pipeline Verification Script

Purpose:
  Prove that the precompute scripts do NOT corrupt latents and that metadata
  generation is structurally correct.

This test is intentionally aggressive because a broken cache permanently poisons
training until the cache is rebuilt.
"""

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

results = []


def _check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")


def _snap_up(x: int, block: int) -> int:
    return max(block, ((int(x) + block - 1) // block) * block)


def _make_temp_dataset(data_dir: Path) -> dict:
    """
    Create two images that are NOT multiples of 16 so we can verify grid snapping.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    items = []
    # (name, w, h, color)
    specs = [
        ("img_a", 250, 250, (255, 0, 0)),
        ("img_b", 191, 127, (0, 0, 255)),
    ]
    for name, w, h, color in specs:
        img = Image.new("RGB", (w, h), color=color)
        d = ImageDraw.Draw(img)
        d.rectangle([w // 4, h // 4, (3 * w) // 4, (3 * h) // 4], outline=(0, 255, 0), width=3)
        img_path = data_dir / f"{name}.png"
        txt_path = data_dir / f"{name}.txt"
        img.save(img_path)
        txt_path.write_text(f"{name} caption", encoding="utf-8")
        items.append({"name": name, "w": w, "h": h, "img_path": img_path})

    return {"items": items}


def _read_cached_latents(cache_dir: Path, info: dict) -> np.ndarray:
    shard_path = cache_dir / f"shard_{int(info['shard']):04d}.bin"
    with open(shard_path, "rb") as f:
        f.seek(int(info["offset"]))
        latent_len = struct.unpack("<I", f.read(4))[0]
        latent_bytes = f.read(latent_len)
        token_len = struct.unpack("<I", f.read(4))[0]
        f.read(token_len)

    dtype = np.float16 if info.get("latent_dtype") == "float16" else np.float32
    arr = np.frombuffer(latent_bytes, dtype=dtype).copy()
    arr = arr.reshape(info["latent_shape"])
    return arr


def _aligned_target_tensor(img_path: Path, *, max_size: int, block_size: int) -> torch.Tensor:
    """
    Replicate encoder_backend.PrecomputeDataset._resize_native for scale<=1
    images. Our synthetic images are smaller than max_size so only snapping
    matters (scale-up is not performed).
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        src_w, src_h = img.size

        # PrecomputeDataset only scales DOWN, never up.
        scale = min(max_size / src_w, max_size / src_h)
        if scale < 1.0:
            new_w = max(1, int(src_w * scale))
            new_h = max(1, int(src_h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            new_w, new_h = src_w, src_h

        target_w = _snap_up(new_w, block_size)
        target_h = _snap_up(new_h, block_size)
        if target_w != new_w or target_h != new_h:
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        arr = np.array(img, dtype=np.float32)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return tensor


def _aligned_uint8_tensor(img_path: Path, *, max_size: int, block_size: int) -> torch.Tensor:
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        src_w, src_h = img.size

        scale = min(max_size / src_w, max_size / src_h)
        if scale < 1.0:
            new_w = max(1, int(src_w * scale))
            new_h = max(1, int(src_h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            new_w, new_h = src_w, src_h

        target_w = _snap_up(new_w, block_size)
        target_h = _snap_up(new_h, block_size)
        if target_w != new_w or target_h != new_h:
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        arr_u8 = np.array(img, dtype=np.uint8)
        return torch.from_numpy(arr_u8).permute(2, 0, 1).unsqueeze(0).contiguous()


def run_all_tests() -> bool:
    # Enforce offline loads: the codebase should be able to run without network
    # once models are cached locally.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print("=" * 70)
    print("PRECOMPUTE PIPELINE VERIFICATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    tmp_root = ROOT / ".tmp_precompute_pipeline"
    data_dir = tmp_root / "data"
    cache_dir = tmp_root / "cache"
    meta_path = tmp_root / "dataset_metadata.json"

    shutil.rmtree(tmp_root, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds = _make_temp_dataset(data_dir)

    print("\n[TEST GROUP 1] precompute_metadata.py Smoke Test")
    # Run the script as a subprocess to validate CLI + I/O.
    subprocess.run(
        [sys.executable, str(ROOT / "precompute_metadata.py"), "--data-dir", str(data_dir), "--output", str(meta_path), "--workers", "2"],
        check=True,
    )
    _check("Metadata JSON created", meta_path.exists(), f"path={meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _check("Metadata has version", meta.get("version") == 2, f"version={meta.get('version')}")
    _check("Metadata includes images dict", isinstance(meta.get("images"), dict), f"type={type(meta.get('images'))}")
    _check("Metadata indexed 2 images", meta.get("total_images") == 2, f"total_images={meta.get('total_images')}")

    for item in ds["items"]:
        rel = f"{item['name']}.png"
        info = meta["images"].get(rel)
        _check(f"Metadata entry exists for {rel}", isinstance(info, dict))
        if isinstance(info, dict):
            _check(f"Metadata raw dims for {rel}", info.get("w") == item["w"] and info.get("h") == item["h"], f"got=({info.get('w')},{info.get('h')})")

    print("\n[TEST GROUP 2] encoder_backend _encode_eager Must Not Re-Affine")
    # Numeric proof: with identical RNG state, FastMultimodalEncoder._encode_eager()
    # must match FluxVAE.encode() exactly.
    from encoder_backend import EncoderRuntimeConfig, FastMultimodalEncoder

    enc_cfg = EncoderRuntimeConfig(
        cache_dir=str(cache_dir),
        batch_size=1,
        max_size=256,
        keep_aspect_ratio=True,
        num_workers=0,
        use_compile=False,
        use_cuda_graphs=False,
    )
    enc = FastMultimodalEncoder(enc_cfg)
    enc._ensure_vae()
    vae = enc.vae
    vae = vae.to(DEVICE)
    vae.eval()

    # Load one processed image tensor as uint8 (same as PrecomputeDataset would emit).
    max_size = 256
    block_size = 16
    sample_img = ds["items"][0]["img_path"]
    batch_u8 = _aligned_uint8_tensor(sample_img, max_size=max_size, block_size=block_size)
    x = batch_u8.to(DEVICE, dtype=torch.float32) / 127.5 - 1.0

    with torch.no_grad():
        torch.manual_seed(123)
        lat_direct = vae.encode(x.to(enc.vae_dtype))
        torch.manual_seed(123)
        lat_enc = enc._encode_eager(batch_u8)

    diff = (lat_direct - lat_enc).abs().float()
    _check("encode_eager equals FluxVAE.encode (seeded)", diff.max().item() == 0.0, f"max_diff={diff.max().item():.8f}")

    print("\n[TEST GROUP 3] precompute_latents.py Cache Latents Decode Sanity")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "precompute_latents.py"),
            "--data-dir",
            str(data_dir),
            "--cache-dir",
            str(cache_dir),
            "--batch-size",
            "2",
            "--max-size",
            "256",
            "--workers",
            "0",
            "--prefetch-factor",
            "2",
            "--flush-every",
            "1",
            "--shard-size-gb",
            "0.1",
        ],
        check=True,
    )
    index_path = cache_dir / "index.json"
    _check("Cache index.json created", index_path.exists(), f"path={index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    _check("Cache has 2 entries", len(index) == 2, f"len={len(index)}")

    max_size = 256
    block_size = 16

    for item in ds["items"]:
        info = index.get(item["name"])
        _check(f"Cache entry exists for {item['name']}", isinstance(info, dict))
        if not isinstance(info, dict):
            continue

        # Assert snapping was applied by the encoder backend.
        exp_w = _snap_up(item["w"], block_size)
        exp_h = _snap_up(item["h"], block_size)
        _check(f"Cache stores snapped dims for {item['name']}", info.get("w") == exp_w and info.get("h") == exp_h, f"got=({info.get('w')},{info.get('h')}) exp=({exp_w},{exp_h})")

        lat_np = _read_cached_latents(cache_dir, info)
        _check(f"Cached latent shape matches index for {item['name']}", list(lat_np.shape) == list(info["latent_shape"]), f"shape={lat_np.shape}")

        lat = torch.from_numpy(lat_np).unsqueeze(0).to(DEVICE, dtype=torch.float32)
        with torch.no_grad():
            recon = vae.decode(lat).float()

        target = _aligned_target_tensor(item["img_path"], max_size=max_size, block_size=block_size).to(DEVICE, dtype=torch.float32)
        mse = F.mse_loss(recon, target).item()
        # Broken affine caches typically jump to ~0.07+ MSE on this synthetic image.
        _check(f"Cached latent decodes close to input for {item['name']}", mse < 0.05, f"MSE={mse:.6f}")

    shutil.rmtree(tmp_root, ignore_errors=True)

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


def test_precompute_pipeline() -> None:
    """Pytest entrypoint: run the full precompute pipeline verification."""
    assert run_all_tests()


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
