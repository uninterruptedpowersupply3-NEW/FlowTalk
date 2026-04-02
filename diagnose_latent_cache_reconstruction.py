"""
Diagnose Latent Cache Reconstruction Quality

This script verifies whether a given `.latent_cache` is *numerically consistent*
with FluxVAE's expected latent normalization by decoding cached latents back to
RGB and comparing against the resize+snap preprocessed source images.

Why this matters:
  If the cache latents were written with an incorrect affine transform, training
  will be permanently poisoned until the cache is rebuilt.
"""

import argparse
import json
import os
import random
import struct
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def _snap_up(x: int, block: int) -> int:
    return max(block, ((int(x) + block - 1) // block) * block)


def _resize_native(img: Image.Image, *, max_size: int, block_size: int) -> Image.Image:
    # Mirrors encoder_backend.PrecomputeDataset._resize_native
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        sz = max(block_size, max_size)
        return Image.new("RGB", (sz, sz), (0, 0, 0))

    scale = min(max_size / src_w, max_size / src_h)
    if scale < 1.0:
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    else:
        new_w, new_h = src_w, src_h

    tgt_w = _snap_up(new_w, block_size)
    tgt_h = _snap_up(new_h, block_size)
    if tgt_w != new_w or tgt_h != new_h:
        img = img.resize((tgt_w, tgt_h), Image.Resampling.LANCZOS)
    return img


def _build_image_lookup(data_dir: Path) -> Dict[str, List[Path]]:
    """
    Build a robust lookup to map cache keys -> image paths.

    We index by:
      - relative path (posix):  subdir/img.png
      - basename:              img.png
      - stem:                  img

    Values are lists to detect ambiguity (duplicate stems in different folders).
    """
    lookup: Dict[str, List[Path]] = {}
    exts = {e.lower() for e in IMAGE_EXTS}

    for root, _, files in os.walk(data_dir):
        for fname in files:
            p = Path(root) / fname
            if p.suffix.lower() not in exts:
                continue

            try:
                rel = p.relative_to(data_dir).as_posix()
            except Exception:
                rel = p.name

            keys = {rel, p.name, p.stem}
            for k in keys:
                if not k:
                    continue
                lst = lookup.get(k)
                if lst is None:
                    lookup[k] = [p]
                else:
                    # Avoid duplicates in case multiple keys map to same path.
                    if p not in lst:
                        lst.append(p)

    return lookup


def _lookup_image(lookup: Dict[str, List[Path]], cache_key: str) -> Optional[Path]:
    """
    Resolve a cache index key to an image path.

    Handles keys that may be stems ("4929531"), basenames ("4929531.jpg"),
    or relative paths ("subdir/4929531.jpg").
    """
    key = str(cache_key)
    variants = []
    variants.append(key)
    variants.append(key.replace("\\", "/"))
    variants.append(Path(key).name)
    variants.append(Path(key).stem)
    variants.append(Path(key.replace("\\", "/")).name)
    variants.append(Path(key.replace("\\", "/")).stem)

    for v in variants:
        lst = lookup.get(v)
        if not lst:
            continue
        if len(lst) == 1:
            return lst[0]
        # Ambiguous match: multiple images share the same key (usually stem).
        # Skip to avoid falsely comparing against the wrong image.
        return None

    return None


def _read_latents(cache_dir: Path, info: dict) -> torch.Tensor:
    shard_path = cache_dir / f"shard_{int(info['shard']):04d}.bin"
    with open(shard_path, "rb") as f:
        f.seek(int(info["offset"]))
        n = struct.unpack("<I", f.read(4))[0]
        latent_bytes = f.read(n)
    dtype = np.float16 if info.get("latent_dtype") == "float16" else np.float32
    arr = np.frombuffer(latent_bytes, dtype=dtype).copy().reshape(info["latent_shape"])
    return torch.from_numpy(arr).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode cached latents and compute reconstruction MSE")
    parser.add_argument("--cache-dir", required=True, help="Path to .latent_cache folder (contains index.json + shard_*.bin)")
    parser.add_argument("--data-dir", required=True, help="Path to dataset folder (images with matching stems)")
    parser.add_argument("--samples", type=int, default=32, help="Number of cache entries to test")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling cache entries")
    parser.add_argument("--max-size", type=int, default=256, help="Max resize size used during caching/training")
    parser.add_argument("--block-size", type=int, default=16, help="Snap size (vae_downsample * patch_size)")
    parser.add_argument("--fail-mse", type=float, default=0.05, help="Fail threshold for average MSE")
    args = parser.parse_args()

    # Prefer offline loads: if the model isn't already cached locally, this will fail loudly.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    cache_dir = Path(args.cache_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    index_path = cache_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing cache index: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    keys = list(index.keys())
    if not keys:
        raise RuntimeError("Cache index is empty.")

    image_lookup = _build_image_lookup(data_dir)
    if not image_lookup:
        raise RuntimeError(f"No images found under data-dir: {data_dir}")

    # Resolve cache keys -> image paths first so we don't waste time loading the VAE
    # when the user accidentally points to the wrong dataset directory.
    resolved: Dict[str, Path] = {}
    for k in keys:
        p = _lookup_image(image_lookup, str(k))
        if p is not None:
            resolved[str(k)] = p

    if not resolved:
        # Print a few example keys to make the mismatch obvious.
        example = "\n  - " + "\n  - ".join([str(k) for k in keys[: min(10, len(keys))]])
        raise RuntimeError(
            "No cache keys could be matched to images under --data-dir.\n\n"
            "This usually means you are pointing at a cache built for a different dataset, "
            "or your cache keys are not stems/paths that exist under --data-dir.\n\n"
            f"Example cache keys:{example}"
        )

    random.seed(args.seed)
    key_list = list(resolved.keys())
    random.shuffle(key_list)
    key_list = key_list[: min(len(key_list), int(args.samples))]

    from vae_module import FluxVAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = FluxVAE(dtype=torch.float32).to(device)
    vae.eval()

    mses = []
    tested = 0
    missing = 0
    missing_keys: List[str] = []

    for stem in key_list:
        info = index.get(stem)
        if not isinstance(info, dict):
            continue
        img_path = resolved.get(stem)
        if img_path is None:
            missing += 1
            if len(missing_keys) < 10:
                missing_keys.append(str(stem))
            continue

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = _resize_native(img, max_size=int(args.max_size), block_size=int(args.block_size))
            arr = np.array(img, dtype=np.float32)
            target = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

        lat = _read_latents(cache_dir, info).to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)

        with torch.no_grad():
            recon = vae.decode(lat).float()

        mse = F.mse_loss(recon, target).item()
        mses.append(mse)
        tested += 1

        print(f"{stem}: h,w=({info.get('h')},{info.get('w')}) latent={info.get('latent_shape')} mse={mse:.6f}")

    if tested == 0:
        hint = ""
        if missing_keys:
            hint = "\nExamples of missing/ambiguous keys:\n  - " + "\n  - ".join(missing_keys)
        raise RuntimeError(
            f"No samples could be tested. Missing or ambiguous images for {missing} sampled cache keys."
            f"{hint}\n\nCheck that --data-dir matches the dataset used to build --cache-dir."
        )

    mses_sorted = sorted(mses)
    avg = sum(mses) / len(mses)
    p50 = mses_sorted[len(mses_sorted) // 2]
    p90 = mses_sorted[int(len(mses_sorted) * 0.9) - 1]
    p99 = mses_sorted[int(len(mses_sorted) * 0.99) - 1] if len(mses_sorted) > 1 else mses_sorted[0]

    print("\n=== Summary ===")
    print(f"Tested: {tested} (missing images: {missing})")
    print(f"MSE avg: {avg:.6f}")
    print(f"MSE p50: {p50:.6f}")
    print(f"MSE p90: {p90:.6f}")
    print(f"MSE p99: {p99:.6f}")

    if avg > float(args.fail_mse):
        raise SystemExit(
            f"\nFAIL: average MSE {avg:.6f} > {args.fail_mse}. "
            "This strongly indicates latent-cache corruption. Rebuild the cache."
        )

    print("\nPASS: cache reconstruction is within threshold.")


if __name__ == "__main__":
    main()
