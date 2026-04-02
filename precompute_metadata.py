#!/usr/bin/env python3
"""
Build a lightweight metadata index for faster training startup.
Stores per-image (h, w, file_size) keyed by relative path.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

from tqdm import tqdm

try:
    import imagesize
except ImportError:
    imagesize = None

from encoder_backend import IMAGE_EXTENSIONS, autodetect_loader_workers


def _read_image_meta(path: Path) -> Optional[Tuple[str, Dict[str, int]]]:
    try:
        if imagesize is not None:
            w, h = imagesize.get(str(path))
        else:
            from PIL import Image

            with Image.open(path) as img:
                w, h = img.size
        if not w or not h:
            return None

        return (
            str(path),
            {
                "h": int(h),
                "w": int(w),
                "size": int(path.stat().st_size),
            },
        )
    except Exception:
        return None


def build_metadata_index(data_dir: Path, workers: int) -> Dict[str, Dict[str, int]]:
    files = []
    for root, _, names in os.walk(data_dir):
        for name in names:
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                files.append(Path(root) / name)

    metadata: Dict[str, Dict[str, int]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(_read_image_meta, files)
        for item in tqdm(results, total=len(files), desc="Metadata"):
            if not item:
                continue
            abs_path, info = item
            rel = str(Path(abs_path).relative_to(data_dir)).replace("\\", "/")
            metadata[rel] = info

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute image metadata to speed dataset initialization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", required=True, help="Image root directory")
    parser.add_argument(
        "--output",
        default="",
        help="Output metadata JSON path (defaults to <data-dir>/dataset_metadata.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Thread count for header reads (auto if omitted)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    workers = autodetect_loader_workers(args.workers, reserve_cores=0, max_workers=64)
    index = build_metadata_index(data_dir, workers)

    payload = {
        "version": 2,
        "data_dir": str(data_dir),
        "total_images": len(index),
        "images": index,
    }

    out_path = Path(args.output).resolve() if args.output else (data_dir / "dataset_metadata.json")
    if out_path.suffix.lower() != ".json":
        out_path = out_path.with_suffix(".json")
    out_path.write_text(json.dumps(payload), encoding="utf-8")

    print(f"Metadata index written: {out_path}")
    print(f"Images indexed       : {len(index):,}")
    print(f"Worker threads       : {workers}")


if __name__ == "__main__":
    main()
