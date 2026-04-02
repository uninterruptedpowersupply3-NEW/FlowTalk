"""
Analyze caption/token diversity from a LatentCache directory (index.json + shard_XXXX.bin).

This script is designed to *prove* whether a cached dataset has:
  - repetitive captions (low diversity)
  - low coverage of key tags (e.g. hair colors)

Usage:
  C:\\Users\\ups\\Documents\\Tech\\VLLM\\venv\\Scripts\\python.exe analyze_latent_cache_captions.py ^
    --cache-dir "C:\\Users\\ups\\Documents\\Tech\\VLLM\\New folder\\.latent_cache" ^
    --sample 1000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from data_manager import TiktokenTokenizer


HAIR_TAG_RE = re.compile(r"\\b(black|blonde|blue|brown|green|gray|grey|pink|purple|red|white|silver|orange|yellow)_hair\\b", re.I)
EYE_TAG_RE = re.compile(r"\\b(black|blue|brown|green|gray|grey|pink|purple|red|white|silver|orange|yellow)_eyes\\b", re.I)


def _read_cached_tokens(cache_dir: str, info: Dict[str, Any]) -> np.ndarray:
    shard_path = os.path.join(cache_dir, f"shard_{info['shard']:04d}.bin")
    with open(shard_path, "rb") as f:
        f.seek(int(info["offset"]))
        latent_len = struct.unpack("<I", f.read(4))[0]
        f.read(latent_len)
        token_len = struct.unpack("<I", f.read(4))[0]
        token_bytes = f.read(token_len)
    t_dtype = np.int32 if info.get("token_dtype", "int32") == "int32" else np.int64
    return np.frombuffer(token_bytes, dtype=t_dtype).copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=str, required=True)
    ap.add_argument("--sample", type=int, default=1000, help="How many entries to sample (0=all).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=40)
    args = ap.parse_args()

    cache_dir = args.cache_dir
    index_path = os.path.join(cache_dir, "index.json")
    if not os.path.exists(index_path):
        raise SystemExit(f"Missing index.json at: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    keys = list(index.keys())
    if not keys:
        raise SystemExit("Empty cache index.")

    rng = random.Random(int(args.seed))
    if int(args.sample) and int(args.sample) < len(keys):
        keys = rng.sample(keys, int(args.sample))

    tok = TiktokenTokenizer()

    caption_hashes = Counter()
    token_len_stats: List[int] = []
    hair_counts = Counter()
    eye_counts = Counter()
    word_counts = Counter()

    for k in keys:
        info = index[k]
        token_ids = _read_cached_tokens(cache_dir, info)
        token_len_stats.append(int(token_ids.size))

        # Decode. This can include special tokens; keep as-is for distribution checks.
        try:
            text = tok.decode(token_ids.tolist())
        except Exception:
            # If decode fails (rare), just skip.
            continue

        norm = " ".join(text.strip().split())
        caption_hashes[norm] += 1

        # Tag coverage (danbooru-style underscores).
        for m in HAIR_TAG_RE.finditer(norm):
            hair_counts[m.group(1).lower()] += 1
        for m in EYE_TAG_RE.finditer(norm):
            eye_counts[m.group(1).lower()] += 1

        # Simple tokenized word stats (whitespace split).
        for w in norm.lower().replace(",", " ").split():
            if len(w) <= 1:
                continue
            word_counts[w] += 1

    total = sum(caption_hashes.values())
    unique = len(caption_hashes)
    most_common_caption, most_common_n = caption_hashes.most_common(1)[0]
    uniq_ratio = unique / max(1, total)

    token_len_stats.sort()
    p50 = token_len_stats[len(token_len_stats) // 2]
    p90 = token_len_stats[int(0.9 * (len(token_len_stats) - 1))]
    p99 = token_len_stats[int(0.99 * (len(token_len_stats) - 1))]

    print("\n=== Latent Cache Caption Audit ===")
    print(f"Cache: {os.path.abspath(cache_dir)}")
    print(f"Entries sampled: {len(keys)} (index size={len(index)})")
    print(f"Unique decoded captions: {unique} ({uniq_ratio:.3f} unique/sample)")
    print(f"Most common caption frequency: {most_common_n}/{total} ({(most_common_n/max(1,total))*100:.2f}%)")
    print("Token length (cached token_ids array size):")
    print(f"  p50={p50} p90={p90} p99={p99} max={max(token_len_stats) if token_len_stats else 0}")

    print("\nTop words/tags (whitespace-split, comma removed):")
    for w, n in word_counts.most_common(int(args.topk)):
        print(f"  {w}: {n}")

    print("\nHair tag coverage (xxx_hair):")
    for c, n in hair_counts.most_common():
        print(f"  {c}_hair: {n}")
    if not hair_counts:
        print("  (none detected)")

    print("\nEye tag coverage (xxx_eyes):")
    for c, n in eye_counts.most_common():
        print(f"  {c}_eyes: {n}")
    if not eye_counts:
        print("  (none detected)")


if __name__ == "__main__":
    main()

