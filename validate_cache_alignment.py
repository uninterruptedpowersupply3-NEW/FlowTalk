"""
Cache Alignment Validator (tokens vs captions)
Usage:
  python validate_cache_alignment.py --data-dir "path/to/data" --cache-dir ".latent_cache"
"""
import argparse
import json
import os
import random
import struct
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from data_manager import TiktokenTokenizer
import encoder_backend as eb

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PAD_TOKEN = 100258
EOT_TOKEN = 100257


def _strip_special(tokens: np.ndarray) -> np.ndarray:
    if tokens.size == 0:
        return tokens
    mask = (tokens != PAD_TOKEN) & (tokens != EOT_TOKEN)
    return tokens[mask]


def _match_ratio(a: np.ndarray, b: np.ndarray) -> float:
    a = _strip_special(a)
    b = _strip_special(b)
    if a.size == 0 and b.size == 0:
        return 1.0
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    return float((a[:n] == b[:n]).mean())


def _read_cached_tokens(cache_dir: str, info: Dict[str, Any]) -> np.ndarray:
    shard_path = os.path.join(cache_dir, f"shard_{info['shard']:04d}.bin")
    with open(shard_path, "rb") as f:
        f.seek(info["offset"])
        latent_len = struct.unpack("<I", f.read(4))[0]
        f.read(latent_len)
        token_len = struct.unpack("<I", f.read(4))[0]
        token_bytes = f.read(token_len)
    t_dtype = np.int32 if info.get("token_dtype", "int32") == "int32" else np.int64
    return np.frombuffer(token_bytes, dtype=t_dtype).copy()


def _build_image_map(data_dir: str) -> Dict[str, str]:
    image_map: Dict[str, str] = {}
    with os.scandir(data_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            ext = Path(entry.name).suffix.lower()
            if ext in IMAGE_EXTS:
                name = entry.name
                stem = Path(entry.name).stem
                image_map.setdefault(name, entry.path)
                image_map.setdefault(stem, entry.path)
    return image_map


def _build_caption_pool_from_json(data: Dict[str, Any], fallback: str) -> List[str]:
    pool: List[str] = []

    # Florence (universal fallback)
    florence_text = ""
    try:
        fc = eb._nested_get(data, "florence.more_detailed_caption")
        if not (isinstance(fc, str) and fc.strip()):
            fc = eb._nested_get(data, "florence.caption")
        if isinstance(fc, str) and fc.strip():
            florence_text = fc.strip()
    except Exception:
        pass

    if florence_text:
        for p in eb._DESC_PROMPTS:
            pool.append(eb._chatml(p, florence_text))

    # WD Tagger
    try:
        wd = eb._nested_get(data, "wd_tagger.caption")
        if isinstance(wd, str) and wd.strip():
            for p in eb._TAG_PROMPTS:
                pool.append(eb._chatml(p, wd.strip()))
    except Exception:
        pass

    # BLIP
    try:
        bc = eb._nested_get(data, "blip.caption")
        if isinstance(bc, str) and bc.strip():
            ba = eb._nested_get(data, "blip.answer")
            if isinstance(ba, str) and ba.strip():
                q = eb._nested_get(data, "question_used_for_image") or "What do you see?"
                pool.append(eb._chatml(q, ba.strip()))
            else:
                for p in eb._DESC_PROMPTS:
                    pool.append(eb._chatml(p, bc.strip()))
    except Exception:
        pass

    # OCR
    try:
        ocr = eb._nested_get(data, "florence.ocr_with_region")
        if isinstance(ocr, dict):
            labels = ocr.get("labels")
            if isinstance(labels, list):
                clean = [str(l).strip() for l in labels if isinstance(l, str) and l.strip()]
                if clean:
                    joined = ", ".join(clean)
                    for p in eb._OCR_PROMPTS:
                        pool.append(eb._chatml(p, joined))
    except Exception:
        pass

    # VQA (SmolVLM)
    try:
        pairs = eb._nested_get(data, "smolvlm.qa_pairs")
        if isinstance(pairs, list) and pairs:
            valid_pairs = [
                qa for qa in pairs
                if isinstance(qa, dict)
                and isinstance(qa.get("question"), str) and qa["question"].strip()
                and isinstance(qa.get("answer"), str) and qa["answer"].strip()
            ]
            for qa in valid_pairs:
                pool.append(eb._chatml(qa["question"].strip(), qa["answer"].strip()))
    except Exception:
        pass

    # Object Detection
    try:
        od = eb._nested_get(data, "florence.od")
        if not isinstance(od, dict):
            od = eb._nested_get(data, "florence.dense_region_caption")
        if isinstance(od, dict):
            labels = od.get("labels")
            if isinstance(labels, list):
                seen = set()
                unique = []
                for l in labels:
                    s = str(l).strip() if isinstance(l, str) else ""
                    if s and s not in seen:
                        seen.add(s)
                        unique.append(s)
                if unique:
                    joined = ", ".join(unique)
                    for p in eb._OD_PROMPTS:
                        pool.append(eb._chatml(p, joined))
    except Exception:
        pass

    # Fallback if nothing found
    if not pool:
        for k in ("caption", "text"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                for p in eb._DESC_PROMPTS:
                    pool.append(eb._chatml(p, v.strip()))
                break

    if not pool:
        pool = [eb._chatml("Describe this image.", fallback)]

    # Deduplicate while preserving order
    deduped: List[str] = []
    seen = set()
    for s in pool:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def _get_caption_candidates(img_path: Path, sample_name: str) -> List[str]:
    txt_path = img_path.with_suffix(".txt")
    if txt_path.exists():
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return [text]
        except OSError:
            pass

    json_path = img_path.with_suffix(".json")
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                return _build_caption_pool_from_json(data, fallback=f"image of {sample_name}")
        except (OSError, json.JSONDecodeError):
            pass

    return [f"image of {sample_name}"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cache token/caption alignment")
    parser.add_argument("--data-dir", required=True, help="Image+caption directory")
    parser.add_argument("--cache-dir", default=".latent_cache", help="Latent cache directory")
    parser.add_argument("--samples", type=int, default=50, help="Number of cache entries to validate")
    parser.add_argument("--min-match", type=float, default=0.7, help="Min token match ratio to consider aligned")
    parser.add_argument("--max-text-length", type=int, default=512, help="Tokenizer max length (should match cache)")
    args = parser.parse_args()

    index_path = os.path.join(args.cache_dir, "index.json")
    if not os.path.exists(index_path):
        print(f"ERROR: Cache index not found at {index_path}")
        return

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    print(f"Cache has {len(index)} entries.")

    image_map = _build_image_map(args.data_dir)
    if not image_map:
        print(f"No images found in {args.data_dir}")
        return

    keys = list(index.keys())
    if not keys:
        print("Cache index is empty.")
        return

    random.shuffle(keys)
    keys = keys[: min(len(keys), args.samples)]

    tokenizer = TiktokenTokenizer(max_length=args.max_text_length)

    checked = 0
    mismatches = 0
    missing_images = 0

    for key in keys:
        info = index.get(key)
        if not info:
            continue

        img_path = image_map.get(key)
        if not img_path:
            stem = Path(key).stem
            img_path = image_map.get(stem)

        if not img_path:
            missing_images += 1
            continue

        try:
            cached_tokens = _read_cached_tokens(args.cache_dir, info)
        except Exception as e:
            print(f"  READ ERROR: {key} -> {e}")
            continue

        candidates = _get_caption_candidates(Path(img_path), Path(img_path).stem)
        best_ratio = 0.0
        best_text = ""
        matched = False

        for text in candidates:
            tokens = tokenizer.encode(text, max_length=args.max_text_length, add_pad=False).numpy()
            ratio = _match_ratio(cached_tokens, tokens)
            if ratio > best_ratio:
                best_ratio = ratio
                best_text = text
            if ratio >= args.min_match:
                matched = True
                break

        checked += 1

        if not matched:
            mismatches += 1
            try:
                cached_preview = tokenizer.decode(cached_tokens[:20].tolist())
            except Exception:
                cached_preview = str(cached_tokens[:20].tolist())
            print(f"  MISMATCH: {key} (best_ratio={best_ratio:.2f}, candidates={len(candidates)})")
            print(f"     Cache: '{cached_preview}...'")
            print(f"     Best:  '{best_text[:80]}...'")
        elif checked <= 3:
            print(f"  OK: {key} (best_ratio={best_ratio:.2f})")

    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Checked:        {checked}")
    print(f"  Mismatches:     {mismatches}")
    print(f"  Missing images: {missing_images}")

    if mismatches == 0 and checked > 0:
        print("\nOK: Cache appears aligned with current captions.")
    elif mismatches > 0:
        print(f"\nFAIL: {mismatches}/{checked} samples have low token match.")
        print("  Consider rebuilding cache if mismatches are consistently high.")


if __name__ == "__main__":
    main()
