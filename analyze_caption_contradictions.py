"""
Quantify internal caption contradictions in the multi-caption JSON format.

This is not a subjective "theory" check. It computes concrete mismatch rates for
simple, high-signal attributes (hair color, eye color) between caption sources.

Example contradiction pattern this script detects:
  - Florence: "... girl with long red hair ..."
  - WD tagger: "..., blue hair, ..."

Usage:
  C:\\Users\\chatr\\Documents\\Tech\\VLLM\\venv\\Scripts\\python.exe analyze_caption_contradictions.py ^
    --json-dir "C:\\Users\\chatr\\Documents\\Tech\\VLLM\\New folder\\Image Stage2" ^
    --sample 5000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


HAIR_COLORS = [
    "black",
    "blonde",
    "blue",
    "brown",
    "green",
    "grey",
    "gray",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
    "aqua",
]

EYE_COLORS = [
    "black",
    "blue",
    "brown",
    "green",
    "grey",
    "gray",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
]


def _compile_color_regex(colors: List[str], noun: str) -> re.Pattern[str]:
    # Match: "<color> <noun>" with common punctuation around it.
    # Keep this simple and robust; we want a lower-bound on contradictions.
    colors_alt = "|".join(re.escape(c) for c in colors)
    # NOTE: Use real regex escapes (\b, \s). This must NOT be double-escaped.
    return re.compile(rf"\b({colors_alt})\s+{re.escape(noun)}\b", re.IGNORECASE)


HAIR_RE = _compile_color_regex(HAIR_COLORS, "hair")
EYES_RE = _compile_color_regex(EYE_COLORS, "eyes")


def _extract_colors(text: str, pattern: re.Pattern[str]) -> Set[str]:
    if not text:
        return set()
    found = {m.group(1).lower() for m in pattern.finditer(text)}
    # normalize grey/gray
    if "grey" in found:
        found.discard("grey")
        found.add("gray")
    return found


def _get_nested(d: Dict, path: str) -> Optional[object]:
    cur: object = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _get_caption_sources(d: Dict) -> Dict[str, str]:
    out: Dict[str, str] = {}
    # WD tagger tags.
    wd = _get_nested(d, "wd_tagger.caption")
    if isinstance(wd, str) and wd.strip():
        out["wd_tagger.caption"] = wd
    # Florence detailed caption.
    flor = _get_nested(d, "florence.more_detailed_caption")
    if isinstance(flor, str) and flor.strip():
        out["florence.more_detailed_caption"] = flor
    # BLIP caption.
    blip = _get_nested(d, "blip.caption")
    if isinstance(blip, str) and blip.strip():
        out["blip.caption"] = blip
    return out


@dataclass
class Counter:
    total: int = 0
    both_present: int = 0
    mismatch: int = 0
    agree: int = 0


def _compare_sets(a: Set[str], b: Set[str]) -> Tuple[bool, bool]:
    """
    Returns (both_present, mismatch). We treat mismatch as "both non-empty and not equal".
    """
    if not a or not b:
        return (False, False)
    return (True, a != b)


def analyze(paths: Iterable[str], *, seed: int = 0, max_files: int = 0) -> Dict[str, Counter]:
    rng = random.Random(seed)
    files = list(paths)
    if max_files and max_files < len(files):
        files = rng.sample(files, max_files)

    hair = Counter()
    eyes = Counter()

    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue

        caps = _get_caption_sources(d)
        wd = caps.get("wd_tagger.caption", "")
        flor = caps.get("florence.more_detailed_caption", "")

        # Only compute contradictions between WD and Florence because those are the two primary, high-volume sources.
        wd_hair = _extract_colors(wd, HAIR_RE)
        fl_hair = _extract_colors(flor, HAIR_RE)
        wd_eyes = _extract_colors(wd, EYES_RE)
        fl_eyes = _extract_colors(flor, EYES_RE)

        hair.total += 1
        eyes.total += 1

        both, mism = _compare_sets(wd_hair, fl_hair)
        if both:
            hair.both_present += 1
            if mism:
                hair.mismatch += 1
            else:
                hair.agree += 1

        both, mism = _compare_sets(wd_eyes, fl_eyes)
        if both:
            eyes.both_present += 1
            if mism:
                eyes.mismatch += 1
            else:
                eyes.agree += 1

    return {"hair_color": hair, "eye_color": eyes}


def _iter_json_files(json_dir: str) -> List[str]:
    out: List[str] = []
    for name in os.listdir(json_dir):
        if name.lower().endswith(".json"):
            out.append(os.path.join(json_dir, name))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", type=str, required=True)
    ap.add_argument("--sample", type=int, default=5000, help="How many JSON files to sample (0 = all).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = _iter_json_files(args.json_dir)
    if not files:
        raise SystemExit(f"No .json files found in: {args.json_dir}")

    res = analyze(files, seed=int(args.seed), max_files=int(args.sample))

    print("\n=== Caption Contradiction Audit (WD vs Florence) ===")
    print(f"Dir: {os.path.abspath(args.json_dir)}")
    print(f"Files available: {len(files)} | Sampled: {min(len(files), int(args.sample) if int(args.sample) else len(files))}")

    for k, c in res.items():
        # both_present is the denominator: we only score samples where both sources mention the attribute.
        denom = max(1, c.both_present)
        mismatch_rate = 100.0 * float(c.mismatch) / float(denom)
        agree_rate = 100.0 * float(c.agree) / float(denom)
        coverage = 100.0 * float(c.both_present) / float(max(1, c.total))
        print(f"\n{k}:")
        print(f"  total_json_seen: {c.total}")
        print(f"  both_sources_mentioned_attribute: {c.both_present} ({coverage:.1f}%)")
        print(f"  agree: {c.agree} ({agree_rate:.1f}%)")
        print(f"  mismatch: {c.mismatch} ({mismatch_rate:.1f}%)")


if __name__ == "__main__":
    main()
