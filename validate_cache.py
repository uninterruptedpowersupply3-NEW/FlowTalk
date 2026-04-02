"""
Cache Alignment Validator
Checks that cached tokens match the actual caption files in your dataset.
Usage: python validate_cache.py --data-dir "path/to/99k/images" --cache-dir ".latent_cache"
"""
import json, os, sys, struct, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_manager import TiktokenTokenizer

def main():
    parser = argparse.ArgumentParser(description="Validate cache token/image alignment")
    parser.add_argument("--data-dir", required=True, help="Path to image+caption directory")
    parser.add_argument("--cache-dir", default=".latent_cache", help="Path to latent cache")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to validate")
    args = parser.parse_args()

    tokenizer = TiktokenTokenizer()
    index_path = os.path.join(args.cache_dir, "index.json")
    
    if not os.path.exists(index_path):
        print(f"ERROR: Cache index not found at {index_path}")
        return
    
    with open(index_path, "r") as f:
        index = json.load(f)
    
    print(f"Cache has {len(index)} entries.")
    
    # Build a map of image stems -> caption files
    caption_exts = ['.txt', '.caption', '.xml']
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    
    caption_map = {}
    for root, dirs, files in os.walk(args.data_dir):
        for fname in files:
            stem, ext = os.path.splitext(fname)
            if ext.lower() in image_exts:
                for cext in caption_exts:
                    cap_path = os.path.join(root, stem + cext)
                    if os.path.exists(cap_path):
                        caption_map[stem] = cap_path
                        break
    
    print(f"Found {len(caption_map)} image-caption pairs in {args.data_dir}")
    
    if not caption_map:
        print("No caption pairs found. Make sure captions exist as .txt files alongside images.")
        return
    
    # Validate alignment
    mismatches = 0
    checked = 0
    missing_in_cache = 0
    
    items = list(caption_map.items())
    import random
    random.shuffle(items)
    items = items[:args.samples]
    
    for stem, cap_path in items:
        info = index.get(stem)
        if not info:
            missing_in_cache += 1
            continue
        
        shard_path = os.path.join(args.cache_dir, f"shard_{info['shard']:04d}.bin")
        if not os.path.exists(shard_path):
            print(f"  MISSING SHARD: {shard_path}")
            continue
        
        try:
            with open(shard_path, "rb") as f:
                f.seek(info["offset"])
                latent_len = struct.unpack("<I", f.read(4))[0]
                f.read(latent_len)
                token_len = struct.unpack("<I", f.read(4))[0]
                token_bytes = f.read(token_len)
            
            t_dtype = np.int32 if info.get("token_dtype", "int32") == "int32" else np.int64
            cached_tokens = np.frombuffer(token_bytes, dtype=t_dtype).copy()
        except Exception as e:
            print(f"  READ ERROR: {stem} -> {e}")
            continue
        
        with open(cap_path, "r", encoding="utf-8", errors="ignore") as f:
            caption_text = f.read().strip()
        
        current_tokens = tokenizer.encode(caption_text, add_pad=False).numpy()
        
        # Compare core content (strip special tokens)
        PAD = 100258
        EOT = 100257
        cached_core = cached_tokens[(cached_tokens != PAD) & (cached_tokens != EOT)]
        current_core = current_tokens[(current_tokens != PAD) & (current_tokens != EOT)]
        
        min_len = min(len(cached_core), len(current_core))
        if min_len == 0:
            matches = len(cached_core) == 0 and len(current_core) == 0
        else:
            matches = np.array_equal(cached_core[:min_len], current_core[:min_len])
        
        if not matches:
            mismatches += 1
            try:
                cached_decoded = tokenizer.decode(cached_tokens[:20].tolist())
            except:
                cached_decoded = f"[{cached_tokens[:5].tolist()}...]"
            print(f"  ❌ MISMATCH: {stem}")
            print(f"     Cache tokens ({len(cached_tokens)}): '{cached_decoded}...'")
            print(f"     Actual caption: '{caption_text[:80]}...'")
        else:
            if checked < 3:
                print(f"  ✅ OK: {stem} ({len(cached_core)} tokens)")
        
        checked += 1
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Checked:    {checked}")
    print(f"  Mismatches: {mismatches}")
    print(f"  Not cached: {missing_in_cache}")
    print(f"  Coverage:   {len(index)}/{len(caption_map)} ({100*len(index)/max(1,len(caption_map)):.1f}%)")
    
    if mismatches == 0 and checked > 0:
        print("\n✅ Cache appears aligned with current captions.")
    elif mismatches > 0:
        print(f"\n❌ {mismatches}/{checked} samples have stale/wrong tokens!")
        print("   → Rebuild cache: python precompute_latents.py --data-dir <your_dir>")
    
    if missing_in_cache > args.samples * 0.5:
        print(f"\n⚠️  {missing_in_cache}/{args.samples} sampled images missing from cache.")
        print(f"   Cache has {len(index)} entries but dataset has {len(caption_map)} images.")
        print("   → Cache is INCOMPLETE. Rebuild for full dataset coverage.")

if __name__ == "__main__":
    main()
