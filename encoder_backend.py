#!/usr/bin/env python3
"""
High-throughput multimodal encoding backend.

This module intentionally keeps image/text precompute logic separate from
training logic so it can be optimized independently.
"""

from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def autodetect_loader_workers(
    requested: Optional[int] = None,
    *,
    reserve_cores: int = 2,
    max_workers: int = 16,
) -> int:
    """Choose a practical worker count for Windows/Linux precompute."""
    cpu_count = os.cpu_count() or 1
    if requested is None:
        requested = max(1, cpu_count - reserve_cores)
    return max(0, min(int(requested), int(max_workers), cpu_count))


def _iter_image_files(root: Path) -> Iterable[Path]:
    """Fast recursive file walk using scandir."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            suffix = Path(entry.name).suffix.lower()
                            if suffix in IMAGE_EXTENSIONS:
                                yield Path(entry.path)
                    except OSError:
                        continue
        except OSError:
            continue


def _nested_get(data: Dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


import random as _rng

# Pre-computed weights for caption selection
_CAPTION_WEIGHTS = {
    "florence": 30, "wd": 30, "blip": 20, "ocr": 10, "vqa": 5, "od": 5
}

# Optional hard override for caption selection during cache precompute.
# Set OMNIFUSION_CAPTION_KEY to force a single source and avoid training on
# systematically inconsistent supervision across captioners.
#
# Examples (Windows):
#   set OMNIFUSION_CAPTION_KEY=florence
#   set OMNIFUSION_CAPTION_KEY=blip
#   set OMNIFUSION_CAPTION_KEY=wd
_CAPTION_OVERRIDE_RAW = os.environ.get("OMNIFUSION_CAPTION_KEY", "").strip().lower()
_CAPTION_OVERRIDE_ALIASES = {
    "wd": "wd",
    "wd_tagger": "wd",
    "wd_tagger.caption": "wd",
    "florence": "florence",
    "florence.more_detailed_caption": "florence",
    "florence.caption": "florence",
    "blip": "blip",
    "blip.caption": "blip",
    "ocr": "ocr",
    "florence.ocr_with_region": "ocr",
    "vqa": "vqa",
    "smolvlm": "vqa",
    "smolvlm.qa_pairs": "vqa",
    "od": "od",
    "florence.od": "od",
    "florence.dense_region_caption": "od",
}
_CAPTION_OVERRIDE = _CAPTION_OVERRIDE_ALIASES.get(_CAPTION_OVERRIDE_RAW, _CAPTION_OVERRIDE_RAW) if _CAPTION_OVERRIDE_RAW else None
if _CAPTION_OVERRIDE not in (None, "florence", "wd", "blip", "ocr", "vqa", "od"):
    _CAPTION_OVERRIDE = None

# Varied prompts per caption type to avoid overfitting to a single template
_DESC_PROMPTS = [
    "Describe this image.",
    "What do you see in this image?",
    "Describe what is shown here.",
    "What is depicted in this image?",
    "Explain this image in detail.",
    "Give a detailed description of this image.",
    "What does this picture show?",
]
_TAG_PROMPTS = [
    "List the tags for this image.",
    "What tags describe this image?",
    "Provide booru-style tags for this image.",
    "Tag this image.",
]
_OCR_PROMPTS = [
    "What text appears in this image?",
    "Read the text in this image.",
    "What text is visible here?",
]
_OD_PROMPTS = [
    "What objects are in this image?",
    "Identify the objects in this image.",
    "List the objects visible here.",
]


def _chatml(user_text: str, assistant_text: str) -> str:
    """Format a single turn in ChatML."""
    return (
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_text}<|im_end|>"
    )


def _select_caption(data: Dict[str, Any], fallback: str) -> str:
    """Fast weighted random caption selector — outputs ChatML format.
    
    Each source is wrapped in try/except — broken VQA/OCR data falls back
    to florence, then to any available caption.
    """
    # Always extract florence first as the universal fallback
    florence_text = ""
    try:
        fc = _nested_get(data, "florence.more_detailed_caption")
        if not (isinstance(fc, str) and fc.strip()):
            fc = _nested_get(data, "florence.caption")
        if isinstance(fc, str) and fc.strip():
            florence_text = fc.strip()
    except Exception:
        pass

    pool = []  # (chatml_text, weight)

    # If an override is requested, emit ONLY that caption type (with reasonable fallbacks).
    if _CAPTION_OVERRIDE is not None:
        key = _CAPTION_OVERRIDE
        if key == "florence":
            if florence_text:
                return _chatml(_rng.choice(_DESC_PROMPTS), florence_text)
            return _chatml("Describe this image.", fallback)

        if key == "wd":
            try:
                wd = _nested_get(data, "wd_tagger.caption")
                if isinstance(wd, str) and wd.strip():
                    return _chatml(_rng.choice(_TAG_PROMPTS), wd.strip())
            except Exception:
                pass
            if florence_text:
                return _chatml(_rng.choice(_DESC_PROMPTS), florence_text)
            return _chatml("Describe this image.", fallback)

        if key == "blip":
            try:
                bc = _nested_get(data, "blip.caption")
                if isinstance(bc, str) and bc.strip():
                    ba = _nested_get(data, "blip.answer")
                    if isinstance(ba, str) and ba.strip():
                        q = _nested_get(data, "question_used_for_image") or "What do you see?"
                        return _chatml(str(q), ba.strip())
                    return _chatml(_rng.choice(_DESC_PROMPTS), bc.strip())
            except Exception:
                pass
            if florence_text:
                return _chatml(_rng.choice(_DESC_PROMPTS), florence_text)
            return _chatml("Describe this image.", fallback)

        if key == "ocr":
            try:
                ocr = _nested_get(data, "florence.ocr_with_region")
                if isinstance(ocr, dict):
                    labels = ocr.get("labels")
                    if isinstance(labels, list):
                        clean = [str(l).strip() for l in labels if isinstance(l, str) and l.strip()]
                        if clean:
                            return _chatml(_rng.choice(_OCR_PROMPTS), ", ".join(clean))
            except Exception:
                pass
            if florence_text:
                return _chatml(_rng.choice(_DESC_PROMPTS), florence_text)
            return _chatml("Describe this image.", fallback)

        if key == "vqa":
            try:
                pairs = _nested_get(data, "smolvlm.qa_pairs")
                if isinstance(pairs, list) and pairs:
                    valid_pairs = [
                        qa for qa in pairs
                        if isinstance(qa, dict)
                        and isinstance(qa.get("question"), str) and qa["question"].strip()
                        and isinstance(qa.get("answer"), str) and qa["answer"].strip()
                    ]
                    if valid_pairs:
                        qa = _rng.choice(valid_pairs)
                        return _chatml(qa["question"].strip(), qa["answer"].strip())
            except Exception:
                pass
            if florence_text:
                return _chatml(_rng.choice(_DESC_PROMPTS), florence_text)
            return _chatml("Describe this image.", fallback)

        if key == "od":
            try:
                od = _nested_get(data, "florence.od")
                if not isinstance(od, dict):
                    od = _nested_get(data, "florence.dense_region_caption")
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
                            return _chatml(_rng.choice(_OD_PROMPTS), ", ".join(unique))
            except Exception:
                pass
            if florence_text:
                return _chatml(_rng.choice(_DESC_PROMPTS), florence_text)
            return _chatml("Describe this image.", fallback)

    if florence_text:
        pool.append((_chatml(_rng.choice(_DESC_PROMPTS), florence_text), 30))

    # WD Tagger (30%) — tag-style
    try:
        wd = _nested_get(data, "wd_tagger.caption")
        if isinstance(wd, str) and wd.strip():
            pool.append((_chatml(_rng.choice(_TAG_PROMPTS), wd.strip()), 30))
    except Exception:
        pass

    # BLIP (20%)
    try:
        bc = _nested_get(data, "blip.caption")
        if isinstance(bc, str) and bc.strip():
            ba = _nested_get(data, "blip.answer")
            if isinstance(ba, str) and ba.strip():
                q = _nested_get(data, "question_used_for_image") or "What do you see?"
                pool.append((_chatml(q, ba.strip()), 20))
            else:
                pool.append((_chatml(_rng.choice(_DESC_PROMPTS), bc.strip()), 20))
    except Exception:
        pass

    # OCR (10%)
    try:
        ocr = _nested_get(data, "florence.ocr_with_region")
        if isinstance(ocr, dict):
            labels = ocr.get("labels")
            if isinstance(labels, list):
                clean = [str(l).strip() for l in labels if isinstance(l, str) and l.strip()]
                if clean:
                    pool.append((_chatml(_rng.choice(_OCR_PROMPTS), ", ".join(clean)), 10))
    except Exception:
        pass

    # VQA (5%) — SmolVLM is VERY prone to broken output
    try:
        pairs = _nested_get(data, "smolvlm.qa_pairs")
        if isinstance(pairs, list) and pairs:
            # Pick one random QA pair and format as ChatML turn
            valid_pairs = [
                qa for qa in pairs
                if isinstance(qa, dict)
                and isinstance(qa.get("question"), str) and qa["question"].strip()
                and isinstance(qa.get("answer"), str) and qa["answer"].strip()
            ]
            if valid_pairs:
                qa = _rng.choice(valid_pairs)
                pool.append((_chatml(qa["question"].strip(), qa["answer"].strip()), 5))
    except Exception:
        pass

    # Object Detection (5%)
    try:
        od = _nested_get(data, "florence.od")
        if not isinstance(od, dict):
            od = _nested_get(data, "florence.dense_region_caption")
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
                    pool.append((_chatml(_rng.choice(_OD_PROMPTS), ", ".join(unique)), 5))
    except Exception:
        pass

    # Fast path: nothing found — wrap fallback in ChatML
    if not pool:
        for k in ("caption", "text"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return _chatml(_rng.choice(_DESC_PROMPTS), v.strip())
        return _chatml("Describe this image.", fallback)

    # Single candidate — skip random
    if len(pool) == 1:
        return pool[0][0]

    # Weighted random
    texts, weights = zip(*pool)
    return _rng.choices(texts, weights=weights, k=1)[0]


def _load_caption(img_path: Path, sample_name: str) -> str:
    txt_path = img_path.with_suffix(".txt")
    if txt_path.exists():
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return text
        except OSError:
            pass

    json_path = img_path.with_suffix(".json")
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                return _select_caption(data, fallback=f"image of {sample_name}")
        except (OSError, json.JSONDecodeError):
            pass

    return f"image of {sample_name}"


class PrecomputeDataset(Dataset):
    """CPU-side loader used by DataLoader workers.
    
    Uses scale-to-fit + grid-snap resize to match training script behavior.
    Produces variable-resolution outputs for native resolution training.
    """

    VAE_DOWNSAMPLE = 8
    PATCH_SIZE = 2  # Must match model config

    def __init__(
        self,
        file_paths: Sequence[Path],
        max_size: int,
        already_cached: set[str],
        keep_aspect_ratio: bool = True,
        patch_size: int = 2,
    ):
        self.max_size = int(max_size)
        self.keep_aspect_ratio = bool(keep_aspect_ratio)
        self.patch_size = int(patch_size)
        self.block_size = self.VAE_DOWNSAMPLE * self.patch_size
        self.file_paths = [p for p in file_paths if p.stem not in already_cached]

    def _resize_native(self, img: Image.Image) -> Image.Image:
        """Scale-to-fit + grid-snap: matches training script resize logic exactly.
        
        1. Scale down to fit within max_size (preserves AR)
        2. Snap dimensions UP to vae_downsample * patch_size grid
        """
        src_w, src_h = img.size
        if src_w <= 0 or src_h <= 0:
            sz = max(self.block_size, self.max_size)
            return Image.new("RGB", (sz, sz), (0, 0, 0))

        # Step 1: Scale to fit within max_size
        scale = min(self.max_size / src_w, self.max_size / src_h)
        if scale < 1.0:
            new_w = max(1, int(src_w * scale))
            new_h = max(1, int(src_h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            new_w, new_h = src_w, src_h

        # Step 2: Grid-snap to block_size multiples (ceil-align)
        target_w = max(self.block_size, ((new_w + self.block_size - 1) // self.block_size) * self.block_size)
        target_h = max(self.block_size, ((new_h + self.block_size - 1) // self.block_size) * self.block_size)

        if target_w != new_w or target_h != new_h:
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        return img

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.file_paths[idx]
        name = img_path.stem
        valid = True

        try:
            with Image.open(str(img_path)) as img:
                img = img.convert("RGB")
                img = self._resize_native(img)
                arr = np.array(img, dtype=np.uint8, copy=True)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            h, w = tensor.shape[1], tensor.shape[2]
        except Exception:
            sz = max(self.block_size, self.max_size)
            # Grid-snap the fallback size
            sz = ((sz + self.block_size - 1) // self.block_size) * self.block_size
            tensor = torch.zeros((3, sz, sz), dtype=torch.uint8)
            h, w = sz, sz
            valid = False

        text = _load_caption(img_path, name) if valid else ""
        return {
            "name": name,
            "tensor": tensor,
            "text": text,
            "h": h,
            "w": w,
            "valid": valid,
        }


def collate_precompute(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # [FIX] With native resolution, tensors may have different shapes.
    # Return a list instead of stacking.
    return {
        "names": [item["name"] for item in batch],
        "tensors": [item["tensor"] for item in batch],
        "texts": [item["text"] for item in batch],
        "heights": [item["h"] for item in batch],
        "widths": [item["w"] for item in batch],
        "valid": [bool(item["valid"]) for item in batch],
    }


class ShardedLatentCacheWriter:
    """
    NVMe-friendly sequential shard writer.
    Backward-compatible with existing entry framing:
      [u32 latent_bytes][latent_raw][u32 token_bytes][token_raw]
    """

    def __init__(self, cache_dir: str, shard_max_bytes: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "index.json"
        self.meta_path = self.cache_dir / "cache_meta.json"
        self.shard_max_bytes = int(shard_max_bytes)
        self.cache_meta = {
            "version": 2,
            "token_padding": "none",
        }

        self.index: Dict[str, Dict[str, Any]] = {}
        if self.index_path.exists():
            try:
                self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
            except Exception:
                self.index = {}

        self._validate_cache_meta()

        self.current_shard_id = 0
        self.current_shard_bytes = 0
        self._fh = None

        shards = sorted(self.cache_dir.glob("shard_*.bin"))
        if shards:
            last = shards[-1]
            try:
                self.current_shard_id = int(last.stem.split("_")[1])
                self.current_shard_bytes = int(last.stat().st_size)
            except Exception:
                self.current_shard_id = 0
                self.current_shard_bytes = 0

    def _validate_cache_meta(self) -> None:
        if self.meta_path.exists():
            try:
                existing_meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read cache metadata at {self.meta_path}: {exc}"
                ) from exc

            if (
                existing_meta.get("version") != self.cache_meta["version"]
                or existing_meta.get("token_padding") != self.cache_meta["token_padding"]
            ):
                raise RuntimeError(
                    "Latent cache format mismatch. Use a fresh cache directory or delete the old cache."
                )
            return

        if self.index:
            raise RuntimeError(
                "Existing latent cache has no cache_meta.json and may contain legacy padded tokens. "
                "Use a fresh cache directory or delete the old cache before precomputing again."
            )

        self.meta_path.write_text(json.dumps(self.cache_meta), encoding="utf-8")

    def _shard_path(self, shard_id: int) -> Path:
        return self.cache_dir / f"shard_{shard_id:04d}.bin"

    def _ensure_open(self, pending_bytes: int) -> None:
        if self.current_shard_bytes + pending_bytes > self.shard_max_bytes:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None
            self.current_shard_id += 1
            self.current_shard_bytes = 0

        if self._fh is None:
            path = self._shard_path(self.current_shard_id)
            self._fh = open(path, "ab")
            self.current_shard_bytes = int(self._fh.tell())

    def write_entry(
        self,
        name: str,
        latent: np.ndarray,
        tokens: np.ndarray,
        h: int,
        w: int,
    ) -> None:
        if name in self.index:
            return

        latent_bytes = latent.tobytes(order="C")
        token_bytes = tokens.tobytes(order="C")
        entry_bytes = 4 + len(latent_bytes) + 4 + len(token_bytes)
        self._ensure_open(entry_bytes)

        offset = int(self._fh.tell())
        self._fh.write(struct.pack("<I", len(latent_bytes)))
        self._fh.write(latent_bytes)
        self._fh.write(struct.pack("<I", len(token_bytes)))
        self._fh.write(token_bytes)

        self.current_shard_bytes += entry_bytes
        self.index[name] = {
            "shard": int(self.current_shard_id),
            "offset": int(offset),
            "latent_shape": list(latent.shape),
            "latent_dtype": str(latent.dtype),
            "token_dtype": str(tokens.dtype),
            "token_len": int(tokens.shape[0]),
            "h": int(h),
            "w": int(w),
        }

    def write_batch(
        self,
        names: Sequence[str],
        latents: Sequence[np.ndarray],
        tokens_list: Sequence[np.ndarray],
        heights: Sequence[int],
        widths: Sequence[int],
    ) -> None:
        for name, latent, tokens, h, w in zip(names, latents, tokens_list, heights, widths):
            self.write_entry(name, latent, tokens, h, w)

    def flush_index(self) -> None:
        if self._fh is not None:
            self._fh.flush()
        self.meta_path.write_text(json.dumps(self.cache_meta), encoding="utf-8")
        tmp_path = self.index_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self.index), encoding="utf-8")
        tmp_path.replace(self.index_path)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
        self.flush_index()


@dataclass
class EncoderRuntimeConfig:
    cache_dir: str = ".latent_cache"
    batch_size: int = 32
    max_size: int = 256
    keep_aspect_ratio: bool = True
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    use_compile: bool = False
    compile_mode: str = "reduce-overhead"
    use_cuda_graphs: bool = False
    cuda_graph_batch_size: int = 64
    text_threads: int = 8
    max_text_length: int = 512
    latent_dtype: str = "float16"
    token_dtype: str = "int32"
    shard_max_bytes: int = 2 * 1024 * 1024 * 1024
    flush_every: int = 5000


class FastMultimodalEncoder:
    """
    GPU encoder with optional torch.compile and CUDA graphs.
    """

    def __init__(self, cfg: EncoderRuntimeConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.latent_torch_dtype = torch.float16 if cfg.latent_dtype == "float16" else torch.float32
        self.latent_np_dtype = np.float16 if cfg.latent_dtype == "float16" else np.float32
        self.token_np_dtype = np.int32 if cfg.token_dtype == "int32" else np.int64

        self.writer = ShardedLatentCacheWriter(cfg.cache_dir, cfg.shard_max_bytes)
        from data_manager import TiktokenTokenizer

        self.tokenizer = TiktokenTokenizer(max_length=cfg.max_text_length)

        self.vae: Optional[FluxVAE] = None
        self.use_cuda_graphs = bool(cfg.use_cuda_graphs and torch.cuda.is_available())
        self.graph_bs = int(cfg.cuda_graph_batch_size)
        self._graph = None
        self._graph_input = None
        self._graph_output = None
        self._graph_ready = False

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

    def _ensure_vae(self) -> None:
        if self.vae is not None:
            return

        from vae_module import FluxVAE

        self.vae = FluxVAE(dtype=self.vae_dtype)
        self.vae.eval()
        self.vae.to(self.device)

        if self.cfg.use_compile and hasattr(torch, "compile"):
            try:
                self.vae.vae.encoder = torch.compile(
                    self.vae.vae.encoder,
                    mode=self.cfg.compile_mode,
                    fullgraph=False,
                )
            except Exception:
                # Keep eager path if compile fails.
                pass

    @torch.inference_mode()
    def _encode_eager(self, batch_uint8: torch.Tensor) -> torch.Tensor:
        x = batch_uint8.to(self.device, dtype=torch.float32, non_blocking=True)
        x = x / 127.5 - 1.0
        latents = self.vae.encode(x.to(self.vae_dtype))
        # FluxVAE.encode() already applies the model's shift/scale normalization.
        # Do not apply any additional affine transform here, otherwise cached latents
        # will be permanently corrupted.
        return latents

    def _maybe_init_graph(self) -> None:
        if not self.use_cuda_graphs or self._graph_ready or self.vae is None:
            return

        try:
            h = int(self.cfg.max_size)
            w = int(self.cfg.max_size)
            self._graph_input = torch.zeros(
                (self.graph_bs, 3, h, w),
                device=self.device,
                dtype=torch.float32,
            )

            warm_stream = torch.cuda.Stream()
            warm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm_stream):
                for _ in range(3):
                    _ = self._encode_eager(self._graph_input)
            torch.cuda.current_stream().wait_stream(warm_stream)
            torch.cuda.synchronize()

            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._graph_output = self._encode_eager(self._graph_input)

            self._graph_ready = True
        except Exception:
            self.use_cuda_graphs = False
            self._graph = None
            self._graph_input = None
            self._graph_output = None

    @torch.inference_mode()
    def _encode_batch(self, batch_uint8: torch.Tensor) -> torch.Tensor:
        if not self.use_cuda_graphs:
            return self._encode_eager(batch_uint8)

        self._maybe_init_graph()
        if not self._graph_ready:
            return self._encode_eager(batch_uint8)

        bs = int(batch_uint8.shape[0])
        if bs > self.graph_bs:
            return self._encode_eager(batch_uint8)

        # [FIX] CUDA graphs are captured at a fixed spatial size (max_size×max_size).
        # With native resolution, images may be smaller — fall back to eager.
        _, _, in_h, in_w = batch_uint8.shape
        _, _, graph_h, graph_w = self._graph_input.shape
        if in_h != graph_h or in_w != graph_w:
            return self._encode_eager(batch_uint8)

        self._graph_input[:bs].copy_(
            batch_uint8.to(self.device, dtype=torch.float32, non_blocking=True)
        )
        if bs < self.graph_bs:
            self._graph_input[bs:].zero_()
        self._graph.replay()
        return self._graph_output[:bs]

    def _encode_tokens(self, texts: Sequence[str]) -> List[np.ndarray]:
        tokens = self.tokenizer.encode_batch(
            list(texts),
            max_length=self.cfg.max_text_length,
            # Store variable-length prompt tokens in cache. The shard index
            # already records token_len, so fixed-width padding is unnecessary
            # and would reintroduce the old prompt-dilution bug.
            add_pad=False,
            add_eot=True,
            num_threads=self.cfg.text_threads,
        )
        out: List[np.ndarray] = []
        for tensor in tokens:
            arr = tensor.detach().cpu().numpy().astype(self.token_np_dtype, copy=False)
            out.append(arr)
        return out

    def precompute_directory(self, data_dir: str) -> Dict[str, Any]:
        self._ensure_vae()

        root = Path(data_dir)
        all_files = list(_iter_image_files(root))
        total_files = len(all_files)
        already_cached = set(self.writer.index.keys())
        dataset = PrecomputeDataset(
            all_files,
            self.cfg.max_size,
            already_cached,
            keep_aspect_ratio=self.cfg.keep_aspect_ratio,
        )
        pending = len(dataset)
        skipped = total_files - pending

        loader_workers = autodetect_loader_workers(self.cfg.num_workers)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=loader_workers,
            pin_memory=bool(self.cfg.pin_memory),
            prefetch_factor=self.cfg.prefetch_factor if loader_workers > 0 else None,
            persistent_workers=loader_workers > 0,
            drop_last=False,
            collate_fn=collate_precompute,
        )

        if pending == 0:
            return {
                "total_files": total_files,
                "encoded": 0,
                "skipped": skipped,
                "rate": 0.0,
                "elapsed": 0.0,
                "cached_total": len(self.writer.index),
            }

        start = time.time()
        encoded = 0
        pbar = tqdm(total=pending, desc="Encoding", unit="img")

        for batch in loader:
            valid_indices = [i for i, ok in enumerate(batch["valid"]) if ok]
            if not valid_indices:
                continue

            names = [batch["names"][i] for i in valid_indices]
            texts = [batch["texts"][i] for i in valid_indices]
            heights = [batch["heights"][i] for i in valid_indices]
            widths = [batch["widths"][i] for i in valid_indices]
            tensors = [batch["tensors"][i] for i in valid_indices]

            # [FIX] With native resolution, tensors may have different shapes.
            # Group same-shape tensors for batched GPU encoding,
            # encode different shapes individually.
            from itertools import groupby
            shape_groups = {}
            for idx_in_batch, t in enumerate(tensors):
                shape_key = (t.shape[1], t.shape[2])
                if shape_key not in shape_groups:
                    shape_groups[shape_key] = []
                shape_groups[shape_key].append(idx_in_batch)

            latents_np: List[np.ndarray] = [None] * len(tensors)
            for shape_key, group_indices in shape_groups.items():
                group_tensors = torch.stack([tensors[i] for i in group_indices])
                group_latents_gpu = self._encode_batch(group_tensors)
                for j, orig_idx in enumerate(group_indices):
                    lat = (
                        group_latents_gpu[j]
                        .detach()
                        .to(dtype=self.latent_torch_dtype)
                        .cpu()
                        .numpy()
                        .astype(self.latent_np_dtype, copy=False)
                    )
                    latents_np[orig_idx] = lat

            token_arrays = self._encode_tokens(texts)
            self.writer.write_batch(names, latents_np, token_arrays, heights, widths)

            encoded += len(names)
            pbar.update(len(names))

            if encoded % max(1, self.cfg.flush_every) < len(names):
                self.writer.flush_index()

            elapsed = time.time() - start
            if elapsed > 0:
                pbar.set_postfix(
                    {
                        "img_s": f"{encoded / elapsed:.1f}",
                        "cached": f"{len(self.writer.index):,}",
                    }
                )

        pbar.close()
        self.writer.close()

        elapsed = time.time() - start
        rate = encoded / elapsed if elapsed > 0 else 0.0
        return {
            "total_files": total_files,
            "encoded": encoded,
            "skipped": skipped,
            "rate": rate,
            "elapsed": elapsed,
            "cached_total": len(self.writer.index),
        }


def load_cache_index(cache_dir: str) -> Dict[str, Dict[str, Any]]:
    path = Path(cache_dir) / "index.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
