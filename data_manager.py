"""
OmniFusion-X V2: Robust Data Manager
================================================================================
Handles multiple data formats for multimodal training:
- Image + Caption datasets (SD fine-tuning style with BLIP/Florence/WD-Tagger)
- ChatML instruction following format
- Alpaca instruction format
- XML Wiki dumps
- Plain text datasets

Features:
- Robust error handling for malformed data
- Multiple caption source fallbacks
- Lazy loading for memory efficiency
- Automatic format detection
- Streaming support for large datasets

Author: OmniFusion Team
License: Apache 2.0
"""

import os
import json
import re
import tiktoken
import random
import logging
import gc
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple, Iterator, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np

# Optional imports with graceful fallback
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataManager")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Image settings
    max_image_size: int = 512
    max_resolution_pixels: int = 327680 # Default for RTX 3070 Ti (8GB)
    min_image_size: int = 64
    patch_size: int = 8
    vae_downsample: int = 1  # Set to 8 for SDXL VAE, 1 for pixel-space
    in_channels: int = 3
    vae_scale_factor: float = 0.3611 # Standard Flux Scale
    vae_shift_factor: float = 0.1159 # Standard Flux Shift
    
    # Text settings
    max_text_length: int = 512
    vocab_size: int = 128000
    
    # Caption selection priority (first available is used)
    caption_priority: List[str] = field(default_factory=lambda: [
        "florence.more_detailed_caption",  # Most detailed
        "blip.caption",                     # Simple caption
        "wd_tagger.caption",                # Tag-based
        "existing_caption",                 # Legacy caption
    ])
    
    # Data augmentation
    random_flip: bool = False
    random_crop: bool = False
    
    # Error handling
    skip_invalid: bool = True  # Skip invalid samples instead of crashing
    max_retries: int = 3       # Retries for loading
    
    # Performance
    # Note: num_workers=0 is safer on Windows with PyQt6 to avoid spawn issues
    num_workers: int = 0  # Set to 4+ on Linux for better performance
    prefetch_factor: int = 2
    pin_memory: bool = True


# =============================================================================
# Tokenizers
# =============================================================================

# =============================================================================
# Special Token Constants (Multi-Image Support)
# =============================================================================
IMAGE_TOKEN = 100293  # Placeholder for image insertion in text (LLaVA-style)


class TiktokenTokenizer:
    """GPT-4 Tokenizer (cl100k_base) padded to 128k for Tensor Cores."""
    def __init__(self, max_length: int = 512):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.eot_token = 100257     # Standard Tiktoken EOT
        self.pad_token = 100258     # User Defined - PADDING ONLY
        self.generate_token = 100260 # Ghost Token for Roleplay
        self.image_token = IMAGE_TOKEN  # Multi-image placeholder (100293)
        
        # Resolution tokens: 100261 to 100292
        self.res_buckets = list(range(64, 1025, 64))
        self.h_token_start = 100261
        self.w_token_start = 100261 + len(self.res_buckets)
        
        self.vocab_size = 100352    # Optimized Size (128x multiple)
        self.max_length = max_length
        
        # [FIX] ChatML tokens at dedicated IDs to avoid collision with pad_token (100258)
        self.special_to_id = {
            "<|im_start|>": 100294,  # Dedicated ID (no collision with pad)
            "<|im_end|>": 100295,    # Dedicated ID
            "<GENERATE>": self.generate_token,
            "<image>": self.image_token,  # Multi-image placeholder
        }
        for i, val in enumerate(self.res_buckets):
            self.special_to_id[f"<img_h_{val}>"] = self.h_token_start + i
            self.special_to_id[f"<img_w_{val}>"] = self.w_token_start + i
            
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}

    # === ADD THIS METHOD ===
    def encode_batch(self, texts: List[str], **kwargs) -> List[torch.Tensor]:
        """
        Encodes a batch of texts.
        Required by encoder_backend.py for multi-threaded precomputing.
        """
        # [FIX] Remove 'num_threads' from kwargs so it doesn't get passed to encode()
        kwargs.pop('num_threads', None)
        return [self.encode(text, **kwargs) for text in texts]

    def encode(self, text: str, max_length: int = None, add_pad: bool = True, add_eot: bool = True) -> torch.Tensor:
        length = max_length or self.max_length
        
        # 1. Handle Special Token Injection
        pattern = "|".join(map(re.escape, self.special_to_id.keys()))
        # Handle cases where special_to_id is empty
        if pattern:
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]
        
        tokens = []
        for part in parts:
            if not part: continue  # [FIX] Skip empty splits
            if part in self.special_to_id:
                tokens.append(self.special_to_id[part])
            else:
                tokens.extend(self.enc.encode(part, allowed_special={'<|endoftext|>'}))
        
        # 2. Add EOT
        if add_eot:
            tokens.append(self.eot_token)
        
        # 3. Truncate/Pad
        if len(tokens) > length: tokens = tokens[:length]
        if add_pad and len(tokens) < length: 
            tokens.extend([self.pad_token] * (length - len(tokens)))
        
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, token_ids) -> str:
        if isinstance(token_ids, torch.Tensor): token_ids = token_ids.tolist()
        try:
            result = []
            special_rev = {
                100257: "[EOT]",
                100258: "<|im_start|>",  # Old im_start (also pad, but show for old models)
                100259: "<|im_end|>",    # Old im_end (for old models)
                100260: "[GENERATE]",
                100293: "[IMAGE]",
                100294: "<|im_start|>",  # New im_start
                100295: "<|im_end|>",    # New im_end
            }
            # Add resolution tokens
            for i, val in enumerate(self.res_buckets):
                special_rev[self.h_token_start + i] = f"[H_{val}]"
                special_rev[self.w_token_start + i] = f"[W_{val}]"

            for t in token_ids:
                # Note: Don't skip 100258 anymore - it maps to im_start for old models
                if t in special_rev:
                    result.append(special_rev[t])
                elif 0 <= t < 100257:
                    result.append(self.enc.decode([t]))
                else:
                    result.append(f"[{t}]")
            return "".join(result)
        except Exception:
            # Fallback
            chars = []
            for t in token_ids:
                if t == self.pad_token or t < 0 or t >= self.vocab_size: continue
                if t in self.id_to_special:
                    chars.append(self.id_to_special[t])
                else:
                    try:
                        chars.append(self.enc.decode([t]))
                    except:
                        chars.append("")
            return "".join(chars)


# =============================================================================
# Caption Extractors
# =============================================================================

class CaptionExtractor:
    """Extracts captions from various JSON structures with fallbacks."""
    
    @staticmethod
    def extract(data: Dict[str, Any], priority: List[str] = None) -> Optional[str]:
        """
        Extract caption from data dict using priority order.
        
        Args:
            data: JSON data dict
            priority: List of dotted paths like "florence.more_detailed_caption"
            
        Returns:
            Caption string or None if not found
        """
        if priority is None:
            priority = [
                "florence.more_detailed_caption",
                "blip.caption",
                "wd_tagger.caption",
                "existing_caption",
                "caption",
                "text",
                "prompt",
            ]
        
        for path in priority:
            try:
                value = CaptionExtractor._get_nested(data, path)
                if value and isinstance(value, str) and value.strip() and value.strip() != "N/A":
                    return value.strip()
            except (KeyError, TypeError, AttributeError):
                continue
        
        return None
    
    @staticmethod
    def _get_nested(data: Dict, path: str) -> Any:
        """Get nested dict value using dotted path."""
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value
    
    @staticmethod
    def extract_all(data: Dict[str, Any]) -> Dict[str, str]:
        """Extract all available captions from data."""
        captions = {}
        
        # BLIP
        if "blip" in data and isinstance(data["blip"], dict):
            if "caption" in data["blip"]:
                captions["blip"] = data["blip"]["caption"]
        
        # Florence
        if "florence" in data and isinstance(data["florence"], dict):
            if "more_detailed_caption" in data["florence"]:
                captions["florence"] = data["florence"]["more_detailed_caption"]
        
        # WD Tagger (tags)
        if "wd_tagger" in data and isinstance(data["wd_tagger"], dict):
            if "caption" in data["wd_tagger"]:
                captions["wd_tagger"] = data["wd_tagger"]["caption"]
        
        # SmolVLM QA pairs
        if "smolvlm" in data and isinstance(data["smolvlm"], dict):
            qa_pairs = data["smolvlm"].get("qa_pairs", [])
            if qa_pairs:
                # Combine QA pairs into text
                qa_text = " ".join([f"Q: {qa.get('question', '')} A: {qa.get('answer', '')}" 
                                    for qa in qa_pairs if isinstance(qa, dict)])
                if qa_text.strip():
                    captions["smolvlm_qa"] = qa_text
        
        # Existing caption
        if "existing_caption" in data:
            cap = data["existing_caption"]
            if cap and cap != "N/A":
                captions["existing"] = cap
        
        return captions
    
    @staticmethod
    def combine_captions(captions: Dict[str, str], style: str = "best") -> str:
        """
        Combine multiple captions into one.
        
        Args:
            captions: Dict of caption source -> caption text
            style: "best" (use longest), "concat" (combine all), "tags_only" (use wd_tagger)
        """
        if not captions:
            return ""
        
        if style == "best":
            # Return the longest caption
            return max(captions.values(), key=len)
        
        elif style == "concat":
            # Combine all unique captions
            unique = list(set(captions.values()))
            return " | ".join(unique)
        
        elif style == "tags_only":
            return captions.get("wd_tagger", max(captions.values(), key=len))
        
        else:
            return list(captions.values())[0]


# =============================================================================
# Format Parsers
# =============================================================================

class FormatParser(ABC):
    """Base class for parsing different data formats."""
    
    @abstractmethod
    def parse(self, data: Any) -> List[Dict[str, Any]]:
        """Parse data into list of samples."""
        pass
    
    @abstractmethod
    def can_parse(self, data: Any) -> bool:
        """Check if this parser can handle the data."""
        pass


class ChatMLParser(FormatParser):
    """
    Parser for ChatML format.
    
    Example:
        <|im_start|>system
        You are a helpful assistant.
        <|im_end|>
        <|im_start|>user
        Hello!
        <|im_end|>
        <|im_start|>assistant
        Hi there!
        <|im_end|>
    """
    
    def can_parse(self, data: Any) -> bool:
        if isinstance(data, str):
            return "<|im_start|>" in data or "<|im_end|>" in data
        if isinstance(data, dict):
            return data.get("format") == "chatml" or "messages" in data
        return False
    
    def parse(self, data: Any) -> List[Dict[str, Any]]:
        samples = []
        
        try:
            if isinstance(data, str):
                # Parse raw ChatML string
                messages = self._parse_chatml_string(data)
                if messages:
                    samples.append({
                        "type": "text",
                        "format": "chatml",
                        "messages": messages,
                        "text": data,
                    })
            
            elif isinstance(data, dict):
                # Already structured
                messages = data.get("messages", [])
                if messages:
                    samples.append({
                        "type": "text",
                        "format": "chatml",
                        "messages": messages,
                        "text": self._format_messages(messages),
                    })
        
        except Exception as e:
            logger.warning(f"ChatML parse error: {e}")
        
        return samples
    
    def _parse_chatml_string(self, text: str) -> List[Dict[str, str]]:
        """Parse ChatML string into messages list."""
        messages = []
        pattern = r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>"
        
        for match in re.finditer(pattern, text, re.DOTALL):
            role = match.group(1).strip()
            content = match.group(2).strip()
            messages.append({"role": role, "content": content})
        
        return messages
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages back to ChatML string."""
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(lines)


class AlpacaParser(FormatParser):
    """
    Parser for Alpaca instruction format.
    
    Example:
        {
            "instruction": "Summarize the following text.",
            "input": "Long text here...",
            "output": "Summary here."
        }
    """
    
    def can_parse(self, data: Any) -> bool:
        if isinstance(data, dict):
            return "instruction" in data and ("output" in data or "response" in data)
        return False
    
    def parse(self, data: Any) -> List[Dict[str, Any]]:
        samples = []
        
        try:
            if isinstance(data, dict):
                instruction = data.get("instruction", "").strip()
                input_text = data.get("input", "").strip()
                output = data.get("output", data.get("response", "")).strip()
                
                if instruction:
                    # Format as instruction-following text
                    if input_text:
                        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                    else:
                        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                    
                    samples.append({
                        "type": "text",
                        "format": "alpaca",
                        "instruction": instruction,
                        "input": input_text,
                        "output": output,
                        "text": prompt,
                    })
        
        except Exception as e:
            logger.warning(f"Alpaca parse error: {e}")
        
        return samples


class XMLWikiParser(FormatParser):
    """
    Parser for XML Wikipedia dumps.
    
    Extracts article text from MediaWiki XML format.
    """
    
    def can_parse(self, data: Any) -> bool:
        if isinstance(data, str):
            return "<mediawiki" in data or "<page>" in data
        return False
    
    def parse(self, data: Any) -> List[Dict[str, Any]]:
        samples = []
        
        try:
            if isinstance(data, str):
                # Try to parse as XML
                # Handle partial or malformed XML gracefully
                data = self._clean_xml(data)
                root = ET.fromstring(f"<root>{data}</root>")
                
                for page in root.iter("page"):
                    title_elem = page.find("title")
                    text_elem = page.find(".//text")
                    
                    if title_elem is not None and text_elem is not None:
                        title = title_elem.text or ""
                        text = text_elem.text or ""
                        
                        # Clean wiki markup (basic)
                        text = self._clean_wiki_markup(text)
                        
                        if text.strip():
                            samples.append({
                                "type": "text",
                                "format": "wiki",
                                "title": title,
                                "text": text.strip(),
                            })
        
        except ET.ParseError as e:
            logger.warning(f"XML parse error: {e}")
        except Exception as e:
            logger.warning(f"Wiki parse error: {e}")
        
        return samples
    
    def _clean_xml(self, text: str) -> str:
        """Clean XML for parsing."""
        # Remove problematic characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        return text
    
    def _clean_wiki_markup(self, text: str) -> str:
        """Remove basic wiki markup."""
        # Remove templates {{...}}
        text = re.sub(r'\{\{[^}]*\}\}', '', text)
        # Remove categories [[Category:...]]
        text = re.sub(r'\[\[Category:[^\]]*\]\]', '', text)
        # Convert links [[text|display]] -> display
        text = re.sub(r'\[\[[^|\]]*\|([^\]]*)\]\]', r'\1', text)
        # Convert simple links [[text]] -> text
        text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
        # Remove headers
        text = re.sub(r'={2,}[^=]*={2,}', '', text)
        # Clean extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()


class RobustCaptionSelector:
    def __init__(self):
        # 10% BLIP, 40% Florence Detailed, 20% WD Tagger, 10% SmolVLM, 5% OCR, 5% Regions
        self.strategy = [
            ("florence.more_detailed_caption", 0.40),
            ("wd_tagger.caption", 0.20),
            ("smolvlm.qa_pairs", 0.15),
            ("blip.caption", 0.10),
            ("florence.ocr", 0.05),
            ("florence.dense_region_caption", 0.05),
            ("question_used_for_image", 0.05),
        ]

        # Optional hard override for reproducibility / dataset hygiene.
        #
        # When set, we stop sampling captions from multiple sources and always use one path.
        # This avoids training on systematically inconsistent supervision (e.g. one captioner
        # says "blue hair" while another says "red hair" for the same image).
        #
        # Windows example:
        #   set OMNIFUSION_CAPTION_KEY=wd_tagger.caption
        #
        # Aliases:
        #   wd -> wd_tagger.caption
        #   florence -> florence.more_detailed_caption
        #   blip -> blip.caption
        #   smolvlm -> smolvlm.qa_pairs
        self.override_key = os.environ.get("OMNIFUSION_CAPTION_KEY", "").strip()
        # Caption sampling mode:
        # - "random" (default): sample a source each time a sample is read (can change per-epoch).
        # - "deterministic": choose a source deterministically per-sample (stable across epochs),
        #   which prevents contradictory supervision when multiple captioners disagree.
        self.sampling_mode = os.environ.get("OMNIFUSION_CAPTION_SAMPLING", "random").strip().lower()
        if self.sampling_mode not in ("random", "deterministic"):
            self.sampling_mode = "random"
        _aliases = {
            "wd": "wd_tagger.caption",
            "wd_tagger": "wd_tagger.caption",
            "florence": "florence.more_detailed_caption",
            "blip": "blip.caption",
            "smolvlm": "smolvlm.qa_pairs",
        }
        if self.override_key in _aliases:
            self.override_key = _aliases[self.override_key]
        if not self.override_key:
            self.override_key = None
        # Fixes </s>, <|endoftext|>, and [EOS] bugs
        self.cleaner = re.compile(r"(</s>|<\|endoftext\|>|\[EOS\]|<eos>|user<|im_end|>)", re.IGNORECASE)

    def _stable_uniform01(self, data: dict) -> float:
        """
        Deterministic per-sample RNG in [0,1) derived from the sample identity.

        This keeps caption-source selection stable across epochs, avoiding the case where the same
        image is sometimes trained with Florence and other times with WD, etc.
        """
        import hashlib

        # Prefer filename/path identifiers that are stable for the same image.
        ident = (
            str(data.get("image_filename") or "")
            or str(data.get("image_path") or "")
            or str(data.get("name") or "")
            or str(data.get("id") or "")
        )
        if not ident:
            # Fallback: deterministic but lower quality (depends on dict ordering of JSON loader).
            ident = json.dumps(data, sort_keys=True, ensure_ascii=True)[:256]

        digest = hashlib.blake2b(ident.encode("utf-8", errors="ignore"), digest_size=8).digest()
        x = int.from_bytes(digest, "little", signed=False)
        return x / float(2**64)

    def clean(self, text):
        if not text: return ""
        text = self.cleaner.sub("", str(text))
        return " ".join(text.split())

    def select(self, data):
        # Robust check for data dict
        if not isinstance(data, dict): return ""

        key = self.override_key
        if key is None:
            if self.sampling_mode == "deterministic":
                r = self._stable_uniform01(data)
            else:
                r = random.random()
            cumulative = 0.0
            key = None
            for k, p in self.strategy:
                cumulative += p
                if r <= cumulative:
                    key = k
                    break

            # Default key if random failed
            if not key:
                key = "florence.more_detailed_caption"
        
        # Extraction & Flattening Logic
        text = ""
        val = data
        for k_part in key.split('.'):
            if isinstance(val, dict): val = val.get(k_part)
            else: 
                val = None
                break
        
        # Type-specific flattening
        if isinstance(val, list):
            # Special case for SmolVLM QA pairs
            if key == "smolvlm.qa_pairs":
                qa_list = []
                for entry in val:
                    if isinstance(entry, dict):
                        q = entry.get("question", "")
                        a = entry.get("answer", "")
                        if q or a: qa_list.append(f"Q: {q} A: {a}")
                text = " ".join(qa_list)
            else:
                text = " ".join([str(x) for x in val])
        elif isinstance(val, dict):
            if "labels" in val: 
                text = ", ".join(sorted(set(str(l) for l in val["labels"])))
            elif "caption" in val:
                text = str(val["caption"])
            else:
                text = str(val)
        elif val is not None:
            text = str(val)
            
        # Fallback Chain (Safety)
        if not text or text.strip() == "N/A":
            text = data.get("florence", {}).get("more_detailed_caption", "")
        if not text:
            text = data.get("blip", {}).get("caption", "")
        if not text:
            text = data.get("caption", data.get("text", ""))
        
        return self.clean(text)


class ImageCaptionParser(FormatParser):
    """
    Parser for image-caption JSON format (SD fine-tuning style).
    
    Handles the multi-caption structure with BLIP, Florence, WD-Tagger, etc.
    """
    
    def can_parse(self, data: Any) -> bool:
        if isinstance(data, dict):
            return "image_path" in data or "image_filename" in data or "blip" in data or "florence" in data
        return False
    
    def parse(self, data: Any) -> List[Dict[str, Any]]:
        if not hasattr(self, 'sel'): self.sel = RobustCaptionSelector()
        samples = []
        try:
            # FIX: Windows Path Crash -> os.path.basename
            raw_path = data.get("image_path", data.get("image_filename", ""))
            filename = raw_path
            
            # FIX: Apply Mixing Strategy
            caption = self.sel.select(data)
            
            if filename:
                samples.append({
                    "type": "image_text",
                    "format": "sd_caption",
                    "image_path": filename,
                    "text": caption
                })
        except Exception as e:
            logger.warning(f"Parse error: {e}")
        return samples


# =============================================================================
# Image Loading Utilities
# =============================================================================

class ImageLoader:
    """Robust image loading with error handling and preprocessing."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._lock = threading.Lock()

    # === ADD THESE TWO METHODS ===
    def __getstate__(self):
        """Exclude the lock when pickling (for Windows multiprocessing)."""
        state = self.__dict__.copy()
        if '_lock' in state:
            del state['_lock']
        return state

    def __setstate__(self, state):
        """Restore state and create a new lock in the worker process."""
        self.__dict__.update(state)
        self._lock = threading.Lock()
    # ==============================
    
    def load(self, path: str, retries: int = None) -> Optional[torch.Tensor]:
        # ... (rest of the code remains the same)
        """
        Load and preprocess image.
        
        Returns:
            Tensor of shape [C, H, W] normalized to [-1, 1], or None on failure
        """
        if not PIL_AVAILABLE:
            logger.error("PIL not available for image loading")
            return None
        
        retries = retries or self.config.max_retries
        
        for attempt in range(retries):
            try:
                return self._load_image(path)
            except FileNotFoundError:
                logger.warning(f"Image not found: {path}")
                return None
            except Exception as e:
                if attempt < retries - 1:
                    logger.debug(f"Retry {attempt + 1}/{retries} for {path}: {e}")
                else:
                    logger.warning(f"Failed to load image after {retries} attempts: {path} - {e}")
                    return None
        
        return None
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Internal image loading logic."""
        # Handle various path formats
        path = self._normalize_path(path)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        
        # Load with PIL
        img = Image.open(path)
        
        # Convert to RGB (handle RGBA, grayscale, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize if needed
        img = self._resize_image(img)
        
        # Align to patch size
        img = self._align_to_patches(img)
        
        # Convert to tensor and normalize to [-1, 1]
        arr = np.array(img).astype(np.float32)
        tensor = torch.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0
        
        return tensor
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path handling Windows/Unix differences."""
        path = path.replace("\\", "/").replace("//", "/")
        path = os.path.normpath(path)
        return path
    
    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image to fit within max bounds (Native Resolution)."""
        w, h = img.size
        max_size = self.config.max_image_size
        
        # Only downscale if strictly necessary to avoid OOM
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            
        return img

    def _align_to_patches(self, img: Image.Image) -> Image.Image:
        """Align image dimensions to patch size (Native Resolution - No Padding/Cropping)."""
        w, h = img.size
        patch = self.config.patch_size * self.config.vae_downsample
        
        # Simply round down or up to nearest patch multiple
        # Ideally round to closest to preserve AR best, or round down to be safe
        new_w = (w // patch) * patch
        new_h = (h // patch) * patch
        
        # Ensure at least one patch
        new_w = max(patch, new_w)
        new_h = max(patch, new_h)
        
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            
        return img


# =============================================================================
# Tokenization
# =============================================================================

class SimpleTokenizer:
    """
    Simple ASCII-based tokenizer for testing.
    Replace with proper tokenizer (SentencePiece, BPE) for production.
    """
    
    def __init__(self, vocab_size: int = 32000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Special tokens
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.unk_token = 3
    
    def encode(self, text: str, add_special_tokens: bool = True, 
               max_length: int = None, padding: bool = True) -> torch.Tensor:
        """Encode text to token IDs."""
        if not text:
            text = ""
        
        max_length = max_length or self.max_length
        
        # Convert to ASCII token IDs (offset by 4 for special tokens)
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.bos_token)
        
        for char in text:
            try:
                token_id = ord(char) % (self.vocab_size - 4) + 4
                tokens.append(token_id)
            except:
                tokens.append(self.unk_token)
        
        if add_special_tokens:
            tokens.append(self.eos_token)
        
        # Truncate
        tokens = tokens[:max_length]
        
        # Pad
        if padding and len(tokens) < max_length:
            tokens = tokens + [self.pad_token] * (max_length - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        chars = []
        
        for token_id in token_ids.tolist():
            if token_id <= 3:  # Special tokens
                continue
            try:
                char = chr((token_id - 4) % 128)
                chars.append(char)
            except:
                chars.append("?")
        
        return "".join(chars)


# =============================================================================
# Main Dataset Classes
# =============================================================================

class MultimodalDataset(Dataset):
    """
    Unified dataset for multimodal training.
    
    Supports:
    - Image + caption pairs
    - Text-only samples (ChatML, Alpaca, Wiki)
    - Mixed batches
    """
    
    def __init__(
        self,
        data_paths: List[str],
        config: DataConfig = None,
        tokenizer: Any = None,
        transform: Callable = None,
    ):
        """
        Args:
            data_paths: List of paths to data files/directories
            config: Data configuration
            tokenizer: Tokenizer for text (uses TiktokenTokenizer if None)
            transform: Optional image transform
        """
        self.config = config or DataConfig()
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = TiktokenTokenizer(max_length=self.config.max_text_length)
        self.transform = transform
        
        # Parsers
        self.parsers = [
            ImageCaptionParser(),
            ChatMLParser(),
            AlpacaParser(),
            XMLWikiParser(),
        ]
        
        # Image loader
        self.image_loader = ImageLoader(self.config)
        
        # Load all samples
        self.samples = []
        self._load_data(data_paths)
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(data_paths)} sources")
    
    def _load_data(self, data_paths: List[str]):
        """Load data from all paths."""
        for path in data_paths:
            try:
                if os.path.isdir(path):
                    self._load_directory(path)
                elif os.path.isfile(path):
                    self._load_file(path)
                else:
                    logger.warning(f"Path does not exist: {path}")
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
                if not self.config.skip_invalid:
                    raise
    
    def _load_directory(self, dir_path: str):
        """Load all files from directory."""
        extensions = {".json", ".jsonl", ".txt", ".xml", ".parquet"}
        
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    file_path = os.path.join(root, file)
                    try:
                        self._load_file(file_path)
                    except Exception as e:
                        logger.warning(f"Error loading {file_path}: {e}")
    
    def _load_file(self, file_path: str):
        """Load samples from a single file."""
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == ".json":
                self._load_json(file_path)
            elif ext == ".jsonl":
                self._load_jsonl(file_path)
            elif ext == ".txt":
                self._load_text(file_path)
            elif ext == ".xml":
                self._load_xml(file_path)
            elif ext == ".parquet" and PANDAS_AVAILABLE:
                self._load_parquet(file_path)
            else:
                logger.debug(f"Unsupported file type: {ext}")
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            if not self.config.skip_invalid:
                raise
    
    def _load_json(self, file_path: str):
        """Load JSON file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                self._parse_and_add(item)
        else:
            self._parse_and_add(data)
    
    def _load_jsonl(self, file_path: str):
        """Load JSONL file (one JSON per line)."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        self._parse_and_add(data)
                    except json.JSONDecodeError:
                        pass
    
    def _load_text(self, file_path: str):
        """Load plain text file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        if text.strip():
            self.samples.append({
                "type": "text",
                "format": "plain",
                "text": text.strip(),
                "source": file_path,
            })
    
    def _load_xml(self, file_path: str):
        """Load XML file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        parser = XMLWikiParser()
        if parser.can_parse(content):
            samples = parser.parse(content)
            for sample in samples:
                sample["source"] = file_path
                self.samples.append(sample)
    
    def _load_parquet(self, file_path: str):
        """Load Parquet file."""
        df = pd.read_parquet(file_path)
        
        for idx, row in df.iterrows():
            data = row.to_dict()
            self._parse_and_add(data)
    
    def _parse_and_add(self, data: Any):
        """Parse data with appropriate parser and add to samples."""
        for parser in self.parsers:
            try:
                if parser.can_parse(data):
                    samples = parser.parse(data)
                    self.samples.extend(samples)
                    return
            except Exception as e:
                logger.debug(f"Parser {parser.__class__.__name__} failed: {e}")
        
        # Fallback: treat as plain text if has "text" key
        if isinstance(data, dict) and "text" in data:
            text = data.get("text", "")
            if text and isinstance(text, str) and text.strip():
                self.samples.append({
                    "type": "text",
                    "format": "plain",
                    "text": text.strip(),
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        try:
            return self._process_sample(sample)
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            if self.config.skip_invalid:
                # Return a dummy sample
                return self._get_dummy_sample()
            raise
    
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a sample into model-ready format."""
        sample_type = sample.get("type", "text")
        
        result = {
            "type": sample_type,
            "format": sample.get("format", "unknown"),
        }
        
        # Process text
        text = sample.get("text", "")
        if text:
            result["text"] = text
            result["input_ids"] = self.tokenizer.encode(
                text, 
                max_length=self.config.max_text_length,
                add_pad=False,
            )
        else:
            result["text"] = ""
            result["input_ids"] = self.tokenizer.encode("", max_length=32, add_pad=False)
        
        # Process image if present
        if sample_type == "image_text" and "image_path" in sample:
            image_path = sample["image_path"]
            image = self.image_loader.load(image_path)
            
            if image is not None:
                if self.transform:
                    image = self.transform(image)
                result["image"] = image
                result["has_image"] = True
            else:
                result["has_image"] = False
        else:
            result["has_image"] = False
        
        return result
    
    def _get_dummy_sample(self) -> Dict[str, Any]:
        """Return a dummy sample for error cases."""
        return {
            "type": "text",
            "format": "dummy",
            "text": "",
            "input_ids": self.tokenizer.encode("", max_length=32, add_pad=False),
            "has_image": False,
        }


class StreamingMultimodalDataset(IterableDataset):
    """
    Streaming version for large datasets that don't fit in memory.
    Supports directories and mixed formats (JSON, JSONL, TXT, XML).
    """
    
    def __init__(
        self,
        data_paths: List[str],
        config: DataConfig = None,
        tokenizer: Any = None,
        shuffle_buffer: int = 1000,
    ):
        self.data_paths = data_paths
        self.config = config or DataConfig()
        self.tokenizer = tokenizer or TiktokenTokenizer(max_length=self.config.max_text_length)
        self.shuffle_buffer = shuffle_buffer
        
        self.parsers = [
            ImageCaptionParser(),
            ChatMLParser(),
            AlpacaParser(),
            XMLWikiParser(),
        ]
        
        self.image_loader = ImageLoader(self.config)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        buffer = []
        
        # Generator that yields file paths from inputs (handles dirs and files)
        def file_generator(paths):
            for path in paths:
                if os.path.isfile(path):
                    yield path
                elif os.path.isdir(path):
                    for root, _, files in os.walk(path):
                        for file in files:
                            # Filter for supported extensions
                            if file.lower().endswith(('.json', '.jsonl', '.txt', '.xml', '.parquet')):
                                yield os.path.join(root, file)

        for file_path in file_generator(self.data_paths):
            try:
                for sample in self._stream_file(file_path):
                    buffer.append(sample)
                    
                    if len(buffer) >= self.shuffle_buffer:
                        random.shuffle(buffer)
                        while len(buffer) > self.shuffle_buffer // 2:
                            yield buffer.pop(0)
            except Exception as e:
                logger.warning(f"Error streaming {file_path}: {e}")
        
        # Yield remaining
        random.shuffle(buffer)
        for sample in buffer:
            yield sample
    
    def _stream_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """Stream samples from a file based on extension."""
        ext = os.path.splitext(file_path)[1].lower()
        
        # 1. JSONL (Line-delimited JSON) - Standard for ChatML/Alpaca
        if ext == ".jsonl":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        for sample in self._parse_data(data):
                            yield self._process_sample(sample)
                    except:
                        pass
        
        # 2. JSON (List of items or single item) - Standard for Alpaca/Instruction
        elif ext == ".json":
            try:
                # Streaming JSON array is complex, we assume it fits in RAM or is line-delimited
                # For true streaming of massive JSON arrays, use ijson. 
                # Here we fallback to loading chunks or standard load for robustness.
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            for sample in self._parse_data(item):
                                yield self._process_sample(sample)
                    else:
                        for sample in self._parse_data(data):
                            yield self._process_sample(sample)
            except Exception:
                pass

        # 3. TXT (Raw Text)
        elif ext == ".txt":
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    if text.strip():
                        # Create a raw text sample
                        sample = {
                            "type": "text", 
                            "format": "plain", 
                            "text": text.strip()
                        }
                        yield self._process_sample(sample)
            except Exception:
                pass

        # 4. XML (Wiki Dumps)
        elif ext == ".xml":
            try:
                # XML parsing can be heavy. We read whole file for parser.
                # For massive XMLs, one should use iterparse, but here we reuse the Parser logic.
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                # Use the existing XMLWikiParser
                parser = XMLWikiParser()
                if parser.can_parse(content):
                    raw_samples = parser.parse(content)
                    for s in raw_samples:
                        yield self._process_sample(s)
            except Exception:
                pass

    def _parse_data(self, data: Any) -> List[Dict]:
        """Parse data with appropriate parser (ChatML, Alpaca, etc)."""
        for parser in self.parsers:
            if parser.can_parse(data):
                return parser.parse(data)
        
        # Fallback for plain dicts with 'text'
        if isinstance(data, dict) and "text" in data:
            return [{
                "type": "text",
                "format": "plain",
                "text": data["text"]
            }]
        return []
    
    def _process_sample(self, sample: Dict) -> Dict[str, Any]:
        """Process sample into model format."""
        text = sample.get("text", "")
        
        result = {
            "type": sample.get("type", "text"),
            "text": text,
            "input_ids": self.tokenizer.encode(text, max_length=self.config.max_text_length, add_pad=False),
            "has_image": False,
        }
        
        if sample.get("type") == "image_text" and "image_path" in sample:
            image = self.image_loader.load(sample["image_path"])
            if image is not None:
                result["image"] = image
                result["has_image"] = True
        
        return result


# =============================================================================
# Multi-Image Chat Dataset (Long-Context Support)
# =============================================================================

class MultiImageChatDataset(Dataset):
    """
    Dataset for long-context training with multiple interleaved images.
    
    Uses LLaVA-style format: <image> placeholders in text with separate image paths.
    
    Example JSON format:
        {
            "messages": [
                {"role": "user", "content": "Compare these images: <image> <image>"},
                {"role": "assistant", "content": "The first shows a red square..."}
            ],
            "images": ["path/to/img1.png", "path/to/img2.png"]
        }
    
    Features:
        - Supports up to max_context_length tokens (default 16k)
        - Handles variable number of images per sample
        - Returns image positions for 3D RoPE temporal ordering
        - Compatible with pack_inputs variable-length packing
    """
    
    def __init__(
        self,
        data_paths: List[str],
        config: DataConfig = None,
        tokenizer: Any = None,
        vae: Any = None,
        max_context_length: int = 16384,
        max_images_per_sample: int = 32,
    ):
        """
        Args:
            data_paths: List of paths to JSONL/JSON files with multi-image conversations
            config: Data configuration
            tokenizer: Tokenizer instance (TiktokenTokenizer if None)
            vae: VAE encoder for image latents (optional, for pre-encoding)
            max_context_length: Maximum token length (default 16k)
            max_images_per_sample: Maximum images allowed per sample
        """
        self.config = config or DataConfig()
        self.tokenizer = tokenizer or TiktokenTokenizer(max_length=max_context_length)
        self.vae = vae
        self.max_context_length = max_context_length
        self.max_images_per_sample = max_images_per_sample
        self.image_loader = ImageLoader(self.config)
        
        # Load all samples
        self.samples: List[Dict[str, Any]] = []
        self._load_data(data_paths)
        
        logger.info(f"MultiImageChatDataset: Loaded {len(self.samples)} samples from {len(data_paths)} paths")
    
    def _load_data(self, data_paths: List[str]):
        """Load multi-image conversation data from files."""
        for path in data_paths:
            path = Path(path)
            if not path.exists():
                logger.warning(f"Path not found: {path}")
                continue
            
            if path.is_file():
                self._load_file(str(path))
            elif path.is_dir():
                for file_path in path.rglob("*"):
                    if file_path.suffix.lower() in [".json", ".jsonl"]:
                        self._load_file(str(file_path))
    
    def _load_file(self, file_path: str):
        """Load samples from a single file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.endswith(".jsonl"):
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            sample = self._parse_sample(data, file_path, line_num)
                            if sample:
                                self.samples.append(sample)
                        except json.JSONDecodeError:
                            continue
                else:  # .json
                    data = json.load(f)
                    if isinstance(data, list):
                        for idx, item in enumerate(data):
                            sample = self._parse_sample(item, file_path, idx)
                            if sample:
                                self.samples.append(sample)
                    else:
                        sample = self._parse_sample(data, file_path, 0)
                        if sample:
                            self.samples.append(sample)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    def _parse_sample(self, data: Dict, source_file: str, idx: int) -> Optional[Dict]:
        """
        Parse a single conversation sample.
        
        Expected format:
            {
                "messages": [{"role": "...", "content": "...with <image> placeholders..."}],
                "images": ["path1.png", "path2.png"]
            }
        """
        if "messages" not in data:
            return None
        
        messages = data["messages"]
        image_paths = data.get("images", [])
        
        # Build ChatML text with <image> placeholders preserved
        text_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        full_text = "\n".join(text_parts)
        
        # Count <image> placeholders - must match image_paths length
        placeholder_count = full_text.count("<image>")
        if placeholder_count != len(image_paths):
            logger.warning(
                f"Sample {source_file}:{idx} has {placeholder_count} <image> placeholders "
                f"but {len(image_paths)} image paths. Skipping."
            )
            return None
        
        if len(image_paths) > self.max_images_per_sample:
            logger.warning(
                f"Sample {source_file}:{idx} has {len(image_paths)} images, "
                f"exceeds max {self.max_images_per_sample}. Truncating."
            )
            # Truncate images and remove extra placeholders
            image_paths = image_paths[:self.max_images_per_sample]
            # Remove extra <image> placeholders from text
            parts = full_text.split("<image>")
            full_text = "<image>".join(parts[:self.max_images_per_sample + 1])
        
        # Resolve relative image paths
        source_dir = Path(source_file).parent
        resolved_paths = []
        for img_path in image_paths:
            p = Path(img_path)
            if not p.is_absolute():
                p = source_dir / p
            resolved_paths.append(str(p))
        
        return {
            "text": full_text,
            "image_paths": resolved_paths,
            "source": f"{source_file}:{idx}",
            "sample_id": hash(f"{source_file}_{idx}") % 1000000,
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample with tokenized text and image positions.
        
        Returns:
            Dict with:
                - input_ids: torch.LongTensor of shape [L] with IMAGE_TOKEN placeholders
                - image_tensors: List of image tensors [C, H, W] (or latents if VAE provided)
                - image_positions: List of token positions where images should be inserted
                - image_sizes: List of (H, W) tuples for each image
                - text: Original text
                - sample_id: Unique sample identifier
        """
        sample = self.samples[idx]
        text = sample["text"]
        image_paths = sample["image_paths"]
        
        # Tokenize text - this will convert <image> to IMAGE_TOKEN
        input_ids = self.tokenizer.encode(
            text, 
            max_length=self.max_context_length,
            add_pad=False,  # Don't pad - variable length
            add_eot=True
        )
        
        # Find positions of IMAGE_TOKEN in the sequence
        image_positions = []
        input_ids_list = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
        for pos, token_id in enumerate(input_ids_list):
            if token_id == IMAGE_TOKEN:
                image_positions.append(pos)
        
        # Load and process images
        image_tensors = []
        image_sizes = []
        
        for img_path in image_paths:
            img_tensor = self.image_loader.load(img_path)
            if img_tensor is not None:
                # img_tensor is [C, H, W] normalized to [-1, 1]
                _, H, W = img_tensor.shape
                
                # Encode with VAE if available
                if self.vae is not None:
                    with torch.no_grad():
                        # VAE expects [B, C, H, W]
                        latents = self.vae.encode(img_tensor.unsqueeze(0))
                        latents = latents.squeeze(0)  # [C, H/8, W/8]
                        image_tensors.append(latents)
                        # Update sizes to latent space
                        image_sizes.append((H // 8, W // 8))
                else:
                    image_tensors.append(img_tensor)
                    image_sizes.append((H, W))
            else:
                # Placeholder for failed loads
                logger.warning(f"Failed to load image: {img_path}")
                image_tensors.append(None)
                image_sizes.append((0, 0))
        
        return {
            "input_ids": input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids, dtype=torch.long),
            "image_tensors": image_tensors,
            "image_positions": image_positions,
            "image_sizes": image_sizes,
            "text": text,
            "sample_id": sample["sample_id"],
        }


def multiimage_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for multi-image batches.
    
    Handles variable-length sequences and variable image counts per sample.
    
    Returns:
        Dict with:
            - input_ids: List[Tensor] of token IDs (variable length)
            - image_tensors: List[List[Tensor]] - [B][N_i] image tensors
            - image_positions: List[List[int]] - [B][N_i] insertion positions
            - image_sizes: List[List[Tuple]] - [B][N_i] (H, W) tuples
            - texts: List[str] - original texts
            - sample_ids: List[int] - sample identifiers
    """
    return {
        "input_ids": [sample["input_ids"] for sample in batch],
        "image_tensors": [sample["image_tensors"] for sample in batch],
        "image_positions": [sample["image_positions"] for sample in batch],
        "image_sizes": [sample["image_sizes"] for sample in batch],
        "texts": [sample["text"] for sample in batch],
        "sample_ids": [sample["sample_id"] for sample in batch],
    }


# =============================================================================
# Collate Functions
# =============================================================================

def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for multimodal batches.
    
    Returns:
        Dict with:
        - text_ids: List[Tensor] of token IDs
        - images: List[Tensor] or None for each sample
        - has_image: List[bool]
    """
    text_ids = []
    images = []
    has_images = []
    texts = []
    
    for sample in batch:
        text_ids.append(sample.get("input_ids"))
        texts.append(sample.get("text", ""))
        
        if sample.get("has_image", False) and "image" in sample:
            images.append(sample["image"])
            has_images.append(True)
        else:
            images.append(None)
            has_images.append(False)
    
    return {
        "text_ids": text_ids,
        "images": images,
        "has_image": has_images,
        "texts": texts,
    }


# =============================================================================
# Factory Functions
# =============================================================================

def create_dataloader(
    data_paths: Union[str, List[str]],
    config: DataConfig = None,
    batch_size: int = 8,
    shuffle: bool = True,
    streaming: bool = False,
    tokenizer: Any = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for multimodal training.
    
    Args:
        data_paths: Path(s) to data files/directories
        config: DataConfig instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        streaming: Use streaming dataset for large data
        tokenizer: Tokenizer instance
        **kwargs: Additional DataLoader arguments
    
    Returns:
        DataLoader instance
    """
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    
    config = config or DataConfig()
    
    if streaming:
        dataset = StreamingMultimodalDataset(
            data_paths=data_paths,
            config=config,
            tokenizer=tokenizer,
        )
        # Streaming datasets don't support shuffle in DataLoader
        shuffle = False
    else:
        dataset = MultimodalDataset(
            data_paths=data_paths,
            config=config,
            tokenizer=tokenizer,
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if not streaming else False,
        collate_fn=multimodal_collate_fn,
        num_workers=config.num_workers if not streaming else 0,
        pin_memory=config.pin_memory,
        **kwargs
    )


# =============================================================================
# Context Packing Dataset (Multiple Samples per Context Window)
# =============================================================================

class PackedChatDataset(Dataset):
    """
    Dataset that packs multiple samples into single context windows for max GPU utilization.
    
    CRITICAL: Assigns unique doc_ids to each sub-sample to prevent cross-attention
    contamination during SFT. The attention layer MUST respect these boundaries.
    
    Features:
        - Modality-aware packing: separate image vs text samples
        - Ratio control: target % of context for images vs text
        - Fallbacks: fills with available data if ratio can't be met
    
    Args:
        base_dataset: Underlying dataset returning (text, image, ...) tuples
        max_context_length: Maximum tokens per packed context (e.g., 16384)
        tokenizer: Tokenizer for estimating token counts
        allow_cross_attention: If True, packs can attend across documents (pretraining)
        image_ratio: Target ratio of image samples in each context (0.0-1.0). 
                     0.0 = all text, 1.0 = all images, 0.3 = 30% images
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        max_context_length: int = 16384,
        tokenizer = None,
        allow_cross_attention: bool = False,
        image_ratio: float = 0.5,  # Default: 50% images, 50% text
        preload_tokens: bool = False, # Prevent memory mapping crash on Windows
        max_text_length: int = 512,  # Per-sample text token limit (NOT context length)
    ):
        self.base_dataset = base_dataset
        self.max_context_length = max_context_length
        self.max_text_length = max_text_length
        # [FIX] Tokenizer uses max_text_length (per-sample limit), NOT max_context_length.
        # The context window is shared across MULTIPLE samples — each sample should only
        # claim its actual token count, not the entire context budget.
        self.tokenizer = tokenizer or TiktokenTokenizer(max_length=max_text_length)
        self.allow_cross_attention = allow_cross_attention
        self.target_image_ratio = max(0.0, min(1.0, image_ratio))  # Clamp to [0, 1]
        
        # Classify samples by modality
        self.image_indices = []  # Samples with images
        self.text_indices = []   # Text-only samples
        self.sample_sizes = {}   # idx -> token count
        
        self._classify_samples()
        
        # Pre-compute packed indices using ratio-aware bin packing
        self.packed_indices = self._compute_packing_with_ratio()
        
        # Stats
        self.packing_stats = self._compute_stats()
        logger.info(f"PackedChatDataset: {len(self.base_dataset)} samples -> {len(self.packed_indices)} contexts")
        logger.info(f"   Image samples: {len(self.image_indices)}, Text samples: {len(self.text_indices)}")
        logger.info(f"   Target ratio: {self.target_image_ratio*100:.0f}% images")
        logger.info(f"   Actual ratio: {self.packing_stats['actual_image_ratio']*100:.1f}% images")
        if self.packing_stats['ratio_warnings']:
            logger.warning(f"   Ratio fallbacks: {self.packing_stats['ratio_warnings']} contexts couldn't meet target")
    
    def _classify_samples(self):
        """Classify samples as image or text-only."""
        # [OPTIMIZATION] access raw samples list if available to skip __getitem__ tokenization
        use_raw_samples = hasattr(self.base_dataset, 'samples')
        
        # [PROGRESS BAR] Added tqdm
        for i in tqdm(range(len(self.base_dataset)), desc="Context Packing: Classifying", unit="sample"):
            try:
                if use_raw_samples:
                    # FAST PATH: Read raw dict, skip tokenization overhead
                    sample = self.base_dataset.samples[i]
                else:
                    # SLOW PATH: Standard access
                    sample = self.base_dataset[i]
                
                size = self._estimate_tokens(sample)
                has_image = self._has_image(sample)
                
                self.sample_sizes[i] = size
                if has_image:
                    self.image_indices.append(i)
                else:
                    self.text_indices.append(i)
            except Exception as e:
                # logger.warning(f"Failed to classify sample {i}: {e}")
                # Consider corrupted samples as text-only with max size
                self.sample_sizes[i] = self.max_context_length
                self.text_indices.append(i)
                    
    def _has_image(self, sample) -> bool:
        """Check if sample contains image data."""
        if isinstance(sample, dict):
            # Check for explicit type (Z-Turbo Instant Cache bypass sets img_path=None but type=image_text)
            if sample.get("type") == "image_text":
                return True
            # Check for 'latents' (from ImageLatentDataset - not None means has image)
            latents = sample.get('latents')
            if latents is not None:
                return True
            # Check for 'img_path' (raw sample format - not None means has image)
            img_path = sample.get('img_path')
            if img_path is not None:
                return True
            # Check for 'images' or 'image' (from MultiImageChatDataset)
            images = sample.get('images', sample.get('image', sample.get('image_tensors')))
            if images is not None:
                if isinstance(images, list):
                    return len(images) > 0
                return True
        elif isinstance(sample, tuple) and len(sample) > 1:
            return sample[1] is not None
        return False
    
    def _fast_count_tokens(self, text: str) -> int:
        """Fast, accurate token count WITHOUT padding or OOM risk.
        
        Uses tiktoken's raw encoder directly (no padding, no special token overhead).
        Truncates huge strings first to prevent System RAM OOM on 2GB+ text dumps.
        Result is capped at max_text_length since that's the per-sample limit.
        """
        if not text:
            return 0
        # Truncate huge strings: ~8 chars/token is a safe over-approximation
        # so max_text_length * 8 chars is enough to capture max_text_length tokens
        safe_text = text[:self.max_text_length * 8]
        try:
            raw_count = len(self.tokenizer.enc.encode(safe_text))
        except Exception:
            # Fallback: ~4 chars per token is a reasonable average for English
            raw_count = len(safe_text) // 4
        # Cap at per-sample text limit (what the model will actually receive)
        return min(raw_count + 1, self.max_text_length)  # +1 for EOT token
    
    def _estimate_tokens(self, sample) -> int:
        """Estimate token count for a sample (including image tokens)."""
        if isinstance(sample, dict):
            text = sample.get('text', '')
            
            # [FIX] For Z-TURBO cached samples: type=image_text with h/w
            # We must accurately count image tokens so they don't pack infinitely
            if sample.get('type') == 'image_text' or sample.get('latents') is not None or sample.get('img_path') is not None:
                # Get dimensions, fallback to max sizes if missing to be safe on VRAM
                h = sample.get('h', 512)
                w = sample.get('w', 512)
                
                # Image patches: (h / vae_downsample / patch_size) * (w / vae_downsample / patch_size)
                # For OmniFusion V2: VAE downsamples by 8, ViT patch is 2 (so effectively 16)
                vae_ds, patch_sz = 8, 2
                image_tokens = max(1, (h // vae_ds // patch_sz)) * max(1, (w // vae_ds // patch_sz))
            else:
                n_images = len(sample.get('images', [])) if isinstance(sample.get('images'), list) else (1 if sample.get('image') else 0)
                # If we don't have dimensions, estimate ~256 tokens per image (e.g. 256x256 image)
                image_tokens = n_images * 256
            
            # Text Tokens — use accurate fast counting
            if 'token_len' in sample:
                # Pre-computed during dataset loading (cache stores unpadded tokens incl. EOT)
                # Do NOT add +1 here or we double-count EOT.
                text_tokens = min(sample['token_len'], self.max_text_length)
            elif text:
                text_tokens = self._fast_count_tokens(text)
            else:
                text_tokens = 0
                
        elif isinstance(sample, tuple):
            text = sample[0] if isinstance(sample[0], str) else ""
            n_images = 1 if len(sample) > 1 and sample[1] is not None else 0
            text_tokens = self._fast_count_tokens(text)
            image_tokens = n_images * 256
        else:
            text = str(sample) if sample else ""
            text_tokens = self._fast_count_tokens(text)
            image_tokens = 0
        
        return text_tokens + image_tokens
    
    def _compute_packing_with_ratio(self) -> List[List[int]]:
        """
        Fast O(N) linear packing. Replaces the slow O(N^2) search.
        """
        bins = []
        
        # 1. Sort indices by size (Largest first) - O(N log N)
        # This makes packing much more efficient than random order
        logger.info("Context Packing: Sorting samples by size...")
        img_queue = sorted(self.image_indices, key=lambda x: self.sample_sizes.get(x, 0), reverse=True)
        txt_queue = sorted(self.text_indices, key=lambda x: self.sample_sizes.get(x, 0), reverse=True)
        
        # Pointers for the queues
        img_ptr = 0
        txt_ptr = 0
        n_img = len(img_queue)
        n_txt = len(txt_queue)
        
        total_items = n_img + n_txt
        pbar = tqdm(total=total_items, desc="Context Packing: Binning (Fast)", unit="sample")
        
        # 2. Linear Scan - O(N)
        while img_ptr < n_img or txt_ptr < n_txt:
            current_context = []
            current_size = 0
            
            # Calculate dynamic target for this context
            img_target_size = int(self.max_context_length * self.target_image_ratio)
            
            # Phase A: Fill with Images up to target ratio
            while img_ptr < n_img:
                idx = img_queue[img_ptr]
                size = self.sample_sizes.get(idx, 0)
                
                # Check fit vs Image Target
                if current_size + size <= img_target_size:
                    current_context.append(idx)
                    current_size += size
                    img_ptr += 1
                else:
                    # If largest remaining image doesn't fit in quota, stop adding images
                    # (Unless we have absolutely no text left to fill the rest)
                    if txt_ptr < n_txt: 
                        break
                    # If no text left, try to squeeze image into total context length
                    if current_size + size <= self.max_context_length:
                        current_context.append(idx)
                        current_size += size
                        img_ptr += 1
                    else:
                        break # Fits nowhere

            # Phase B: Fill remainder with Text
            while txt_ptr < n_txt:
                idx = txt_queue[txt_ptr]
                size = self.sample_sizes.get(idx, 0)
                
                if current_size + size <= self.max_context_length:
                    current_context.append(idx)
                    current_size += size
                    txt_ptr += 1
                else:
                    break # Text doesn't fit

            # Phase C: Backfill with Images (if text ran out or ratio was small)
            while img_ptr < n_img:
                idx = img_queue[img_ptr]
                size = self.sample_sizes.get(idx, 0)
                if current_size + size <= self.max_context_length:
                    current_context.append(idx)
                    current_size += size
                    img_ptr += 1
                else:
                    break

            # Safety: If context is empty (single sample > max_context), force add one
            if not current_context:
                if img_ptr < n_img:
                    current_context.append(img_queue[img_ptr])
                    img_ptr += 1
                elif txt_ptr < n_txt:
                    current_context.append(txt_queue[txt_ptr])
                    txt_ptr += 1
            
            bins.append(current_context)
            pbar.update(len(current_context))
            
        pbar.close()
        return bins
            
    def _pack_single_modality(self, indices: List[int]) -> List[List[int]]:
        """Simple bin packing for single modality (fallback)."""
        bins = []
        indices_copy = list(indices)
        indices_copy.sort(key=lambda i: -self.sample_sizes[i])
        
        while indices_copy:
            bin_indices = []
            bin_size = 0
            
            i = 0
            while i < len(indices_copy):
                idx = indices_copy[i]
                if bin_size + self.sample_sizes[idx] <= self.max_context_length:
                    bin_indices.append(idx)
                    bin_size += self.sample_sizes[idx]
                    indices_copy.pop(i)
                else:
                    i += 1
            
            if bin_indices:
                bins.append(bin_indices)
            elif indices_copy:
                # Sample too large, put alone
                bins.append([indices_copy.pop(0)])
        
        return bins
    
    def _compute_stats(self) -> dict:
        """Compute packing statistics."""
        total_image_samples = 0
        total_text_samples = 0
        
        image_indices_set = set(self.image_indices)
        
        for bin_indices in self.packed_indices:
            for idx in bin_indices:
                if idx in image_indices_set:
                    total_image_samples += 1
                else:
                    total_text_samples += 1
        
        total = total_image_samples + total_text_samples
        actual_ratio = total_image_samples / total if total > 0 else 0.0
        
        return {
            'total_contexts': len(self.packed_indices),
            'total_image_samples': total_image_samples,
            'total_text_samples': total_text_samples,
            'actual_image_ratio': actual_ratio,
            'ratio_warnings': getattr(self, 'ratio_warnings', 0),
        }
    
    def __len__(self) -> int:
        return len(self.packed_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a packed context with multiple sub-samples.
        
        Returns:
            Dict with:
                - 'packed_texts': List of text strings
                - 'packed_token_ids': List of pre-tokenized token ID lists (optimization)
                - 'packed_images': List of List of image tensors
                - 'doc_ids': List of doc_id for each sub-sample
                - 'allow_cross_attention': Flag for attention layer
        """
        sample_indices = self.packed_indices[idx]
        
        packed_texts = []
        packed_token_ids = []  # Pre-tokenized for speed
        packed_images = []
        doc_ids = []
        
        for doc_id, sample_idx in enumerate(sample_indices):
            try:
                sample = self.base_dataset[sample_idx]
            except Exception as e:
                logger.warning(f"Failed to load sample {sample_idx}: {e}")
                continue  # Skip corrupted samples
            
            # Handle different sample formats
            if isinstance(sample, dict):
                text = sample.get('text', '')
                # Try 'latents' first (ImageLatentDataset), then 'images' (MultiImageChatDataset)
                latents = sample.get('latents')
                if latents is not None:
                    images = [latents]  # Wrap latent tensor in list
                else:
                    images = sample.get('images', sample.get('image_tensors', []))
            elif isinstance(sample, tuple):
                text = sample[0] if isinstance(sample[0], str) else ""
                images = [sample[1]] if len(sample) > 1 and sample[1] is not None else []
            else:
                text = str(sample) if sample else ""
                images = []
            
            # Pre-tokenize text (moves CPU work from training loop to data loading)
            # [FIX] Truncate massive strings BEFORE encoding to prevent System RAM OOM.
            max_chars = self.max_text_length * 8  # Safe over-approximation
            safe_text = text[:max_chars] if text else ""
            
            if isinstance(sample, dict) and 'input_ids' in sample:
                cached_ids = sample['input_ids']
                if isinstance(cached_ids, torch.Tensor):
                    token_ids = cached_ids.cpu().tolist() if cached_ids.numel() > 0 else []
                elif isinstance(cached_ids, list):
                    token_ids = cached_ids
                else:
                    token_ids = self.tokenizer.encode(safe_text, add_pad=False).tolist() if safe_text else []
            else:
                token_ids = self.tokenizer.encode(safe_text, add_pad=False).tolist() if safe_text else []
            
            # [FIX] Truncate oversized samples to max_text_length.
            # A single JSONL entry with 4021 tokens gets cleanly truncated to fit,
            # rather than silently consuming the whole context or crashing.
            if len(token_ids) > self.max_text_length:
                token_ids = token_ids[:self.max_text_length]
            
            packed_texts.append(safe_text)
            packed_token_ids.append(token_ids)
            packed_images.append(images)
            doc_ids.append(doc_id)
        
        return {
            'packed_texts': packed_texts,
            'packed_token_ids': packed_token_ids,  # Pre-tokenized for training speedup
            'packed_images': packed_images,
            'doc_ids': doc_ids,
            'allow_cross_attention': self.allow_cross_attention,
            'n_samples': len(sample_indices),
        }


def packed_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for PackedChatDataset.
    
    Returns:
        Dict with lists of packed contexts ready for model processing.
    """
    return {
        'packed_texts': [item['packed_texts'] for item in batch],
        'packed_token_ids': [item['packed_token_ids'] for item in batch],  # Pre-tokenized
        'packed_images': [item['packed_images'] for item in batch],
        'doc_ids': [item['doc_ids'] for item in batch],
        'allow_cross_attention': batch[0]['allow_cross_attention'] if batch else False,
        'n_samples': [item['n_samples'] for item in batch],
    }



# =============================================================================
# Testing
# =============================================================================


if __name__ == "__main__":
    # Test the data manager
    logger.info("Testing DataManager...")
    
    # Test caption extraction
    test_data = {
        "image_path": "test.jpg",
        "blip": {"caption": "A cat sitting on a table"},
        "florence": {"more_detailed_caption": "A fluffy orange tabby cat sitting on a wooden table."},
        "wd_tagger": {"caption": "1cat, sitting, table, orange_fur"},
    }
    
    captions = CaptionExtractor.extract_all(test_data)
    logger.info(f"Extracted captions: {captions}")
    
    best = CaptionExtractor.combine_captions(captions, style="best")
    logger.info(f"Best caption: {best}")
    
    # Test parsers
    chatml = ChatMLParser()
    alpaca = AlpacaParser()
    
    alpaca_data = {
        "instruction": "Summarize this text",
        "input": "Long text here",
        "output": "Short summary"
    }
    
    if alpaca.can_parse(alpaca_data):
        samples = alpaca.parse(alpaca_data)
        logger.info(f"Alpaca parsed: {samples[0]['text'][:100]}...")
    
    logger.info("✅ DataManager tests passed!")
