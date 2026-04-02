"""
OmniFusion-X V2: Unified Multimodal Foundation Model (Native Resolution)
================================================================================
Based on State-of-the-Art Architectures:
- Z-Image (S3-DiT, QK-Norm, Native Resolution, Sandwich-Norm)
- NiT (Native Resolution Packing, FlashAttention-2)
- SD3 (MM-DiT Structure, Timestep Shifting, Logit-Normal Sampling)
- Hunyuan 3.0 (Generalized RoPE)

Features:
- Native Resolution Packing (No Padding) via FlashAttention VarLen
- 3D Axial Rotary Embeddings (Time/Text, Height, Width)
- Shared Attention / Separate MLP (MM-DiT)
- Grouped Query Attention (GQA) - Optimization #1
- Tied Word Embeddings - Optimization #6
- AdaLN-Zero Conditioning (Low-Rank)
- Rectified Flow Matching (Linear Velocity) with Logit-Normal Sampling
- Async Resource Loading & Mixed Precision Support
- Native CUDA Graph Support
- Dynamic Context Length based on Hardware
- FlexAttention Support (PyTorch 2.5+)
- Stabilization: Sandwich-Norm, LayerScale, QK-Norm, Soft-Capping, DropPath

Author: OmniFusion Team
License: Apache 2.0
"""

import os
import math
import copy
import time
import queue
import threading
import logging
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Dict, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast

# -----------------------------------------------------------------------------
# 0. Dependencies & Environment Setup
# -----------------------------------------------------------------------------

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OmniFusionV2")

# Check for Flash Attention 2
try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
    logger.info("✅ FlashAttention-2 Available.")
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# Check for FlexAttention
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
    logger.info("✅ FlexAttention Available.")
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False

# Check for xFormers
try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
    logger.info("✅ xFormers Available.")
except ImportError:
    XFORMERS_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------

@dataclass
class KVCache:
    """
    Stores cached Key/Value tensors for efficient autoregressive generation.
    Used during text generation to avoid recomputing attention for past tokens.
    """
    keys: List[torch.Tensor]    # [n_layers] of [cached_tokens, n_kv_heads, head_dim]
    values: List[torch.Tensor]  # [n_layers] of [cached_tokens, n_kv_heads, head_dim]
    seq_len: int = 0            # Current number of cached tokens
    
    @staticmethod
    def empty(n_layers: int) -> 'KVCache':
        """Create an empty cache for n_layers."""
        return KVCache(keys=[None] * n_layers, values=[None] * n_layers, seq_len=0)
    
    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a given layer and return full K/V tensors.
        new_k, new_v: [new_tokens, n_kv_heads, head_dim]
        Returns: (full_k, full_v) with cached + new tokens
        """
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = new_k
            self.values[layer_idx] = new_v
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], new_k], dim=0)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], new_v], dim=0)
        
        return self.keys[layer_idx], self.values[layer_idx]

@dataclass
class OmniConfigV2:
    """
    Configuration for the OmniFusion-X V2 Model.
    Targeting ~50M Parameters for the Base Model (Excluding VAE/Text Enc).
    """
    # Architecture - Scaled for ~56M Params
    d_model: int = 768              # Hidden dimension
    n_layers: int = 8               # Depth
    n_heads: int = 12               # Attention Heads
    n_kv_heads: Optional[int] = None # GQA KV Heads (Optim #1). If None, defaults to n_heads (MHA)
    head_dim: int = 64              # Dimension per head (768/12 = 64)
    mlp_ratio: float = 4.0          # MLP expansion
    
    # Inputs
    vocab_size: int = 100352         # optimized for Tensor Cores (128x)
    # Max seq len is now dynamic, this is a soft cap for RoPE precalc
    max_seq_len: int = 32768        
    
    # Vision
    patch_size: int = 2             # Latent Patch Size
    in_channels: int = 16           # VAE Latent Channels (Flux VAE has 16 channels)
    vae_scale_factor: float = 0.3611 # Adjusted for Unit Variance
    max_resolution_pixels: int = 327680 # Default for RTX 3070 Ti (8GB)
    
    # RoPE
    rope_base: float = 10000.0      # Base frequency
    
    # Conditioning
    time_embed_dim: int = 256       # Timestep embedding dimension

    # Text pooling / conditioning injection (global text -> image AdaLN shortcut)
    # - text_pooling="mean": mean-pool token embeddings (excluding PAD) to get a pooled text vector.
    # - text_pooling="attn": learn attention weights over tokens to pool more discriminatively.
    # pooled_text_cond_scale scales the pooled vector before it is added into image token conditioning.
    # pooled_text_drop_prob optionally drops pooled conditioning during training to encourage token-level conditioning.
    text_pooling: str = "mean"      # "mean" | "attn"
    pooled_text_cond_scale: float = 1.0
    pooled_text_drop_prob: float = 0.0  # 0..1, applied per-sample during training
    
    # Optimization
    dropout: float = 0.0
    attention_dropout: float = 0.0
    qk_norm: bool = True            # Enable QK-Norm (Z-Image stability)
    rms_norm_eps: float = 1e-6      # Llama 3 style
    tie_word_embeddings: bool = True # Optimization #6
    
    # Stabilization
    sandwich_norm: bool = True      # Z-Image Sandwich Norm
    layer_scale_init: float = 1.0   # [FIX] Changed from 1e-5. AdaLN handles zero-init stability.
    drop_path_rate: float = 0.1     # Stochastic Depth rate
    attention_logit_cap: float = 50.0 # Soft-capping for attention logits (Gemma 2)
    
    # System
    dtype: str = "bfloat16"         # Preferred training dtype
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_checkpointing: bool = False
    regional_compile: bool = False  # Optim #10: Regional Compilation (PyTorch 2.5+)
    lazy_logits: bool = False       # If True, skip materializing full token logits during training (saves VRAM for long contexts)
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.torch_dtype = getattr(torch, self.dtype)
        
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads for GQA"
        
        # Hardware-aware dynamic context length adjustment
        if self.device == "cuda":
            try:
                # Estimate Safe Max Context based on VRAM
                t = torch.cuda.get_device_properties(0).total_memory
                # Heuristic: 1GB ~ 2k tokens for this size model with optimizer states
                # Adjust based on rough parameter count
                estimated_safe_len = int((t / (1024**3)) * 2048)
                if estimated_safe_len < self.max_seq_len:
                    logger.info(f"Hardware Constraint: Adjusting max_seq_len from {self.max_seq_len} to {estimated_safe_len}")
                    self.max_seq_len = estimated_safe_len
            except Exception as e:
                logger.warning(f"Could not determine VRAM for dynamic context: {e}")

        # Validate text pooling options (kept lightweight for torch.compile friendliness).
        if self.text_pooling not in ("mean", "attn"):
            logger.warning(f"Unknown text_pooling={self.text_pooling!r}; falling back to 'mean'.")
            self.text_pooling = "mean"
        try:
            self.pooled_text_cond_scale = float(self.pooled_text_cond_scale)
        except Exception:
            self.pooled_text_cond_scale = 1.0
        try:
            self.pooled_text_drop_prob = float(self.pooled_text_drop_prob)
        except Exception:
            self.pooled_text_drop_prob = 0.0
        if self.pooled_text_drop_prob < 0.0:
            self.pooled_text_drop_prob = 0.0
        if self.pooled_text_drop_prob > 1.0:
            self.pooled_text_drop_prob = 1.0

# -----------------------------------------------------------------------------
# 2. Core Layers (Norms, Activations, Embeddings)
# -----------------------------------------------------------------------------

class OmniRMSNorm(nn.Module):
    """
    Llama 3 / Z-Image style RMSNorm. (Optim #3)
    Uses fused flash_attn kernel when available for 10-15% speedup.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        
        # Try to use fused RMSNorm from flash_attn
        self._use_fused = False
        try:
            from flash_attn.ops.rms_norm import rms_norm as flash_rms_norm
            self._flash_rms_norm = flash_rms_norm
            self._use_fused = True
        except ImportError:
            self._flash_rms_norm = None

    def forward(self, x):
        if self._use_fused and self._flash_rms_norm is not None:
            # Use fused kernel (single CUDA launch)
            return self._flash_rms_norm(x, self.weight, self.eps)
        else:
            # Fallback to standard implementation
            input_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return self.weight.to(input_dtype) * x.to(input_dtype)

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit. Standard in modern LLMs/DiTs. (Optim #2)
    """
    def __init__(self, in_features, hidden_features, out_features, bias=False):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations. (Optim #10)
    """
    def __init__(self, hidden_dim, freq_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embedding_size, hidden_dim, bias=False), # Bias elim
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.freq_embedding_size = freq_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # [FIX] Scale t from [0,1] to [0,1000] for proper sinusoidal frequency spread.
        # Without this, cos/sin arguments are ~0 for all frequencies, making embeddings
        # constant across timesteps — the model becomes "t-blind".
        t_scaled = t * 1000.0
        t_freq = self.timestep_embedding(t_scaled, self.freq_embedding_size)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)

# -----------------------------------------------------------------------------
# 2.0b Text Pooling Helpers
# -----------------------------------------------------------------------------

class TextAttentionPool(nn.Module):
    """
    Learnable attention pooling over a token sequence.

    This provides a pooled text vector c = sum_i softmax(score_i) * h_i,
    where scores are produced by a small linear head. It can be more
    discriminative than mean pooling for long tag lists.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, 1, bias=False)
        # Start from uniform pooling (equivalent to mean over non-pad tokens)
        # so enabling this mode does not introduce a random pooled shortcut at init.
        nn.init.zeros_(self.score.weight)

    def forward(self, x: torch.Tensor, non_pad: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [L, D] token embeddings
            non_pad: [L] or [L,1] bool/float mask (True/1.0 for real tokens)
        Returns:
            pooled: [1, D]
        """
        if x.dim() != 2:
            raise ValueError(f"TextAttentionPool expects x as [L,D], got {tuple(x.shape)}")
        if x.numel() == 0:
            return x.new_zeros((1, x.shape[-1]))

        if non_pad.dim() == 2:
            non_pad = non_pad.squeeze(-1)
        if non_pad.dtype != torch.bool:
            non_pad = non_pad > 0

        # If everything is masked (shouldn't happen with sane tokenization), fall back to mean.
        if not bool(non_pad.any()):
            return x.mean(dim=0, keepdim=True)

        x_norm = self.norm(x)
        scores = self.score(x_norm).squeeze(-1)  # [L]
        scores = scores.masked_fill(~non_pad, float("-inf"))

        # Softmax in float32 for stability, then cast weights back to x dtype.
        w = torch.softmax(scores.float(), dim=0).to(dtype=x.dtype)  # [L]
        pooled = (w.unsqueeze(-1) * x).sum(dim=0, keepdim=True)  # [1,D]
        return pooled

# -----------------------------------------------------------------------------
# 2.1 Stabilization Layers (LayerScale, DropPath)
# -----------------------------------------------------------------------------

class LayerScale(nn.Module):
    """
    LayerScale: Learnable per-channel scaling for residual branches.
    Initialize with small epsilon to improve deep network convergence.
    """
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

# -----------------------------------------------------------------------------
# 3. Advanced Positional Embeddings (3D Axial RoPE)
# -----------------------------------------------------------------------------

class AxialRoPE3D(nn.Module):
    """
    Implements 3D Axial Rotary Positional Embeddings.
    Splits head_dim into 3 axes: Time, Height, Width for video/long-context support.
    
    This enables the model to be 'time-aware' and handle sequences with 100+ images.
    """
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        
        # SAFETY CHECK: 
        # We need at least 2 dimensions per half (complex numbers) * 3 axes = 6 total
        assert head_dim >= 6, f"head_dim must be >= 6 for 3-axis RoPE, got {head_dim}"
        assert head_dim % 2 == 0, "Head dim must be even for RoPE"
        
        self.head_dim = head_dim
        self.base = base
        
        # Calculate dimensions for each axis
        # We process complex numbers, so we work with half the head_dim
        d_half = head_dim // 2
        self.d_time = d_half // 3
        self.d_height = d_half // 3
        self.d_width = d_half - (2 * self.d_time)  # Give remainder to width to ensure sum is d_half
        
        # Precompute frequencies for the largest possible axis
        max_axis_dim = max(self.d_time, self.d_height, self.d_width)
        dim = max_axis_dim * 2  # Real dimension size
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        """
        Apply 3D Axial RoPE.
        q, k: [Total_Tokens, N_Heads, Head_Dim] (Packed)
        positions: [Total_Tokens, 3] -> (Time, Height, Width) coordinates
        """
        # 1. Extract Positions
        t_idx = positions[:, 0]
        h_idx = positions[:, 1]
        w_idx = positions[:, 2]
        
        # 2. Get Frequencies for each axis
        # Slice inv_freq to match each axis dimension
        inv_freq = self.inv_freq.to(q.device)
        freqs_t = torch.outer(t_idx, inv_freq[:self.d_time])
        freqs_h = torch.outer(h_idx, inv_freq[:self.d_height])
        freqs_w = torch.outer(w_idx, inv_freq[:self.d_width])
        
        # 3. Create Real Rotary Embeddings (Cos and Sin)
        cos_t, sin_t = torch.cos(freqs_t.float()), torch.sin(freqs_t.float())
        cos_h, sin_h = torch.cos(freqs_h.float()), torch.sin(freqs_h.float())
        cos_w, sin_w = torch.cos(freqs_w.float()), torch.sin(freqs_w.float())
        
        cos_emb = torch.cat([cos_t, cos_h, cos_w], dim=-1).unsqueeze(1) # [Tokens, 1, d_half]
        sin_emb = torch.cat([sin_t, sin_h, sin_w], dim=-1).unsqueeze(1)
        
        # 4. Unpack Q/K into Real and Imag components
        d_half = q.shape[-1] // 2
        q_reshaped = q.float().reshape(*q.shape[:-1], d_half, 2)
        k_reshaped = k.float().reshape(*k.shape[:-1], d_half, 2)
        
        q_r, q_i = q_reshaped[..., 0], q_reshaped[..., 1]
        k_r, k_i = k_reshaped[..., 0], k_reshaped[..., 1]
        
        # 5. Apply Rotations using Real Operations
        q_out_r = q_r * cos_emb - q_i * sin_emb
        q_out_i = q_r * sin_emb + q_i * cos_emb
        
        k_out_r = k_r * cos_emb - k_i * sin_emb
        k_out_i = k_r * sin_emb + k_i * cos_emb
        
        # 6. Repack and Flatten
        q_out = torch.stack([q_out_r, q_out_i], dim=-1).flatten(2).to(q.dtype)
        k_out = torch.stack([k_out_r, k_out_i], dim=-1).flatten(2).to(k.dtype)
        
        return q_out, k_out

# -----------------------------------------------------------------------------
# 4. Attention Mechanism (FlashAttention Packed + GQA + FlexAttention)
# -----------------------------------------------------------------------------

def document_mask_mod(b, h, q_idx, kv_idx, doc_ids):
    """
    FlexAttention mask mod for document masking (packed sequences).
    """
    return doc_ids[q_idx] == doc_ids[kv_idx]

class PackedSelfAttention(nn.Module):
    """
    Self-Attention block supporting:
    - QK Norm (Stability) (Optim #8)
    - FlashAttention-2 VarLen (Native Packing) (Optim #26)
    - FlexAttention (PyTorch 2.5+)
    - GQA (Optim #1)
    - Soft-Capping (Stability)
    """
    def __init__(self, config: OmniConfigV2):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.logit_cap = config.attention_logit_cap
        
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        
        # Projections (Bias elimination Optim #7)
        self.q_proj = nn.Linear(config.d_model, self.q_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.kv_dim, bias=False)
        self.proj = nn.Linear(self.q_dim, config.d_model, bias=False)
        
        # QK Normalization (Optim #8)
        if config.qk_norm:
            self.q_norm = OmniRMSNorm(self.head_dim)
            self.k_norm = OmniRMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            
    def forward(self, x, rope_func, positions, cu_seqlens, max_seqlen, doc_ids=None, causal=False,
                mod_mask=None, kv_cache: Optional['KVCache'] = None, layer_idx: int = 0, block_mask=None):
        """
        x: [Total_Tokens, D] (Packed)
        mod_mask: [Total_Tokens] - 0.0 for text, 1.0 for image
        kv_cache: Optional KVCache for incremental decoding
        layer_idx: Layer index for cache storage
        """
        total_tokens, _ = x.shape
        
        # 1. Project
        q = self.q_proj(x) # [T, n_heads * head_dim]
        k = self.k_proj(x) # [T, n_kv_heads * head_dim]
        v = self.v_proj(x) # [T, n_kv_heads * head_dim]
        
        # 2. Reshape for Heads
        q = q.view(total_tokens, self.n_heads, self.head_dim)
        k = k.view(total_tokens, self.n_kv_heads, self.head_dim)
        v = v.view(total_tokens, self.n_kv_heads, self.head_dim)
        
        # 3. QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # 4. Apply RoPE
        q, k = rope_func(q, k, positions)
        
        # 5. KV Cache handling (for incremental generation)
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)
        
        # 6. Detect Packed Sequences (multiple documents in single context)
        # This is critical for SFT - packed documents MUST NOT cross-attend
        is_packed = False
        if doc_ids is not None:
            valid_doc_ids = doc_ids[doc_ids >= 0]  # Exclude padding
            if len(valid_doc_ids) > 0:
                unique_docs = torch.unique(valid_doc_ids)
                is_packed = len(unique_docs) > 1
        
        # 7. Attention Dispatch
        # Priority: FlexAttention > FlashAttn2 (VarLen) > xFormers (MemEff) > PyTorch SDPA
        # CRITICAL: If is_packed=True, we MUST use FlexAttention for document isolation
        
        # A. FlexAttention (PyTorch 2.5+) - REQUIRED for packed sequences
        if FLEX_ATTENTION_AVAILABLE and doc_ids is not None and q.is_cuda and kv_cache is None:
            try:
                # FlexAttention expects (B, H, S, D). Here B=1, S=Total_Tokens.
                q_flex = q.permute(1, 0, 2).unsqueeze(0) # [1, H, T, D]
                k_flex = k.permute(1, 0, 2).unsqueeze(0) # [1, H_kv, T, D]
                v_flex = v.permute(1, 0, 2).unsqueeze(0)
                
                # Soft-capping via score_mod (FlexAttention feature)
                def soft_cap_mod(score, b, h, q_idx, kv_idx):
                    return self.logit_cap * torch.tanh(score / self.logit_cap)

                # Use precomputed BlockMask when provided (saves per-layer create_block_mask work).
                current_mask = block_mask
                if current_mask is None:
                    t_pos = positions[:, 0]

                    # === HYBRID DOCUMENT + TEMPORAL CAUSAL MASK ===
                    def hybrid_block_mask(b, h, q_idx, kv_idx):
                        # 1. Document boundary check (stay within same packed sample)
                        same_doc = doc_ids[q_idx] == doc_ids[kv_idx]
                        # Exclude padding tokens (doc_id == -1) from attention
                        valid_docs = (doc_ids[q_idx] >= 0) & (doc_ids[kv_idx] >= 0)

                        # 2. Strict Temporal Causality
                        # Guarantees causality for text tokens while allowing intra-image (same t_pos)
                        # tokens bidirectional visibility.
                        time_causal = t_pos[q_idx] >= t_pos[kv_idx]
                        valid_attn = time_causal if causal else True

                        return same_doc & valid_docs & valid_attn

                    current_mask = create_block_mask(
                        hybrid_block_mask,
                        B=1,
                        H=None,
                        Q_LEN=total_tokens,
                        KV_LEN=total_tokens,
                        device=q.device,
                    )
                
                # Apply soft capping if configured
                score_mod = soft_cap_mod if self.logit_cap > 0 else None
                
                x_out = flex_attention(q_flex, k_flex, v_flex, block_mask=current_mask, score_mod=score_mod)
                
                # Reshape back
                x_out = x_out.squeeze(0).permute(1, 0, 2).reshape(total_tokens, -1)
                return self.proj(x_out)
            except Exception as e:
                # FlexAttention can fail to compile/dispatch on some systems. Don't silently swallow it:
                # fallback paths may be slower and/or have different masking behavior.
                if not getattr(self, "_flexattention_warned", False):
                    logger.warning(f"FlexAttention failed; falling back to other attention backends. Error: {e}")
                    self._flexattention_warned = True
                pass
        
        # CRITICAL SAFETY CHECK: Packed sequences REQUIRE FlexAttention for document isolation
        # If FlexAttention failed/unavailable and we have packed docs, this is UNSAFE for SFT
        if is_packed and not getattr(self, '_allow_cross_attention', False):
            if not FLEX_ATTENTION_AVAILABLE:
                raise RuntimeError(
                    f"Context packing detected ({len(torch.unique(doc_ids[doc_ids >= 0]))} docs) but "
                    f"FlexAttention is unavailable (requires PyTorch 2.5+). "
                    f"FlashAttn/SDPA cannot enforce document isolation, risking cross-contamination during SFT. "
                    f"Either upgrade PyTorch or disable context packing (--no-context-pack). "
                    f"For pretraining where cross-attention is acceptable, use --allow-cross-attention."
                )
            # If we reach here, FlexAttention was available but failed - also an error
            raise RuntimeError(
                f"FlexAttention failed but is required for packed sequences. "
                f"Cannot fall back to FlashAttn/SDPA as they don't support document masking."
            )

        # B. Flash Attention 2 (VarLen)
        # Requires: fp16/bf16, CUDA
        # [FIX] FlashAttn 2 does NOT support the Hybrid Mask (Causal Text + Bi-Dir Image).
        # We ONLY use it if the batch is purely one modality.
        # If images AND text are present, we MUST skip FlashAttn to ensure text remains causal.
        # ALSO: Skip for packed sequences (they need document masking)
        is_hybrid = (mod_mask is not None) and ((mod_mask == 0.0).any() and (mod_mask == 1.0).any())
        
        if FLASH_ATTN_AVAILABLE and not is_hybrid and not is_packed and q.dtype in [torch.float16, torch.bfloat16] and q.is_cuda and kv_cache is None:
            try:
                x_out = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    dropout_p=0.0,
                    softmax_scale=None,
                    causal=causal  # Trust the global causal flag for pure batches
                )
                x_out = x_out.reshape(total_tokens, -1)
                return self.proj(x_out)
            except Exception as e:
                pass

        # D. PyTorch SDPA (Scaled Dot Product Attention)
        # GQA Expansion needed if not natively supported by backend
        
        # Expand K, V for GQA if needed
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        
        # 1. Identify Batch Size from cu_seqlens
        B_size = len(cu_seqlens) - 1
        
        # SPECIAL PATH: When using KV cache, Q length differs from K/V length
        # In this case, K and V already contain the full cached sequence
        if kv_cache is not None:
            # For incremental decoding: Q is [new_tokens, H, D], K/V are [cached+new, H_kv, D]
            # Use simple SDPA for single sequence generation
            q_gen = q.unsqueeze(0).transpose(1, 2)  # [1, H, new_tokens, D]
            k_gen = k.unsqueeze(0).transpose(1, 2)  # [1, H_kv, cached+new, D]
            v_gen = v.unsqueeze(0).transpose(1, 2)  # [1, H_kv, cached+new, D]
            
            # Expand K/V for GQA
            if self.n_kv_heads != self.n_heads:
                n_rep = self.n_heads // self.n_kv_heads
                k_gen = k_gen.repeat_interleave(n_rep, dim=1)
                v_gen = v_gen.repeat_interleave(n_rep, dim=1)
            
            # Causal mask: new Q tokens can only attend to past K/V tokens.
            # For 1-token generation, is_causal=False is fine (it attends to all history).
            # For prefill (many tokens), is_causal must match the 'causal' argument.
            is_gen_causal = causal if q_gen.shape[-2] > 1 else False
            
            # Use Manual Attention if logit-capping is enabled, else use SDPA
            if self.logit_cap > 0:
                scale = 1.0 / math.sqrt(self.head_dim)
                attn_scores = torch.matmul(q_gen, k_gen.transpose(-2, -1)) * scale
                attn_scores = self.logit_cap * torch.tanh(attn_scores / self.logit_cap)
                
                if is_gen_causal:
                    L_q, L_k = attn_scores.shape[-2:]
                    mask = torch.triu(torch.ones(L_q, L_k, device=q.device), diagonal=L_k - L_q + 1).bool()
                    attn_scores.masked_fill_(mask, float('-inf'))
                
                attn_probs = F.softmax(attn_scores, dim=-1)
                o_gen = torch.matmul(attn_probs, v_gen)
            else:
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                    o_gen = F.scaled_dot_product_attention(q_gen, k_gen, v_gen, is_causal=is_gen_causal)
            
            x_out = o_gen.transpose(1, 2).squeeze(0).flatten(1)  # [new_tokens, H*D]
            return self.proj(x_out)
        
        # Manual unbatching for SDPA to handle variable lengths
        outputs = []
        for i in range(B_size):
            start = cu_seqlens[i]
            end = cu_seqlens[i+1]
            
            # [Seq, H, D] -> [1, H, Seq, D]
            q_slice = q[start:end].unsqueeze(0).transpose(1, 2)
            k_slice = k[start:end].unsqueeze(0).transpose(1, 2)
            v_slice = v[start:end].unsqueeze(0).transpose(1, 2)

            # === FIX: Hybrid Mask Construction for SDPA ===
            L = q_slice.shape[2]
            
            # 1. Base Causal Mask (Lower Triangular)
            # Mask positions where col > row (Future)
            causal_mask = torch.triu(torch.ones((L, L), device=q.device, dtype=torch.bool), diagonal=1)
            
            # 2. Modality Logic (Text = 0.0, Image = 1.0)
            # IMPORTANT: In diffusion training, image tokens are noisy latents. Allowing TEXT queries to attend to
            # image keys can inject noise into the text pathway. We always block Text->Image here in the SDPA fallback.
            m_slice = mod_mask[start:end]  # [L]
            is_text_q = (m_slice == 0.0).unsqueeze(1)  # [L,1]
            is_text_k = (m_slice == 0.0).unsqueeze(0)  # [1,L]
            is_img_k = (m_slice == 1.0).unsqueeze(0)   # [1,L]

            # 1) Text-to-Text: apply causal mask (mask future tokens only)
            text_causal_block = (is_text_q & is_text_k) & causal_mask
            if not causal:
                text_causal_block = torch.zeros_like(text_causal_block)

            # 2) Text-to-Image: always blocked in SDPA fallback
            text_to_img_block = is_text_q & is_img_k

            # 3) Final mask (True = masked out / -inf)
            final_mask = text_causal_block | text_to_img_block
            
            # Manual Attention with Soft-Capping
            if self.logit_cap > 0:
                scale = 1.0 / math.sqrt(self.head_dim)
                attn_scores = torch.matmul(q_slice, k_slice.transpose(-2, -1)) * scale
                attn_scores = self.logit_cap * torch.tanh(attn_scores / self.logit_cap)
                
                # Apply Boolean Mask (True = -inf)
                attn_scores.masked_fill_(final_mask, float('-inf'))
                
                attn_probs = F.softmax(attn_scores, dim=-1)
                o_slice = torch.matmul(attn_probs, v_slice)
            else:
                # Standard SDPA supports attn_mask (True values are masked out)
                # Ideally convert to float bias for safety:
                attn_bias = torch.zeros_like(final_mask, dtype=q.dtype)
                attn_bias.masked_fill_(final_mask, float('-inf'))
                
                # FORCE Hardware Accelerated Attention ONLY to prevent OOM
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                    o_slice = F.scaled_dot_product_attention(q_slice, k_slice, v_slice, attn_mask=attn_bias)
            
            outputs.append(o_slice.transpose(1, 2).flatten(2)) # [1, L, H*D]
            
        x_out = torch.cat(outputs, dim=1).squeeze(0)
        real_token_count = int(cu_seqlens[-1].item()) if len(cu_seqlens) > 0 else x_out.shape[0]
        padding_len = total_tokens - real_token_count
        if padding_len > 0:
            # cu_seqlens intentionally excludes tail block-padding tokens.
            # Preserve the full packed shape by appending zero attention output
            # for those inert padding slots before the output projection.
            pad_out = torch.zeros(padding_len, x_out.shape[1], device=x_out.device, dtype=x_out.dtype)
            x_out = torch.cat([x_out, pad_out], dim=0)
        return self.proj(x_out)

# -----------------------------------------------------------------------------
# 5. MM-DiT Block (Shared Attn, Separate MLPs, Sandwich Norm, Low-Rank AdaLN)
# -----------------------------------------------------------------------------

class AdaLNZero(nn.Module):
    """
    Modulation layer.
    Using Low-Rank Factorization for parameter efficiency and stability.
    AdaLN-Zero: Start with small modulation that grows during training.
    """
    def __init__(self, d_model):
        super().__init__()
        self.silu = nn.SiLU()
        # Factorized projection (Bottleneck)
        # d_model -> d_model // 4 -> 6 * d_model
        rank = max(d_model // 4, 32)
        self.proj_down = nn.Linear(d_model, rank, bias=False)
        self.act = nn.SiLU()
        self.proj_up = nn.Linear(rank, 6 * d_model, bias=True)
        
        # Proper initialization for gradient flow
        # proj_down: Xavier for good signal propagation
        nn.init.xavier_uniform_(self.proj_down.weight)
        
        # [FIX] Use small normal init instead of strict zeros.
        # With image_head also zero-initialized, strict zero gates create a
        # gradient deadlock: gates=0 → blocks=identity → image_head gets zero
        # gradient → gates never open. std=0.02 cracks them open enough for
        # gradient signal while keeping blocks near-identity for stability.
        nn.init.normal_(self.proj_up.weight, std=0.02)
        nn.init.zeros_(self.proj_up.bias)
        
    def forward(self, c):
        # c: [Total_Tokens, D]
        x = self.silu(c)
        x = self.proj_down(x)
        x = self.act(x)
        x = self.proj_up(x)
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = x.chunk(6, dim=-1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

class MMDiTBlock(nn.Module):
    """
    Single-Stream Block with Shared Attention but Separate MLPs for Modalities.
    Stabilized with Sandwich-Norm, LayerScale, DropPath.
    """
    def __init__(self, config: OmniConfigV2):
        super().__init__()
        self.config = config
        
        # Split Attention Pre-Norm (Text and Image have different statistics)
        self.norm1_text = OmniRMSNorm(config.d_model, config.rms_norm_eps)
        self.norm1_img = OmniRMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = PackedSelfAttention(config)
        self.ls1 = LayerScale(config.d_model, config.layer_scale_init)
        self.drop_path1 = DropPath(config.drop_path_rate) if config.drop_path_rate > 0 else nn.Identity()
        
        # Sandwich Norm (Extra norm before MLP)
        if config.sandwich_norm:
            self.sandwich_norm = OmniRMSNorm(config.d_model, config.rms_norm_eps)
        else:
            self.sandwich_norm = nn.Identity()
        
        # Separate MLPs
        # 1. Text MLP
        self.norm2_text = OmniRMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp_text = SwiGLU(config.d_model, int(config.d_model * config.mlp_ratio), config.d_model, bias=False)
        
        # 2. Image MLP
        self.norm2_img = OmniRMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp_img = SwiGLU(config.d_model, int(config.d_model * config.mlp_ratio), config.d_model, bias=False)
        
        self.ls2 = LayerScale(config.d_model, config.layer_scale_init)
        self.drop_path2 = DropPath(config.drop_path_rate) if config.drop_path_rate > 0 else nn.Identity()
        
        # Modulation
        self.adaLN = AdaLNZero(config.d_model)

    def forward(self, x, c, modality_mask, rope_func, positions, cu_seqlens, max_seqlen, doc_ids=None, causal=False,
                kv_cache: Optional['KVCache'] = None, layer_idx: int = 0, block_mask=None):
        """
        Forward pass optimized for torch.compile (No boolean indexing).
        """
        # Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c)
        
        dtype = x.dtype
        # Cast modulation params
        shift_msa, scale_msa, gate_msa = shift_msa.to(dtype), scale_msa.to(dtype), gate_msa.to(dtype)
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.to(dtype), scale_mlp.to(dtype), gate_mlp.to(dtype)
        
        # Expand mask for broadcasting
        mask = modality_mask.unsqueeze(-1).to(dtype)
        inv_mask = 1.0 - mask
        
        # --- NEW: AdaLN Modality-Aware Bypass ---
        c_shift_msa = shift_msa * mask
        c_scale_msa = scale_msa * mask
        c_gate_msa  = (gate_msa * mask) + inv_mask 
        
        c_shift_mlp = shift_mlp * mask
        c_scale_mlp = scale_mlp * mask
        c_gate_mlp  = (gate_mlp * mask) + inv_mask
        # ----------------------------------------

        # --- Modality-Aware Pre-Norm for Attention ---
        residual = x
        
        # Apply separate norms then blend
        x_norm_text = self.norm1_text(x)
        x_norm_img = self.norm1_img(x)
        x_norm = (x_norm_text * inv_mask) + (x_norm_img * mask)
        
        x_norm = x_norm * (1 + c_scale_msa) + c_shift_msa
        
        attn_out = self.attn(
            x_norm, rope_func, positions, cu_seqlens, max_seqlen,
            doc_ids=doc_ids, causal=causal, mod_mask=modality_mask,
            kv_cache=kv_cache, layer_idx=layer_idx, block_mask=block_mask
        )
        
        x = residual + self.drop_path1(c_gate_msa * self.ls1(attn_out).to(dtype))
        
        # --- Sandwich Norm ---
        x = self.sandwich_norm(x)
        
        # --- Separate MLPs (Dense Masked Implementation) ---
        residual = x
        
        # 1. Text Path (Computed for ALL tokens)
        x_txt_in = self.norm2_text(x)
        x_txt_in = x_txt_in * (1 + c_scale_mlp) + c_shift_mlp 
        out_txt = self.mlp_text(x_txt_in)
        
        # 2. Image Path (Computed for ALL tokens)
        x_img_in = self.norm2_img(x)
        x_img_in = x_img_in * (1 + c_scale_mlp) + c_shift_mlp
        out_img = self.mlp_img(x_img_in)
        
        # 3. Blend based on mask
        mlp_out = (out_txt * inv_mask) + (out_img * mask)
        
        x = residual + self.drop_path2(c_gate_mlp * self.ls2(mlp_out))
        
        return x
# -----------------------------------------------------------------------------
# 6. Main Model: OmniFusionV2
# -----------------------------------------------------------------------------

class OmniFusionV2(nn.Module):
    def __init__(self, config: OmniConfigV2):
        super().__init__()
        self.config = config
        
        # 1. Initialize Layers (Define the architecture)
        # Use Conv2d for Patch Embedding to fix blocky artifacts
        self.patch_embed = nn.Conv2d(config.in_channels, config.d_model, 
                                     kernel_size=config.patch_size, 
                                     stride=config.patch_size, bias=False)
        
        self.text_embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=100258)
        self.time_embed = TimestepEmbedder(config.d_model, freq_embedding_size=256)

        # Optional learned pooling for global text conditioning.
        # Only instantiated when enabled to keep backward-compatibility with older checkpoints.
        self.text_attn_pool = TextAttentionPool(config.d_model) if config.text_pooling == "attn" else None
         
        # [CRITICAL ARCHITECTURE FIX] 
        # Added LayerNorm to equalize Text signal variance with Timestep signal variance.
        # Prevents AdaLN from becoming text-blind due to embedding initialization scaling.
        self.text_pool_proj = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model, bias=False),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model, bias=False),
        )
        
        # Backbone
        self.rope = AxialRoPE3D(config.head_dim, config.rope_base)
        
        # Regional Compilation
        blocks =[]
        for _ in range(config.n_layers):
            block = MMDiTBlock(config)
            if config.regional_compile:
                try:
                    block = torch.compile(block, mode="reduce-overhead")
                except Exception as e:
                    logger.warning(f"Regional Compilation failed: {e}")
                    block = MMDiTBlock(config) 
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        
        self.final_norm = OmniRMSNorm(config.d_model, config.rms_norm_eps)
        
        # Heads
        self.text_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.image_head = nn.Linear(config.d_model, config.in_channels * config.patch_size**2, bias=True)
        
        # Init weights
        self.initialize_weights()
        
        # Tied Embeddings
        if config.tie_word_embeddings:
            self.text_head.weight = self.text_embed.weight
                    
    def initialize_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Conv2d):
             torch.nn.init.xavier_uniform_(m.weight)
             if m.bias is not None:
                 nn.init.zeros_(m.bias)
        
        self.apply(_init)

        # self.apply(_init) overwrites nn.Embedding's zeroed padding row; restore it.
        self.zero_padding_embedding()

        # [FIX] Restored strict ZERO-initialization for image_head.
        # "Dead gradients" at step 0 were actually normal and desirable for DiT/Flow Matching. 
        # The previous random init was only "helping" because it encouraged the model 
        # to exploit the linear short-circuit bypass.
        nn.init.zeros_(self.image_head.weight)
        if self.image_head.bias is not None:
            nn.init.zeros_(self.image_head.bias)
        
        # [FIX] Xavier init for text_pool_proj so text conditioning starts at
        # the SAME magnitude as timestep embeddings (~1.0). Previous std=0.002
        # made text signal 500x weaker than t_emb, causing "text-blindness".
        for m in self.text_pool_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # If attention pooling is enabled, keep its score head zero-initialized so it starts as uniform pooling.
        if getattr(self, "text_attn_pool", None) is not None:
            nn.init.zeros_(self.text_attn_pool.score.weight)
         
        # [FIX] Re-apply AdaLN gate initialization AFTER self.apply(_init).
        # self.apply(_init) above applies Xavier to ALL nn.Linear modules,
        # which silently overwrites AdaLNZero.__init__'s std=0.02 normal init.
        # We must re-apply to ensure gates start small-but-nonzero.
        for block in self.blocks:
            if hasattr(block, 'adaLN'):
                nn.init.normal_(block.adaLN.proj_up.weight, std=0.02)
                nn.init.zeros_(block.adaLN.proj_up.bias)
                
                # [FIX] Force attention and MLP gates open at step 0
                # Shift(0), Scale(1), Gate_MSA(2), Shift(3), Scale(4), Gate_MLP(5)
                d = self.config.d_model
                with torch.no_grad():
                    block.adaLN.proj_up.bias[2*d : 3*d].fill_(0.1) # Open Gate_MSA
                    block.adaLN.proj_up.bias[5*d : 6*d].fill_(0.1) # Open Gate_MLP

    def set_allow_cross_attention(self, allow: bool):
        """
        Set whether to allow cross-document attention in packed sequences.
        
        Args:
            allow: If True, packed documents can attend to each other (for pretraining).
                   If False (default), strictly enforces document isolation (for SFT).
        """
        for block in self.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, '_allow_cross_attention'):
                block.attn._allow_cross_attention = allow
            elif hasattr(block, 'attn'):
                block.attn._allow_cross_attention = allow

    def zero_padding_embedding(self):
        """Keep PAD inert even when loading legacy checkpoints with a learned pad row."""
        with torch.no_grad():
            padding_idx = getattr(self.text_embed, "padding_idx", None)
            if padding_idx is not None:
                self.text_embed.weight[padding_idx].zero_()

    def _compute_pooled_text_cond(self, txt_ids: torch.Tensor, txt_emb: torch.Tensor, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """
        Compute the pooled text conditioning vector used for image-token AdaLN conditioning.

        Returns:
            text_cond: [1, D] in `dtype`, already scaled by config.pooled_text_cond_scale
                       and optionally dropped (returns None when disabled/dropped).
        """
        scale = float(getattr(self.config, "pooled_text_cond_scale", 1.0))
        drop_p = float(getattr(self.config, "pooled_text_drop_prob", 0.0))

        # Optional dropout during training to force reliance on token-level conditioning.
        if self.training and drop_p > 0.0:
            if torch.rand((), device=txt_emb.device) < drop_p:
                scale = 0.0

        if scale == 0.0:
            return None

        pad_token_id = 100258
        non_pad = (txt_ids != pad_token_id)

        # Pool token embeddings into a single [1,D] vector.
        if getattr(self.config, "text_pooling", "mean") == "attn" and getattr(self, "text_attn_pool", None) is not None:
            pooled = self.text_attn_pool(txt_emb, non_pad.unsqueeze(-1))  # [1,D]
        else:
            non_pad_mask = non_pad.to(dtype).unsqueeze(-1)  # [L,1]
            denom = non_pad_mask.sum().clamp(min=1.0)
            pooled = (txt_emb * non_pad_mask).sum(dim=0, keepdim=True) / denom  # [1,D]

        text_cond = self.text_pool_proj(pooled).to(dtype)  # [1,D]
        if scale != 1.0:
            text_cond = text_cond * scale
        return text_cond

    def pack_inputs(self, 
                    text_ids: List[torch.Tensor], 
                    images: Optional[Union[List[torch.Tensor], List[List[torch.Tensor]]]] = None, 
                    timesteps: Optional[torch.Tensor] = None,
                    text_pos_offset: Optional[List[int]] = None,
                    pad: bool = True,
                    image_positions: Optional[List[List[int]]] = None):
        """
        Native Resolution Packing Logic with Multi-Image Support.
        
        Args:
            text_ids: List of token ID tensors, one per sample
            images: Either:
                - Old format: List[Tensor] - one image [C,H,W] or [C,T,H,W] per sample
                - New format: List[List[Tensor]] - multiple images per sample
            timesteps: Flow matching timesteps
            text_pos_offset: Optional position offsets for each sample
            pad: Whether to pad to block size
            image_positions: (New) List[List[int]] - positions where each image should be inserted
                             If None, uses old behavior (image appended after text)
        
        Returns:
            packed_x, packed_c, packed_pos, modality_mask, cu_seqlens, doc_ids, image_shapes
        """
        # Auto-detect device/dtype
        if len(text_ids) > 0:
            device = text_ids[0].device
        elif images and len(images) > 0:
            # Handle both old and new format
            first_img = images[0]
            if isinstance(first_img, list) and len(first_img) > 0:
                device = first_img[0].device if first_img[0] is not None else self.config.device
            elif isinstance(first_img, torch.Tensor):
                device = first_img.device
            else:
                device = self.config.device
        else:
            device = self.config.device
            
        dtype = self.patch_embed.weight.dtype
        
        packed_x_list = []
        packed_c_list = []
        packed_pos_list = []
        modality_mask_list = []
        doc_ids_list = []
        cu_seqlens = [0]
        image_shapes = []  # List to store grid sizes (h, w) for the Conv Head
        
        B = len(text_ids)
        if images is None: 
            images = [None] * B
        
        # Detect format: old (List[Tensor]) vs new (List[List[Tensor]])
        is_multi_image_format = (
            image_positions is not None or 
            (images and len(images) > 0 and isinstance(images[0], list))
        )
        
        # Normalize to new format
        if not is_multi_image_format:
            # Convert old format to new format for unified processing
            images = [[img] if img is not None else [] for img in images]
            # Old format: images go after all text (no specific positions)
            image_positions = [None] * B
        else:
            # Ensure image_positions matches batch size
            if image_positions is None:
                image_positions = [None] * B
        
        # [FIX] Broadcast timesteps if single value for whole batch
        if timesteps is None:
            timesteps = torch.zeros(B, device=device, dtype=dtype)
        elif timesteps.numel() == 1:
            timesteps = timesteps.expand(B)
        
        p = self.config.patch_size
        
        for i, (txt, img_list, t, img_pos) in enumerate(zip(text_ids, images, timesteps, image_positions)):
            # 1. Embed Time
            t_emb = self.time_embed(t.view(1)).to(dtype)
            
            # 2. Process Text
            if txt.ndim == 0: txt = txt.unsqueeze(0)
            L_txt = txt.shape[0]
            txt_emb = self.text_embed(txt).to(dtype)
            
            offset = text_pos_offset[i] if text_pos_offset else 0
            
            sample_parts_x = []
            sample_parts_pos = []
            sample_parts_mask = []
            sample_parts_c = []
            sample_image_shapes = []
            
            # Track running temporal position
            temporal_pos = offset
            
            # Handle multi-image format with position-based insertion
            if img_pos is not None and len(img_list) > 0:
                # Sort positions to process in order
                sorted_pairs = sorted(zip(img_pos, img_list), key=lambda x: x[0])
                
                # Global pooled text conditioning (optional, scale/drop configurable).
                text_cond_mi = self._compute_pooled_text_cond(txt, txt_emb, dtype)
                
                current_text_idx = 0
                for pos, img in sorted_pairs:
                    # Add text tokens BEFORE this image position
                    if pos > current_text_idx:
                        text_chunk = txt_emb[current_text_idx:pos]
                        chunk_len = text_chunk.shape[0]
                        
                        txt_indices = torch.arange(chunk_len, device=device) + temporal_pos
                        txt_pos_3d = torch.stack([
                            txt_indices,
                            torch.zeros_like(txt_indices),
                            torch.zeros_like(txt_indices)
                        ], dim=-1)
                        txt_mask = torch.zeros(chunk_len, device=device).to(dtype)
                        
                        sample_parts_x.append(text_chunk)
                        sample_parts_pos.append(txt_pos_3d)
                        sample_parts_mask.append(txt_mask)
                        sample_parts_c.append(t_emb.repeat(chunk_len, 1))
                        
                        temporal_pos += chunk_len
                    
                    # Process the image at this position
                    if img is not None:
                        img_feat, img_pos_3d, img_mask, img_c, grid_h, grid_w = self._process_single_image(
                            img, t, temporal_pos, device, dtype, p
                        )
                        sample_parts_x.append(img_feat)
                        sample_parts_pos.append(img_pos_3d)
                        sample_parts_mask.append(img_mask)
                        # Inject pooled text into image conditioning (if enabled)
                        if text_cond_mi is not None:
                            sample_parts_c.append(img_c + text_cond_mi.expand_as(img_c))
                        else:
                            sample_parts_c.append(img_c)
                        sample_image_shapes.append((grid_h, grid_w))
                        
                        temporal_pos += 1  # Each image occupies 1 temporal slot
                    
                    # Skip the IMAGE_TOKEN placeholder in text
                    current_text_idx = pos + 1
                
                # Add remaining text after last image
                if current_text_idx < L_txt:
                    text_chunk = txt_emb[current_text_idx:]
                    chunk_len = text_chunk.shape[0]
                    
                    txt_indices = torch.arange(chunk_len, device=device) + temporal_pos
                    txt_pos_3d = torch.stack([
                        txt_indices,
                        torch.zeros_like(txt_indices),
                        torch.zeros_like(txt_indices)
                    ], dim=-1)
                    txt_mask = torch.zeros(chunk_len, device=device).to(dtype)
                    
                    sample_parts_x.append(text_chunk)
                    sample_parts_pos.append(txt_pos_3d)
                    sample_parts_mask.append(txt_mask)
                    sample_parts_c.append(t_emb.repeat(chunk_len, 1))
                    
            else:
                # Old behavior: all text first, then all images appended
                txt_indices = torch.arange(L_txt, device=device) + offset
                txt_pos = torch.stack([
                    txt_indices,
                    torch.zeros_like(txt_indices),
                    torch.zeros_like(txt_indices)
                ], dim=-1)
                txt_mask = torch.zeros(L_txt, device=device).to(dtype)
                
                sample_parts_x.append(txt_emb)
                sample_parts_pos.append(txt_pos)
                sample_parts_mask.append(txt_mask)
                sample_parts_c.append(t_emb.repeat(L_txt, 1))
                
                temporal_pos = offset + L_txt
                
                # Global pooled text conditioning (optional, scale/drop configurable).
                text_cond = self._compute_pooled_text_cond(txt, txt_emb, dtype)
                
                # Process images (old format: appended after text)
                for img in img_list:
                    if img is not None:
                        img_feat, img_pos_3d, img_mask, img_c, grid_h, grid_w = self._process_single_image(
                            img, t, temporal_pos, device, dtype, p
                        )
                        sample_parts_x.append(img_feat)
                        sample_parts_pos.append(img_pos_3d)
                        sample_parts_mask.append(img_mask)
                        # Inject pooled text into image conditioning (if enabled)
                        if text_cond is not None:
                            sample_parts_c.append(img_c + text_cond.expand_as(img_c))
                        else:
                            sample_parts_c.append(img_c)
                        sample_image_shapes.append((grid_h, grid_w))
                        
                        temporal_pos += 1
            
            # Store image shapes for this sample
            if sample_image_shapes:
                image_shapes.extend(sample_image_shapes)
            else:
                image_shapes.append(None)
                
            # 4. Concat Sample
            if sample_parts_x:
                sample_x = torch.cat(sample_parts_x, dim=0)
                sample_c = torch.cat(sample_parts_c, dim=0)
                sample_pos = torch.cat(sample_parts_pos, dim=0)
                sample_mask = torch.cat(sample_parts_mask, dim=0)
            else:
                # Empty sample fallback
                sample_x = torch.zeros(1, self.config.d_model, device=device, dtype=dtype)
                sample_c = t_emb
                sample_pos = torch.zeros(1, 3, device=device, dtype=torch.long)
                sample_mask = torch.zeros(1, device=device, dtype=dtype)
            
            # [FIX] REMOVED the linear short-circuit here.
            # Previously, `sample_c` was added directly to `sample_x`, creating a massive
            # gradient bypass around the transformer blocks. The model learned to act as a 
            # linear autoencoder without ever opening the AdaLN gates, 
            # completely ignoring the cross-attention text tokens!
            
            sample_len = sample_x.shape[0]
            doc_ids_list.append(torch.full((sample_len,), i, device=device, dtype=torch.int32))
            
            packed_x_list.append(sample_x)
            packed_c_list.append(sample_c)
            packed_pos_list.append(sample_pos)
            modality_mask_list.append(sample_mask)
            
            cu_seqlens.append(cu_seqlens[-1] + sample_len)
            
        # 5. Pack Batch
        packed_x = torch.cat(packed_x_list, dim=0)
        packed_c = torch.cat(packed_c_list, dim=0)
        packed_pos = torch.cat(packed_pos_list, dim=0)
        modality_mask = torch.cat(modality_mask_list, dim=0)
        doc_ids = torch.cat(doc_ids_list, dim=0)
        
        # Padding logic (unchanged)
        total_len = packed_x.shape[0]
        pad_block_size = 512
        target_len = ((total_len + pad_block_size - 1) // pad_block_size) * pad_block_size
        padding_len = target_len - total_len
        
        if pad and padding_len > 0:
            pad_x = torch.zeros(padding_len, packed_x.shape[1], device=device, dtype=dtype)
            pad_c = torch.zeros(padding_len, packed_c.shape[1], device=device, dtype=dtype)
            pad_pos = torch.zeros(padding_len, packed_pos.shape[1], device=device, dtype=packed_pos.dtype)
            pad_mask = torch.zeros(padding_len, device=device, dtype=modality_mask.dtype)
            pad_doc = torch.full((padding_len,), -1, device=device, dtype=torch.int32)
            
            packed_x = torch.cat([packed_x, pad_x], dim=0)
            packed_c = torch.cat([packed_c, pad_c], dim=0)
            packed_pos = torch.cat([packed_pos, pad_pos], dim=0)
            modality_mask = torch.cat([modality_mask, pad_mask], dim=0)
            doc_ids = torch.cat([doc_ids, pad_doc], dim=0)
            # [FIX] Do NOT add padding to cu_seqlens - it's not a real sequence!
            # FlashAttention uses cu_seqlens to define batch boundaries.
            # Adding padding length creates a phantom sequence causing attention to garbage data.
            
        cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)
        
        return packed_x, packed_c, packed_pos, modality_mask, cu_seqlens, doc_ids, image_shapes

    def _process_single_image(self, img: torch.Tensor, t: torch.Tensor, 
                               temporal_pos: int, device, dtype, p: int):
        """
        Process a single image through patch embedding.
        
        Args:
            img: Image tensor [C, H, W] or [C, T, H, W]
            t: Timestep tensor
            temporal_pos: Starting temporal position for 3D RoPE
            device: Target device
            dtype: Target dtype
            p: Patch size
            
        Returns:
            img_feat, img_pos, img_mask, img_c, grid_h, grid_w
        """
        if img.dim() == 3:
            img = img.unsqueeze(1)  # [C, 1, H, W]
        
        C, T, H, W = img.shape
        
        # Timestep Shifting for Resolution
        num_patches = (H // p) * (W // p)
        alpha = math.sqrt(num_patches / ((256 // p) ** 2)) if num_patches > 0 else 1.0
        t_shifted = (alpha * t) / (1 + (alpha - 1) * t)
        t_emb_img = self.time_embed(t_shifted.view(1)).to(dtype)
        
        all_feats = []
        all_pos = []
        all_mask = []
        all_c = []
        grid_h, grid_w = 0, 0
        
        # Iterate Frames (T=1 for Images)
        for frame_idx in range(T):
            frame = img[:, frame_idx, :, :]  # [C, H, W]
            
            # === Conv2d Patch Embedding ===
            img_feat_map = self.patch_embed(frame.unsqueeze(0).to(dtype))
            
            grid_h, grid_w = img_feat_map.shape[2], img_feat_map.shape[3]
            
            # Flatten: [1, D, Grid_H, Grid_W] -> [L, D]
            img_feat = img_feat_map.flatten(2).transpose(1, 2).squeeze(0)
            
            # Positional Embeddings
            y_coords = torch.arange(grid_h, device=device).repeat_interleave(grid_w)
            x_coords = torch.arange(grid_w, device=device).repeat(grid_h)
            
            t_coords = torch.full_like(y_coords, temporal_pos + frame_idx)
            
            img_pos = torch.stack([t_coords, y_coords, x_coords], dim=-1)
            img_mask = torch.ones(grid_h * grid_w, device=device).to(dtype)
            
            all_feats.append(img_feat)
            all_pos.append(img_pos)
            all_mask.append(img_mask)
            all_c.append(t_emb_img.repeat(grid_h * grid_w, 1))
        
        # Concatenate all frames
        return (
            torch.cat(all_feats, dim=0),
            torch.cat(all_pos, dim=0),
            torch.cat(all_mask, dim=0),
            torch.cat(all_c, dim=0),
            grid_h,
            grid_w
        )

    def forward(self, 
                text_ids: List[torch.Tensor], 
                images: Optional[Union[List[torch.Tensor], List[List[torch.Tensor]]]] = None, 
                timesteps: Optional[torch.Tensor] = None,
                causal_text: bool = False,
                kv_cache: Optional['KVCache'] = None,
                text_pos_offset: Optional[List[int]] = None,
                image_positions: Optional[List[List[int]]] = None):
        """
        Forward pass with packing.
        
        Args:
            text_ids: List of token ID tensors
            images: List[Tensor] or List[List[Tensor]] for multi-image
            timesteps: Flow matching timesteps
            causal_text: Use causal attention for text
            kv_cache: Optional KV cache for generation
            text_pos_offset: Position offsets
            image_positions: (Multi-image) List[List[int]] of insertion positions
        """
        pad = (kv_cache is None)
        x, c, pos, mod_mask, cu_seqlens, doc_ids, image_shapes = self.pack_inputs(
            text_ids, images, timesteps, text_pos_offset, pad=pad, image_positions=image_positions
        )
    
        # Calculate max_seqlen as a tensor to avoid graph breaks
        # torch.compile / Inductor will handle this better if it stays a tensor
        max_seqlen_tensor = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        max_seqlen = max_seqlen_tensor

        # Precompute FlexAttention BlockMask once per forward pass.
        # PackedSelfAttention previously created this mask once per layer (and again during
        # gradient checkpoint recomputation). Hoisting it here avoids redundant work.
        block_mask = None
        if FLEX_ATTENTION_AVAILABLE and doc_ids is not None and x.is_cuda and kv_cache is None:
            try:
                total_tokens = x.shape[0]
                t_pos = pos[:, 0]

                def hybrid_block_mask(b, h, q_idx, kv_idx):
                    same_doc = doc_ids[q_idx] == doc_ids[kv_idx]
                    valid_docs = (doc_ids[q_idx] >= 0) & (doc_ids[kv_idx] >= 0)
                    time_causal = t_pos[q_idx] >= t_pos[kv_idx]
                    valid_attn = time_causal if causal_text else True
                    return same_doc & valid_docs & valid_attn

                block_mask = create_block_mask(
                    hybrid_block_mask,
                    B=1,
                    H=None,
                    Q_LEN=total_tokens,
                    KV_LEN=total_tokens,
                    device=x.device,
                )
            except Exception:
                # If mask creation fails, attention will fall back to its internal creation path.
                block_mask = None
        
        for i, block in enumerate(self.blocks):
            if self.config.grad_checkpointing and self.training:
                # [OPTIMIZATION - VRAM] Gradient Checkpointing
                # Recomputes activations during backward pass instead of storing them all in VRAM.
                # Drops VRAM by ~50% at the cost of ~15% slower forward passes.
                # We use use_reentrant=False as it's the modern PyTorch standard.
                x = torch.utils.checkpoint.checkpoint(
                    block, x, c, mod_mask, self.rope, pos, cu_seqlens, max_seqlen, doc_ids, causal_text,
                    kv_cache, i, block_mask, use_reentrant=False
                )
            else:
                x = block(x, c, mod_mask, self.rope, pos, cu_seqlens, max_seqlen, doc_ids=doc_ids, causal=causal_text,
                         kv_cache=kv_cache, layer_idx=i, block_mask=block_mask)
                
        x = self.final_norm(x)
    
        # Heads on packed tokens (no spatial reshape needed).
        # For long contexts, materializing full [T, vocab] text logits can dominate VRAM.
        if self.training and getattr(self.config, "lazy_logits", False):
            return {
                "image": None,
                "text": None,
                "x_out": x,  # hidden states for lazy head evaluation in the loss
                "modality_mask": mod_mask,
                "cu_seqlens": cu_seqlens,
            }

        return {
            "image": self.image_head(x),
            "text": self.text_head(x),
            "x_out": x,
            "modality_mask": mod_mask,
            "cu_seqlens": cu_seqlens,
        }

    @torch.no_grad()
    def generate(self, 
                 text_ids: Union[torch.Tensor, List[torch.Tensor]], 
                 height: int = 256, 
                 width: int = 256, 
                 steps: int = 20, 
                 # CFG scale of 1.0 disables classifier-free guidance (unconditional behavior).
                 # Most text-to-image usage expects guidance > 1.0, so we default to the same
                 # value used by the GUI / generate_multimodal unless the caller overrides it.
                 cfg_scale: float = 4.5,
                 solver: str = "euler"):
        """
        Flow Matching Generation Loop.
        Solvers: 'euler' (1st order), 'midpoint' (2nd order Heun).
        """
        # === FIX: Robust Input Handling (List vs Tensor 1D/2D) ===
        if isinstance(text_ids, list):
            # Handle List[Tensor]
            if len(text_ids) == 0:
                raise ValueError("Empty text_ids list provided to generate()")
            device = text_ids[0].device
            B = len(text_ids)
            text_list = text_ids

        elif isinstance(text_ids, torch.Tensor):
            device = text_ids.device
            if text_ids.dim() == 2:
                # [B, L] -> Standard batch
                B = text_ids.shape[0]
                text_list = [text_ids[i] for i in range(B)]
            elif text_ids.dim() == 1:
                # [L] -> Single sequence, treat as Batch=1
                B = 1
                text_list = [text_ids]
            else:
                raise ValueError(f"text_ids tensor must be 1D or 2D, got {text_ids.dim()}D")
        else:
            raise TypeError(f"text_ids must be Tensor or List[Tensor], got {type(text_ids)}")
        # =========================================================

        dtype = self.patch_embed.weight.dtype
        
        # 1. Init Noise (height/width are PIXEL dimensions, convert to latent)
        p = self.config.patch_size
        vae_downsample = 8  # Flux VAE spatial downsample factor
        c_vae = self.config.in_channels
        latent_h, latent_w = height // vae_downsample, width // vae_downsample
        
        # Create latent list (latent space, NOT pixel space)
        latents_list = [torch.randn(c_vae, latent_h, latent_w, device=device, dtype=dtype) for _ in range(B)]
        
        dt = 1.0 / steps
        
        def _unpatchify_velocities(out, num_samples, latents_ref):
            """Extract and unpatchify image velocity predictions from packed output."""
            v_preds = []
            cu_sq = out["cu_seqlens"]
            img_packed = out["image"]
            mask = out["modality_mask"]
            patches_h, patches_w = latent_h // p, latent_w // p
            
            for i in range(num_samples):
                start, end = cu_sq[i], cu_sq[i + 1]
                img_tokens = img_packed[start:end][mask[start:end] == 1.0]
                
                if img_tokens.numel() == 0:
                    v_preds.append(torch.zeros_like(latents_ref[i % len(latents_ref)]))
                    continue
                
                img_tokens = img_tokens.view(patches_h, patches_w, c_vae * p * p)
                img_tokens = img_tokens.permute(2, 0, 1)
                img_tokens = F.fold(img_tokens.view(1, c_vae * p * p, -1),
                                    output_size=(latent_h, latent_w),
                                    kernel_size=p, stride=p)
                v_preds.append(img_tokens.squeeze(0))
            return v_preds
        
        def get_velocity(latents_in, t_val):
            """Get velocity with optional Classifier-Free Guidance."""
            if cfg_scale <= 1.0:
                # Conditional-only (no CFG)
                t_batch = torch.full((B,), t_val, device=device, dtype=dtype)
                out = self.forward(text_list, latents_in, t_batch, causal_text=True)
                return _unpatchify_velocities(out, B, latents_in)
            else:
                # CFG: batch conditional + unconditional together
                # Create an "empty" unconditional text with the SAME LENGTH as the conditional
                # prompt. This keeps image-token RoPE temporal positions aligned between the
                # conditional/unconditional halves, which is required for correct CFG subtraction.
                uncond_list = []
                for txt in text_list:
                    pad_len = int(txt.numel())
                    if pad_len <= 0:
                        uncond_list.append(txt.new_tensor([100257]))
                        continue
                    # PAD everywhere, EOT at the end (matches typical padded empty prompt).
                    uncond = txt.new_full((pad_len,), 100258)
                    uncond[-1] = 100257
                    uncond_list.append(uncond)
                
                combined_text = text_list + uncond_list
                combined_latents = latents_in + latents_in
                t_batch = torch.full((2 * B,), t_val, device=device, dtype=dtype)
                
                out = self.forward(combined_text, combined_latents, t_batch, causal_text=True)
                v_all = _unpatchify_velocities(out, 2 * B, latents_in)
                
                # Apply CFG: v = v_uncond + cfg * (v_cond - v_uncond)
                v_guided = []
                for i in range(B):
                    v_cond = v_all[i]
                    v_uncond = v_all[i + B]
                    v_guided.append(v_uncond + cfg_scale * (v_cond - v_uncond))
                return v_guided

        for step in range(steps):
            t_curr = step * dt
            
            if solver == "euler":
                v_pred = get_velocity(latents_list, t_curr)
                for i in range(B):
                    latents_list[i] = latents_list[i] + v_pred[i] * dt
                    
            elif solver == "midpoint":
                # Heun's Method
                v1 = get_velocity(latents_list, t_curr)
                x_tmp_list = [latents_list[i] + v1[i] * dt for i in range(B)]
                
                # Clamp t_next to 1.0
                t_next = min(t_curr + dt, 1.0)
                v2 = get_velocity(x_tmp_list, t_next)
                
                for i in range(B):
                    latents_list[i] = latents_list[i] + 0.5 * (v1[i] + v2[i]) * dt
            else:
                raise ValueError(f"Unknown solver: {solver}")
                
        return latents_list
        
    @torch.no_grad()
    def generate_text(self, 
                      prompt_ids: torch.Tensor,
                      max_new_tokens: int = 100,
                      temperature: float = 1.0,
                      top_p: float = 0.9,
                      top_k: int = 50,
                      eos_token_id: int = 100257) -> torch.Tensor: # FIX 1: Correct EOT ID
        """
        Generate text autoregressively with KV cache.
        
        Args:
            prompt_ids: [L] or [1, L] input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = default, <1.0 = more deterministic)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling limit
            eos_token_id: Token ID to stop generation
            
        Returns:
            [prompt_len + generated_len] tensor of token IDs
        """
        self.eval()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        # Normalize input shape
        if prompt_ids.dim() == 2:
            prompt_ids = prompt_ids.squeeze(0)
        prompt_ids = prompt_ids.to(device)
        
        # Initialize KV cache
        n_layers = len(self.blocks)
        kv_cache = KVCache.empty(n_layers)
        # Output buffer
        generated = prompt_ids.clone()
        
        # ===== PREFILL PHASE =====
        seq_len = prompt_ids.shape[-1]
        txt_list = [prompt_ids]
        
        # FIX: Force Time = 1.0 (Clean Data)
        t_clean = torch.ones(len(txt_list), device=device, dtype=dtype)
        
        out = self.forward(txt_list, timesteps=t_clean, causal_text=True, kv_cache=kv_cache)
        
        logits = out["text"][-1:]  # Only last token for next prediction
        
        kv_cache.seq_len = seq_len
        
        # ===== GENERATION PHASE =====
        for step in range(max_new_tokens):
            # Sample next token
            if temperature > 0:
                next_token_logits = logits[-1] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits[-1:], dim=-1)
            
            # Check EOS
            print(f"DEBUG_GEN: Step {step}, sampled token: {next_token.item()}")
            if next_token.item() == eos_token_id:
                break
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=0)
            
            # ===== INCREMENTAL DECODE =====
            # Only process the new token
            new_pos = kv_cache.seq_len
            x = self.text_embed(next_token)  # [1, D]
            
            # [FIX] Text positions should be [time, 0, 0], not [time, time, time]
            # This matches the 3D RoPE encoding where text only uses temporal axis
            positions = torch.tensor([[new_pos, 0, 0]], device=device, dtype=torch.long)
            cu_seqlens = torch.tensor([0, 1], device=device, dtype=torch.int32)
            c = x.clone()
            modality_mask = torch.zeros(1, device=device)
            doc_ids = torch.zeros(1, device=device, dtype=torch.long)
            
            # Incremental forward
            # FIX: Force Time = 1.0 here too
            t_step = torch.ones(1, device=device, dtype=dtype)
            
            out = self.forward([next_token.view(-1)], timesteps=t_step, causal_text=True, kv_cache=kv_cache, text_pos_offset=[new_pos])
            logits = out["text"][-1:] # [1, Vocab]
            
            kv_cache.seq_len += 1
        
        return generated

    def _sample_next_token(self, logits, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0, generated_tokens=None):
        """
        Robust sampling with advanced controls.
        """
        # Apply Repetition Penalty
        if repetition_penalty != 1.0 and generated_tokens is not None:
             # Create a mask of counts or just penalize presence
             # Standard implementation: penalize logits of already generated tokens
             score = torch.gather(logits, 1, generated_tokens)
             
             # If score < 0 then multiply check, but here logits are raw.
             # usually: where score < 0, score * penalty; where score > 0, score / penalty 
             # But standard VLLM/HF approach:
             score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
             
             logits.scatter_(1, generated_tokens, score)

        # Apply Temperature
        if temperature > 0:
            logits = logits / max(temperature, 1e-6)
        
        # Apply Min-P (Dynamic Threshold)
        if min_p > 0.0:
            p_max = torch.softmax(logits, dim=-1).max(dim=-1).values
            thresh = p_max * min_p
            indices_to_remove = torch.softmax(logits, dim=-1) < thresh.unsqueeze(-1)
            logits[indices_to_remove] = float('-inf')

        # Apply Top-K
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v[:, -1].unsqueeze(-1)
            logits[logits < pivot] = float('-inf')

        # Apply Top-P (Nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        # Sample
        if temperature > 0:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
        return next_token

    @torch.no_grad()
    def generate_multimodal(self, 
                          prompt_ids: torch.Tensor, 
                          images: Optional[List[torch.Tensor]] = None,
                          max_new_tokens: int = 256,
                          temperature: float = 0.7,
                          top_k: int = 40,
                          top_p: float = 0.9,
                          min_p: float = 0.0,
                          repetition_penalty: float = 1.0,
                          image_token_id: int = 100260, 
                          default_height: int = 512,
                          default_width: int = 512,
                          cfg_scale: float = 4.5):
        """
        Multimodal Generation Loop with <image> tag support and Advanced Sampling.
        """
        # === ROBUSTNESS FIXES ===
        # 1. Ensure input is on the correct device
        model_device = next(self.parameters()).device
        if prompt_ids.device != model_device:
            prompt_ids = prompt_ids.to(model_device)
            
        # 2. Normalize to 1D [Seq_Len] (Handles [1, Seq_Len] inputs)
        if prompt_ids.dim() == 2:
            prompt_ids = prompt_ids.squeeze(0)
        # ========================

        device = prompt_ids.device
        
        # Prefill
        kv_cache = KVCache.empty(len(self.blocks))
        # Now prompt_ids is guaranteed 1D, so we just wrap it in a list
        txt_list = [prompt_ids] 
        
        # Initial Forward (Fill Cache)
        out = self.forward(txt_list, images=images, causal_text=True, kv_cache=kv_cache)
        
        logits = out["text"][-1:]
        seq_len = prompt_ids.shape[0]
        kv_cache.seq_len = seq_len
        
        generated_tokens = prompt_ids.clone()
        results = {"text": generated_tokens, "images": [], "stream": []}
        
        # Track predicted resolution attributes
        last_h = default_height
        last_w = default_width
        
        # Token ranges for resolution (aligned with TiktokenTokenizer in data_manager.py)
        # buckets: [64, 128, ..., 1024] (16 buckets)
        res_buckets = list(range(64, 1025, 64))
        h_start = 100261
        w_start = 100261 + len(res_buckets)
        
        for _ in range(max_new_tokens):
            # Sample with Advanced Specs
            # reshape generated_tokens to [1, L] for gather
            gen_history = generated_tokens.unsqueeze(0)
            
            # Helper handles the complex logic
            next_token = self._sample_next_token(logits, temperature, top_k, top_p, min_p, repetition_penalty, gen_history)
            
            token_val = next_token.item()
            
            # [LOGGING] Trace generation steps
            # logger.info(f"[GEN] Sampled: {token_val}")
            
            # 1. Check for <image> Trigger
            if token_val == image_token_id:
                # == IMAGE GENERATION MODE ==
                # append <image>
                generated_tokens = torch.cat([generated_tokens, next_token.view(-1)], dim=0)
                results["stream"].append({"type": "token", "val": token_val})
                
                # Apply Resolution Safety Caps
                target_h, target_w = last_h, last_w
                total_pixels = target_h * target_w
                
                if total_pixels > self.config.max_resolution_pixels:
                    # Scale down while maintaining aspect ratio
                    scale = math.sqrt(self.config.max_resolution_pixels / total_pixels)
                    target_h = int((target_h * scale) // 16) * 16 # Align to VAE/Patch
                    target_w = int((target_w * scale) // 16) * 16
                    # Ensure minimums
                    target_h = max(target_h, 64)
                    target_w = max(target_w, 64)
                
                # Generate Image
                ctx_tensor = generated_tokens
                img_gen = self.generate([ctx_tensor], height=target_h, width=target_w, steps=30, cfg_scale=cfg_scale)
                
                # Add to results
                if img_gen:
                    results["images"].append(img_gen[0]) 
                    results["stream"].append({"type": "image", "val": img_gen[0]})
                
                # Resume text: Update KV Cache with the Image Token
                new_pos = kv_cache.seq_len
                out = self.forward([next_token.view(-1)], causal_text=True, kv_cache=kv_cache, text_pos_offset=[new_pos])
                logits = out["text"][-1:]
                kv_cache.seq_len += 1
                continue
            
            # 2. Check for Resolution Tokens
            if h_start <= token_val < h_start + len(res_buckets):
                last_h = res_buckets[token_val - h_start]
            elif w_start <= token_val < w_start + len(res_buckets):
                last_w = res_buckets[token_val - w_start]
            
            # 3. Stop if EOT, im_end, or pad token (100258)
            if token_val in [100257, 100259, 100258]:
                break
                
            # 4. Append Text
            generated_tokens = torch.cat([generated_tokens, next_token.view(-1)], dim=0)
            results["text"] = generated_tokens 
            results["stream"].append({"type": "token", "val": token_val})
            
            # Forward
            new_pos = kv_cache.seq_len
            
            # [FIX] Force pad=False inside pack_inputs via forwarding kv_cache logic
            # forward() logic checks 'pad = (kv_cache is None)'
            out = self.forward([next_token.view(-1)], causal_text=True, kv_cache=kv_cache, text_pos_offset=[new_pos])
            logits = out["text"][-1:]
            kv_cache.seq_len += 1
            
        return results

# -----------------------------------------------------------------------------
# 7. Rectified Flow Loss with Configurable Sampling
# -----------------------------------------------------------------------------


class FlowMatchingLoss(nn.Module):
    def __init__(self, model, uniform_sampling: bool = False):
        super().__init__()
        self.model = model
        self.uniform_sampling = uniform_sampling
    
    def forward(self, text_ids, images):
        """
        Computes Flow Matching Loss (Images) + Cross Entropy Loss (Text).
        """
        B = len(text_ids)
        # Determine device from first non-None image or first text input
        if images is not None and any(img is not None for img in images):
            for img in images:
                if img is not None:
                    device = img.device
                    dtype = img.dtype
                    break
        else:
            device = text_ids[0].device
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

        if self.uniform_sampling:
            t = torch.rand(B, device=device, dtype=dtype)
        else:
            u = torch.normal(mean=-0.5, std=1.0, size=(B,), device=device).to(dtype)
            t = torch.sigmoid(u)
        
        # 1. Prepare Image Targets (Masked Flow Matching)
        noisy_images = []
        target_v = []
        has_images = False
        
        effective_images = images if images is not None else [None] * B
        
        for i, x1 in enumerate(effective_images):
            if x1 is not None:
                has_images = True
                x0 = torch.randn_like(x1) # Noise
                t_curr = t[i]
                xt = t_curr * x1 + (1 - t_curr) * x0
                v = x1 - x0
                noisy_images.append(xt)
                target_v.append(v)
            else:
                noisy_images.append(None)
                target_v.append(None)
            
        # 2. Model Forward
        res = self.model(text_ids, noisy_images, t, causal_text=True)
        x_out = res.get("x_out", None)
        
        # 3. Compute Image Loss (MSE)
        img_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        if has_images:
            pred_v_packed = res.get("image", None)
            mod_mask = res["modality_mask"]
            
            p = self.model.config.patch_size
            valid_targets = []
            for v_map in target_v:
                if v_map is not None:
                    # [FIX] Use explicit reshape/permute to match Conv2d memory layout
                    c, h, w = v_map.shape
                    gh, gw = h // p, w // p
                    
                    # 1. View as grid: (C, GH, P, GW, P)
                    reshaped = v_map.reshape(c, gh, p, gw, p)
                    
                    # 2. Permute to (GH, GW, C, P, P) to group spatial blocks
                    packed = reshaped.permute(1, 3, 0, 2, 4)
                    
                    # 3. Flatten to (Num_Patches, Patch_Content)
                    flat = packed.reshape(gh * gw, c * p * p)
                    valid_targets.append(flat)
            
            if valid_targets:
                # Filter to only image tokens (mod_mask == 1.0)
                is_img = (mod_mask == 1.0)

                if pred_v_packed is None and x_out is not None:
                    pred_img = self.model.image_head(x_out[is_img]) if is_img.any() else torch.empty(0, device=device, dtype=dtype)
                else:
                    pred_img = pred_v_packed[is_img]

                target_packed = torch.cat(valid_targets, dim=0).to(pred_img.dtype)
                
                if pred_img.shape[0] == target_packed.shape[0]:
                    img_loss = F.mse_loss(pred_img, target_packed)
                else:
                    min_len = min(pred_img.shape[0], target_packed.shape[0])
                    img_loss = F.mse_loss(pred_img[:min_len], target_packed[:min_len])

        # 4. Compute Text Loss (Cross Entropy) - GRAPH FRIENDLY VERSION
        text_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        cu_seqlens = res["cu_seqlens"]
        text_logits_packed = res.get("text", None)
        mod_mask = res["modality_mask"]
        
        total_text_tokens = torch.tensor(0.0, device=device, dtype=dtype)
        ce_loss_sum = torch.tensor(0.0, device=device, dtype=dtype)
        
        for i in range(B):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            sample_logits = None if text_logits_packed is None else text_logits_packed[start:end]
            sample_mask = mod_mask[start:end]
            
            # [FIX] Filter combined modalities sequence space into explicitly only text probability outputs
            is_text = (sample_mask == 0.0)

            if sample_logits is None:
                if x_out is None:
                    raise RuntimeError("lazy_logits active but model did not return x_out.")
                sample_x = x_out[start:end]
                text_only_logits = self.model.text_head(sample_x[is_text]) if is_text.any() else torch.empty(0, device=device, dtype=dtype)
            else:
                text_only_logits = sample_logits[is_text]
            
            # Targets are the inputs shifted by 1
            targets = text_ids[i].to(device)
            
            # [CRITICAL FIX] Alignment for I2T (Image-to-Text)
            # pack_inputs physically REMOVES the IMAGE_TOKEN (100293) from the sequence
            # and replaces it with image patches (which are now hidden in sample_mask == 1.0).
            # To maintain 1:1 alignment, we MUST strip the IMAGE_TOKEN from the targets
            # so the model isn't trying to predict 'Image' when it's actually predicting 'Assistant'.
            targets = targets[targets != 100293]
            
            L_txt = targets.shape[0]
            
            # Ensure we don't go out of bounds
            if text_only_logits.shape[0] >= L_txt:
                # Shift: Logits[0..L-2] predict Targets[1..L-1]
                logits = text_only_logits[:L_txt-1]
                targs = targets[1:].long()
                
                # Must not be pad token (100258)
                pad_mask = (targs != 100258).float()
                
                # Compute reduction='none' loss and multiply by mask
                ce = F.cross_entropy(logits, targs, reduction='none')
                masked_ce = ce * pad_mask
                
                ce_loss_sum = ce_loss_sum + masked_ce.sum()
                total_text_tokens = total_text_tokens + pad_mask.sum()

        if total_text_tokens > 0:
            text_loss = ce_loss_sum / total_text_tokens
        
        # 5. Dynamic Loss Balancing
        # [CORRECTED] Latents are already scaled at input (Step 1). 
        # We rely on lambda_img to balance the tasks, NOT internal loss scaling.
        
        # Weighted sum: Boost image weight to prioritize visual reconstruction
        return (img_loss * 5.0) + text_loss
# -----------------------------------------------------------------------------
# 8. EMA Helper
# -----------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        # We need to assume the model is currently holding the shadow weights (after apply_shadow)
        # but to restore, we would need a backup of the original weights.
        # For simplicity in this script, we assume apply_shadow is done only for eval/saving.
        # This implementation does NOT store backup. User must reload or accept weight swap.
        # Ideally, backup before apply.
        pass

def compute_loss(model, text_list, image_list):
    """Wrapper for external calls."""
    loss_fn = FlowMatchingLoss(model)
    return loss_fn(text_list, image_list)

# -----------------------------------------------------------------------------
# 9. CUDA Graph Helper
# -----------------------------------------------------------------------------

def capture_graph_static(model, text_ids, images, t):
    """
    Captures CUDA graph for static shapes.
    Requires input shapes to be fixed during replay.
    """
    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            model(text_ids, images, t)
    torch.cuda.current_stream().wait_stream(s)
    
    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model(text_ids, images, t)
    
    return g, out
