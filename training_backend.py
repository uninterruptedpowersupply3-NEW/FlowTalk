"""
training_backend.py (v6.0 - OmniFusion-X V2 Production Backend)
===============================================================================
Feature-Complete Training Backend for OmniFusionV2 (Flow Matching + Native Resolution)

Capabilities:
- Trains on heterogeneous data: images, raw text, captions, ChatML, XMLs, Alpaca
- Full integration with data_manager.py parsers and loaders
- Flow Matching loss for image generation
- Cross-Entropy loss for text generation
- EMA support for model weights
- Gradient checkpointing for memory efficiency
- Mixed precision training (bfloat16)
- Comprehensive checkpointing and resumption
- Progress callbacks for GUI integration
- Tensorboard logging

Author: Antigravity Assistant
Version: 6.0.0
===============================================================================
"""

import logging
import json
import math
import os
import gc
import time
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple, Union
from functools import partial
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image, ImageFile
import numpy as np

# --- Conditional Imports ---
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    bnb = None
    BNB_AVAILABLE = False

# --- Local Imports ---
from omni_model_v2 import OmniFusionV2, OmniConfigV2, FlowMatchingLoss, EMA
from vae_module import FluxVAE
from data_manager import (
    DataConfig, 
    MultimodalDataset, 
    StreamingMultimodalDataset,
    SimpleTokenizer,
    multimodal_collate_fn,
    create_dataloader
)

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
)
logger = logging.getLogger("TrainingBackend")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Device / Precision Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

# Enable TF32 for faster matmuls
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# Scale Factor (consistent with minimal_overfit_test.py)
# Scale Factor (consistent with minimal_overfit_test.py)
# Flux Latents are typically scaled by 0.3611 (inverse 1/0.3611 ~ 2.77)
# But standard implementation often relies on the VAE's internal scaling Config.
# We will use 1.0 here because we will use the raw latents or relying on `vae_scale_factor` from config.
# Ideally, we should measure the std of latents from FluxVAE. 
VAE_SCALE_FACTOR = 0.3611 # Adjusted for Unit Variance based on Diagnostics
VAE_SHIFT_FACTOR = 0.1159
# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class TrainingConfig:
    """Complete configuration for a training run with OmniFusionV2."""
    
    # --- Data Configuration ---
    data_paths: List[str] = field(default_factory=lambda: ["./Train_Img"])
    max_text_length: int = 512
    min_image_size: int = 32
    max_image_size: int = 512
    data_streaming: bool = False
    shuffle_buffer_size: int = 1000
    num_data_workers: int = 0  # 0 for Windows compatibility
    
    # --- Model Configuration ---
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    head_dim: int = 64
    patch_size: int = 8
    in_channels: int = 16
    vocab_size: int = 32000
    max_seq_length: int = 4096
    vae_scale_factor: float = 0.3611
    grad_checkpointing: bool = False  # Disabled by default (dynamic attention masks cause CheckpointError)
    
    # --- Training Hyperparameters ---
    batch_size: int = 1
    epochs: int = 100
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # --- Optimizer Settings ---
    optimizer_type: str = "adamw"  # "adamw", "adamw8bit", "sgd"
    beta1: float = 0.9
    beta2: float = 0.999
    
    # --- Loss Configuration ---
    text_loss_weight: float = 1.0
    image_loss_weight: float = 1.0
    uniform_timestep_sampling: bool = False  # True for uniform, False for logit-normal
    
    # --- EMA Configuration ---
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # --- Mixed Precision ---
    use_amp: bool = False  # GradScaler for AMP (use with caution)
    dtype: str = "bfloat16"  # "bfloat16", "float32", "float16"
    
    # --- Checkpointing ---
    save_dir: str = "./checkpoints"
    save_interval: int = 500  # Save every N steps
    max_checkpoints: int = 5
    log_interval: int = 10
    
    # --- Tensorboard ---
    use_tensorboard: bool = True
    tensorboard_dir: str = "./runs"
    
    # --- Device ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_omni_config(self) -> OmniConfigV2:
        """Convert TrainingConfig to OmniConfigV2."""
        dtype_map = {
            "bfloat16": "bfloat16",
            "float32": "float32",
            "float16": "float16"
        }
        return OmniConfigV2(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_length,
            device=self.device,
            dtype=dtype_map.get(self.dtype, "bfloat16"),
            vae_scale_factor=self.vae_scale_factor,
            grad_checkpointing=self.grad_checkpointing
        )

    def to_data_config(self) -> DataConfig:
        """Convert TrainingConfig to DataConfig."""
        return DataConfig(
            max_text_length=self.max_text_length,
            min_image_size=self.min_image_size,
            max_image_size=self.max_image_size,
            patch_size=self.patch_size,
            vocab_size=self.vocab_size,
            num_workers=self.num_data_workers
        )

    _GLOBAL_VAE = None

    def get_vae():
        """Accesses or initializes the Flux VAE safely."""
        global _GLOBAL_VAE
        if _GLOBAL_VAE is None:
            logger.info("Initializing FluxVAE for encoding/decoding...")
            # Note: Ensure FluxVAE is imported from your vae_module
            # Using DEVICE and DTYPE defined in your script's global scope
            _GLOBAL_VAE = FluxVAE(dtype=DTYPE).to(DEVICE)
        return _GLOBAL_VAE


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

class CosineWarmupScheduler:
    """Cosine Annealing with Linear Warmup."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, 
                 min_lr: float = 1e-6, max_lr: float = 1e-4):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = max(1, total_steps)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_step = 0
        
    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * (self.current_step / max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
    
    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        
    def get_last_lr(self) -> List[float]:
        return [self.get_lr()]


# =============================================================================
# Mixed Precision Manager
# =============================================================================

class AMPManager:
    """Manages automatic mixed precision training."""
    
    def __init__(self, enabled: bool = False, dtype: torch.dtype = torch.bfloat16):
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype
        self.scaler = torch.cuda.amp.GradScaler() if self.enabled else None
        
    def autocast(self):
        if self.enabled:
            return torch.cuda.amp.autocast(dtype=self.dtype)
        return nullcontext()
    
    def backward(self, loss):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
    def step(self, optimizer, model=None, max_grad_norm=1.0):
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
            if model is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if model is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


# =============================================================================
# Tokenizer Wrapper (for testing without full data_manager)
# =============================================================================

def ascii_tokenize(text: str, length: int = 32, vocab_size: int = 32000) -> torch.Tensor:
    """Simple ASCII tokenization for testing."""
    tokens = [ord(c) % vocab_size for c in text]
    if len(tokens) < length:
        tokens = tokens + [0] * (length - len(tokens))
    else:
        tokens = tokens[:length]
    return torch.tensor([tokens], device=DEVICE)


# =============================================================================
# Image Processing Utilities
# =============================================================================

def load_image_as_latents(
    image_path: str, 
    patch_size: int = 2, 
    max_size: int = 256,
    vae_scale_factor: float = 0.3611,
    vae_shift_factor: float = 0.1159
) -> Tuple[torch.Tensor, int, int]:
    raw_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = raw_img.size
    
    # Resize logic
    scale = min(max_size / orig_w, max_size / orig_h)
    if scale < 1.0:
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        raw_img = raw_img.resize((new_w, new_h), resample=Image.LANCZOS)
    
    # Align to patch size (Defines target_h and target_w locally)
    curr_w, curr_h = raw_img.size
    target_h = max(patch_size, (curr_h // patch_size) * patch_size)
    target_w = max(patch_size, (curr_w // patch_size) * patch_size)
    raw_img = raw_img.resize((target_w, target_h), resample=Image.LANCZOS)
    
    # Convert to tensor and move to device
    img_tensor = torch.tensor(np.array(raw_img)).float().permute(2, 0, 1) / 127.5 - 1.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE, dtype=DTYPE)
    
    # Encode
    vae = get_vae()
    with torch.no_grad():
        latents = vae.encode(img_tensor)
        if hasattr(latents, "latent_dist"):
            latents = latents.latent_dist.sample()

    # Flux Normalization: (x - shift) * scale
    latents = (latents - vae_shift_factor) * vae_scale_factor
    
    return latents, target_h // patch_size, target_w // patch_size

# =============================================================================
# Collate Function (Extended for Training)
# =============================================================================

def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for multimodal batches.
    
    Returns:
        Dict with:
        - text_ids: List[Tensor] of token IDs
        - images: List[Tensor] or None for each sample
        - has_image: List[bool]
        - texts: List[str]
    """
    text_ids = []
    images = []
    has_images = []
    texts = []
    
    for sample in batch:
        # Handle input_ids
        ids = sample.get("input_ids")
        if ids is None:
            ids = torch.zeros(32, dtype=torch.long)
        elif not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        text_ids.append(ids)
        texts.append(sample.get("text", ""))
        
        # Handle images
        if sample.get("has_image", False) and "image" in sample:
            img = sample["image"]
            if img is not None:
                if not isinstance(img, torch.Tensor):
                    img = torch.tensor(img)
                images.append(img)
                has_images.append(True)
            else:
                images.append(None)
                has_images.append(False)
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
# Trainer Class
# =============================================================================

class Trainer:
    """
    Full-featured trainer for OmniFusionV2.
    
    Features:
    - Flow Matching for images
    - Cross-Entropy for text
    - EMA
    - Mixed precision
    - Gradient accumulation
    - Checkpointing
    - Tensorboard logging
    """
    
    def __init__(
        self,
        model: OmniFusionV2,
        config: TrainingConfig,
        optimizer: torch.optim.Optimizer = None,
        dataloader: DataLoader = None,
        progress_callback: Optional[Callable[[Dict], None]] = None
    ):
        self.model = model
        self.config = config
        self.progress_callback = progress_callback
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Dtype conversion
        dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
        self.dtype = dtype_map.get(config.dtype, torch.bfloat16)
        if self.dtype == torch.bfloat16 and hasattr(self.model, 'bfloat16'):
            self.model = self.model.bfloat16()
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
            
        # Setup dataloader
        self.dataloader = dataloader
        
        # Loss functions
        self.flow_loss_fn = FlowMatchingLoss(model, uniform_sampling=config.uniform_timestep_sampling)
        self.text_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Scheduler (will be initialized in train())
        self.scheduler = None
        
        # EMA
        self.ema = EMA(model, decay=config.ema_decay) if config.use_ema else None
        
        # AMP
        self.amp_manager = AMPManager(enabled=config.use_amp, dtype=self.dtype)
        
        # Tensorboard
        self.writer = None
        if config.use_tensorboard and HAS_TENSORBOARD:
            os.makedirs(config.tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=config.tensorboard_dir)
        
        # Create checkpoint directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        logger.info(f"Trainer initialized on {self.device} with dtype {self.dtype}")
        logger.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Creates the optimizer based on config."""
        config = self.config
        
        if config.optimizer_type == "adamw8bit" and BNB_AVAILABLE:
            optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
            logger.info("Using 8-bit AdamW optimizer")
        elif config.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9
            )
            logger.info("Using SGD optimizer")
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
            logger.info("Using AdamW optimizer")
            
        return optimizer
    
    def train_epoch(self, epoch: int) -> float:
        """
        Trains for one epoch.
        
        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        if self.dataloader is None:
            logger.warning("No dataloader provided!")
            return 0.0
        
        for batch_idx, batch in enumerate(self.dataloader):
            if batch is None or len(batch.get("text_ids", [])) == 0:
                continue
                
            loss = self._training_step(batch)
            
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Update weights
                self.amp_manager.step(self.optimizer, self.model, self.config.max_grad_norm)
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                if self.ema is not None:
                    self.ema.update(self.model)
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    avg_loss = total_loss / max(1, num_batches)
                    logger.info(f"Epoch {epoch} | Step {self.global_step} | Loss: {avg_loss:.6f} | LR: {lr:.2e}")
                    
                    if self.writer is not None:
                        self.writer.add_scalar("Loss/train", avg_loss, self.global_step)
                        self.writer.add_scalar("LR", lr, self.global_step)
                
                # Progress callback
                if self.progress_callback is not None:
                    self.progress_callback({
                        "epoch": epoch,
                        "step": self.global_step,
                        "loss": loss.item() if loss is not None else 0.0,
                        "lr": self.optimizer.param_groups[0]['lr']
                    })
                    
                # Checkpointing
                if self.global_step > 0 and self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(epoch)
        
        avg_loss = total_loss / max(1, num_batches)
        return avg_loss
    
    def _training_step(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Executes a single training step.
        
        Returns:
            Loss tensor (scalar).
        """
        text_ids_list = batch.get("text_ids", [])
        images_list = batch.get("images", [])
        has_image_list = batch.get("has_image", [])
        
        if not text_ids_list:
            return None
            
        # Move text and images to device
        text_ids_device = []
        for ids in text_ids_list:
            if ids is not None:
                t = ids.to(self.device) if isinstance(ids, torch.Tensor) else torch.tensor(ids, device=self.device)
                text_ids_device.append(t)
            else:
                text_ids_device.append(torch.zeros(32, dtype=torch.long, device=self.device))
                
        device_images = []
        for i, img in enumerate(images_list):
            if img is not None:
                img_tensor = img.to(device=self.device, dtype=self.dtype) if isinstance(img, torch.Tensor) else torch.tensor(img, device=self.device, dtype=self.dtype)
                if img_tensor.dim() == 4:
                    img_tensor = img_tensor.squeeze(0)
                device_images.append(img_tensor)
            else:
                device_images.append(None)
            
        with self.amp_manager.autocast():
            # The unified flow_loss_fn now handles both Image (Flux FM) and Text (Causal AR)
            # It also implements assistant-only masking automatically.
            loss = self.flow_loss_fn(text_ids_device, device_images)
            total_loss = loss
            
        # Backward pass
        if total_loss.requires_grad:
            scaled_loss = total_loss / self.config.gradient_accumulation_steps
            self.amp_manager.backward(scaled_loss)
        
        return total_loss
    
    def train(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Full training loop.
        
        Returns:
            Training history dict.
        """
        if epochs is None:
            epochs = self.config.epochs
            
        # Initialize scheduler
        if self.dataloader is not None:
            total_steps = len(self.dataloader) * epochs // self.config.gradient_accumulation_steps
        else:
            total_steps = 10000
            
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
            min_lr=self.config.min_learning_rate,
            max_lr=self.config.learning_rate
        )
        
        history = {"train_loss": [], "lr": []}
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Total steps: ~{total_steps}")
        
        for epoch in range(epochs):
            self.epoch = epoch
            epoch_loss = self.train_epoch(epoch)
            history["train_loss"].append(epoch_loss)
            history["lr"].append(self.optimizer.param_groups[0]['lr'])
            
            logger.info(f"Epoch {epoch} completed. Average Loss: {epoch_loss:.6f}")
            
            # Track best loss
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_checkpoint(epoch, is_best=True)
        
        # Final save
        self.save_checkpoint(epochs - 1, is_final=True)
        
        if self.writer is not None:
            self.writer.close()
            
        logger.info("Training completed!")
        return history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False):
        """Saves a training checkpoint."""
        if is_best:
            filename = "best_model.pt"
        elif is_final:
            filename = "final_model.pt"
        else:
            filename = f"checkpoint_step_{self.global_step}.pt"
            
        path = os.path.join(self.config.save_dir, filename)
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "best_loss": self.best_loss,
        }
        
        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
        # Manage old checkpoints
        if not is_best and not is_final and self.config.max_checkpoints > 0:
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Removes old checkpoints keeping only the most recent."""
        import glob
        pattern = os.path.join(self.config.save_dir, "checkpoint_step_*.pt")
        checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime)
        
        while len(checkpoints) > self.config.max_checkpoints:
            old = checkpoints.pop(0)
            try:
                os.remove(old)
                logger.info(f"Removed old checkpoint: {old}")
            except:
                pass
    
    def load_checkpoint(self, path: str):
        """Loads a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        
        if self.ema is not None and "ema_shadow" in checkpoint:
            self.ema.shadow = checkpoint["ema_shadow"]
            
        logger.info(f"Loaded checkpoint from {path} (Epoch {self.epoch}, Step {self.global_step})")


# =============================================================================
# Factory Functions
# =============================================================================

def create_model(config: TrainingConfig) -> OmniFusionV2:
    """Creates an OmniFusionV2 model from TrainingConfig."""
    omni_config = config.to_omni_config()
    model = OmniFusionV2(omni_config)
    return model


def create_trainer(
    model: OmniFusionV2,
    config: TrainingConfig,
    data_paths: Optional[List[str]] = None,
    progress_callback: Optional[Callable] = None
) -> Trainer:
    """
    Creates a complete Trainer with model and dataloader.
    
    Args:
        model: OmniFusionV2 model instance
        config: TrainingConfig
        data_paths: Override data paths from config
        progress_callback: Optional callback for progress updates
        
    Returns:
        Configured Trainer instance
    """
    if data_paths is None:
        data_paths = config.data_paths
        
    data_config = config.to_data_config()
    
    # Create dataloader
    if config.data_streaming:
        dataset = StreamingMultimodalDataset(
            data_paths=data_paths,
            config=data_config,
            shuffle_buffer=config.shuffle_buffer_size
        )
    else:
        dataset = MultimodalDataset(
            data_paths=data_paths,
            config=data_config
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=not config.data_streaming,
        collate_fn=multimodal_collate_fn,
        num_workers=config.num_data_workers,
        pin_memory=True if config.device == "cuda" else False
    )
    
    trainer = Trainer(
        model=model,
        config=config,
        dataloader=dataloader,
        progress_callback=progress_callback
    )
    
    return trainer


def train_model(
    data_paths: List[str],
    config: Optional[TrainingConfig] = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[OmniFusionV2, Dict[str, Any]]:
    """
    High-level function to train a model from scratch.
    
    Args:
        data_paths: List of paths to training data
        config: TrainingConfig (uses defaults if None)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (trained model, training history)
    """
    if config is None:
        config = TrainingConfig()
    config.data_paths = data_paths
    
    model = create_model(config)
    trainer = create_trainer(model, config, data_paths, progress_callback)
    
    history = trainer.train()
    
    return model, history


# =============================================================================
# SSIM Computation
# =============================================================================

def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: [B, C, H, W] tensor
        img2: [B, C, H, W] tensor
        window_size: Size of the Gaussian window
        size_average: If True, returns mean SSIM across all samples
        
    Returns:
        SSIM score (scalar if size_average, else per-sample)
    """
    # Ensure proper shape
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    # Match shapes
    min_h = min(img1.shape[2], img2.shape[2])
    min_w = min(img1.shape[3], img2.shape[3])
    img1 = img1[:, :, :min_h, :min_w]
    img2 = img2[:, :, :min_h, :min_w]
    
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(img1.size(1), 1, window_size, window_size).to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# =============================================================================
# Image Generation (Euler Sampler for Flow Matching)
# =============================================================================

def generate_image_euler(
    model: OmniFusionV2,
    prompt_ids: torch.Tensor,
    target_h: int,
    target_w: int,
    num_steps: int = 20,
    init_latents: Optional[torch.Tensor] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """
    Generates an image using the Euler sampler for flow matching.
    
    Args:
        model: OmniFusionV2 model
        prompt_ids: [L] or [1, L] token IDs
        target_h: Target height in pixels
        target_w: Target width in pixels
        num_steps: Number of Euler steps
        init_latents: Optional initial latents [C, H, W]. Random if None.
        device: Device to use
        dtype: Data type
        
    Returns:
        Generated latents [1, C, H, W]
    """
    model.eval()
    patch_size = model.config.patch_size
    in_channels = model.config.in_channels
    
    # Prepare prompt
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    prompt_ids = prompt_ids.to(device)
    
    # Compute latent dimensions
    h_lat = (target_h // patch_size) * patch_size
    w_lat = (target_w // patch_size) * patch_size
    
    # Initialize latents
    if init_latents is None:
        latents = torch.randn(in_channels, h_lat, w_lat, device=device, dtype=dtype)
    else:
        latents = init_latents.to(device=device, dtype=dtype)
        if latents.dim() == 4:
            latents = latents.squeeze(0)
    
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        for step in range(num_steps):
            t_curr = step * dt
            t_batch = torch.full((1,), t_curr, device=device, dtype=dtype)
            
            # Forward pass
            out = model.forward([prompt_ids[0]], [latents], t_batch)
            
            # Extract velocity
            pred_v_packed = out["image"]
            mod_mask = out["modality_mask"]
            img_tokens = pred_v_packed[mod_mask == 1.0]
            
            # Unpatchify velocity
            p = patch_size
            c_vae = in_channels
            gh = h_lat // p
            gw = w_lat // p
            
            # [L, D] -> [C, H, W]
            img_tokens = img_tokens.view(gh, gw, c_vae * p * p)
            img_tokens = img_tokens.permute(2, 0, 1)
            v_pred = F.fold(
                img_tokens.view(1, c_vae * p * p, -1),  # FIX: Add batch dimension
                output_size=(h_lat, w_lat),
                kernel_size=p,
                stride=p
            ).squeeze(0)  # Remove batch dim
            
            # Euler step
            latents = latents + v_pred.squeeze(0) * dt
    
    return latents.unsqueeze(0)

def save_latent_as_image(
    latents: torch.Tensor,
    filepath: str,
    vae_scale_factor: float = 0.3611,
    vae_shift_factor: float = 0.1159
):
    """
    Decodes FLUX latents back to a high-quality RGB image.
    """
    if latents.dim() == 3:
        latents = latents.unsqueeze(0)
    
    # 1. Denormalize: (x / scale) + shift
    latents = (latents.float() / vae_scale_factor) + vae_shift_factor

    # 2. Decode using global VAE
    vae_model = get_vae()
    with torch.no_grad():
        decoded = vae_model.decode(latents.to(DEVICE, dtype=DTYPE))
        if hasattr(decoded, "sample"):
            decoded = decoded.sample
            
    # 3. Post-process [-1, 1] -> [0, 1]
    img_tensor = decoded.squeeze(0).float().cpu()
    img_tensor = (img_tensor + 1.0) / 2.0
    img_tensor = img_tensor.clamp(0, 1)

    # 4. Save
    img_numpy = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img_numpy).save(filepath)

# =============================================================================
# Fixed Noise Overfit Test (For Proper Reproduction Validation)
# =============================================================================

def run_fixed_noise_overfit_test(
    model: OmniFusionV2,
    test_samples: List[Tuple[str, Tuple[int, int, int], str]],
    data_dir: str,
    output_dir: str,
    num_steps: int = 500,
    learning_rate: float = 1e-3,
    num_gen_steps: int = 20,
    vae_scale_factor: float = 0.3611
) -> Dict[str, Any]:
    """
    Fixed-noise overfit test for proper reproduction validation.
    
    This test uses the SAME fixed noise for both training and generation,
    which enables the model to learn the exact velocity trajectory from
    noise -> target. This is the proper way to verify flow matching reproduction.
    
    Args:
        model: OmniFusionV2 model
        test_samples: List of (name, color, caption) tuples
        data_dir: Directory containing training images
        output_dir: Directory to save generated images
        num_steps: Number of training steps per sample
        learning_rate: Learning rate for overfitting
        num_gen_steps: Number of Euler generation steps
        vae_scale_factor: VAE scaling factor
        
    Returns:
        Dict with reproduction results including SSIM scores
    """
    logger.info("=" * 60)
    logger.info("FIXED NOISE OVERFIT TEST")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    results = []
    
    for name, color, caption in test_samples:
        logger.info(f"\n--- Processing: {name} ---")
        
        # Load original image
        orig_path = os.path.join(data_dir, f"{name}.png")
        orig_img = Image.open(orig_path).convert("RGB")
        orig_w, orig_h = orig_img.size
        
        # Convert to latent format
        orig_tensor = torch.tensor(np.array(orig_img)).float().permute(2, 0, 1) / 127.5 - 1.0
        target_latents = orig_tensor.unsqueeze(0).to(device=device, dtype=dtype) * vae_scale_factor
        
        # Tokenize prompt
        prompt_ids = ascii_tokenize(caption, length=64).to(device)
        
        # Generate FIXED noise (same for training and generation)
        torch.manual_seed(42 + hash(name) % 1000)
        fixed_noise = torch.randn_like(target_latents)
        
        # === PHASE 1: Training with Fixed Noise ===
        logger.info(f"[Phase 1] Training {name} for {num_steps} steps (fixed noise)...")
        
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        patch_size = model.config.patch_size
        in_channels = model.config.in_channels
        
        losses = []
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Sample random timestep
            t_val = torch.rand(1, device=device, dtype=dtype)
            
            # Interpolate: x_t = (1-t) * noise + t * target
            x_t = (1.0 - t_val) * fixed_noise + t_val * target_latents
            
            # Target velocity: v = target - noise
            v_target = target_latents - fixed_noise
            
            # Forward pass
            res = model([prompt_ids[0]], [x_t[0]], t_val)
            
            # Extract predicted velocity
            pred_v_packed = res["image"]
            mod_mask = res["modality_mask"]
            pred_img = pred_v_packed[mod_mask == 1.0]
            
            # Patchify target velocity for loss computation
            p = patch_size
            patches = v_target.unfold(2, p, p).unfold(3, p, p)
            gh, gw = patches.shape[2], patches.shape[3]
            target_flat = patches.permute(0, 2, 3, 1, 4, 5).reshape(1 * gh * gw, -1)
            
            # MSE loss
            loss = F.mse_loss(pred_img, target_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
            
            if step % 100 == 0 or step == num_steps - 1:
                logger.info(f"   Step {step}: Loss = {loss.item():.6f}")
        
        final_loss = losses[-1]
        
        # === PHASE 2: Generation with Fixed Noise ===
        logger.info(f"[Phase 2] Generating {name} from fixed noise...")
        
        model.eval()
        
        with torch.no_grad():
            # Start from the SAME fixed noise
            latents = fixed_noise[0].clone()
            
            h_lat = target_latents.shape[2]
            w_lat = target_latents.shape[3]
            
            dt = 1.0 / num_gen_steps
            
            for step in range(num_gen_steps):
                t_curr = step * dt
                t_batch = torch.full((1,), t_curr, device=device, dtype=dtype)
                
                out = model.forward([prompt_ids[0]], [latents], t_batch)
                
                # Extract velocity
                pred_v_packed = out["image"]
                mod_mask = out["modality_mask"]
                img_tokens = pred_v_packed[mod_mask == 1.0]
                
                # Unpatchify
                p = patch_size
                c_vae = in_channels
                gh = h_lat // p
                gw = w_lat // p
                
                img_tokens = img_tokens.view(gh, gw, c_vae * p * p)
                img_tokens = img_tokens.permute(2, 0, 1)
                v_pred = F.fold(
                    img_tokens.view(1, c_vae * p * p, -1),  # FIX: Add batch dimension
                    output_size=(h_lat, w_lat),
                    kernel_size=p,
                    stride=p
                ).squeeze(0)  # Remove batch dim
                
                # Euler step
                latents = latents + v_pred.squeeze(0) * dt
        
        gen_latents = latents.unsqueeze(0)
        
        # === PHASE 3: Compute Metrics ===
        ssim_score = compute_ssim(gen_latents[:, :3], target_latents[:, :3]).item()
        mse_score = F.mse_loss(gen_latents, target_latents).item()
        
        # Save generated image
        gen_path = os.path.join(output_dir, f"fixed_repro_{name}.png")
        save_latent_as_image(gen_latents, gen_path, vae_scale_factor)
        
        # Save comparison
        comparison_path = os.path.join(output_dir, f"fixed_comparison_{name}.png")
        create_comparison_image(target_latents, gen_latents, comparison_path, vae_scale_factor,
                                f"Original: {name}", f"Generated (SSIM: {ssim_score:.4f})")
        
        result = {
            "name": name,
            "caption": caption,
            "ssim": ssim_score,
            "mse": mse_score,
            "final_loss": final_loss,
            "gen_path": gen_path,
            "comparison_path": comparison_path
        }
        results.append(result)
        
        logger.info(f"   ✓ {name}: SSIM={ssim_score:.4f}, MSE={mse_score:.6f}, Loss={final_loss:.6f}")
        
        # Cleanup
        del optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # === Summary ===
    avg_ssim = sum(r["ssim"] for r in results) / len(results) if results else 0
    avg_mse = sum(r["mse"] for r in results) / len(results) if results else 0
    
    logger.info("\n" + "-" * 40)
    logger.info("Fixed Noise Overfit Results:")
    logger.info(f"   Average SSIM: {avg_ssim:.4f}")
    logger.info(f"   Average MSE:  {avg_mse:.6f}")
    
    passed = avg_ssim >= 0.5
    if passed:
        logger.info("   ✅ Fixed noise overfit test PASSED (SSIM >= 0.5)")
    else:
        logger.warning("   ⚠️ Fixed noise overfit test WARNING (SSIM < 0.5) - Try more training steps")
    
    return {
        "samples": results,
        "avg_ssim": avg_ssim,
        "avg_mse": avg_mse,
        "pass": passed
    }


# =============================================================================
# Reproduction & Generalization Tests
# =============================================================================

def run_reproduction_test(
    model: OmniFusionV2,
    trainer: 'Trainer',
    test_samples: List[Tuple[str, Tuple[int, int, int], str]],
    data_dir: str,
    output_dir: str,
    num_epochs: int = 50,
    num_gen_steps: int = 20,
    vae_scale_factor: float = 0.3611
) -> Dict[str, Any]:
    """
    Tests if the model can reproduce training data.
    
    1. Trains the model on provided samples
    2. Generates images from training prompts
    3. Computes SSIM between generated and original
    
    Args:
        model: OmniFusionV2 model
        trainer: Configured Trainer instance
        test_samples: List of (name, color, caption) tuples
        data_dir: Directory containing training images
        output_dir: Directory to save generated images
        num_epochs: Number of training epochs
        num_gen_steps: Number of Euler generation steps
        vae_scale_factor: VAE scaling factor
        
    Returns:
        Dict with reproduction results
    """
    logger.info("=" * 60)
    logger.info("REPRODUCTION TEST")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # --- Phase 1: Training ---
    logger.info(f"[Phase 1] Training for {num_epochs} epochs...")
    
    history = trainer.train(epochs=num_epochs)
    final_loss = history["train_loss"][-1] if history["train_loss"] else float('inf')
    logger.info(f"   Training complete. Final loss: {final_loss:.6f}")
    
    # --- Phase 2: Reproduction ---
    logger.info("[Phase 2] Generating reproductions from training prompts...")
    
    results = []
    
    for name, color, caption in test_samples:
        # Load original image
        orig_path = os.path.join(data_dir, f"{name}.png")
        orig_img = Image.open(orig_path).convert("RGB")
        orig_w, orig_h = orig_img.size
        
        # Convert to latent format
        orig_tensor = torch.tensor(np.array(orig_img)).float().permute(2, 0, 1) / 127.5 - 1.0
        orig_latents = orig_tensor.unsqueeze(0).to(device=device, dtype=dtype) * vae_scale_factor
        
        # Tokenize prompt
        prompt_ids = ascii_tokenize(caption, length=64)
        
        # Generate
        logger.info(f"   Generating: {name} ('{caption[:30]}...')")
        
        gen_latents = generate_image_euler(
            model=model,
            prompt_ids=prompt_ids.squeeze(0),
            target_h=orig_h,
            target_w=orig_w,
            num_steps=num_gen_steps,
            device=str(device),
            dtype=dtype
        )
        
        # Compute SSIM
        ssim_score = compute_ssim(gen_latents[:, :3], orig_latents[:, :3]).item()
        
        # Compute MSE
        mse_score = F.mse_loss(gen_latents, orig_latents).item()
        
        # Save generated image
        gen_path = os.path.join(output_dir, f"reproduced_{name}.png")
        save_latent_as_image(gen_latents, gen_path, vae_scale_factor)
        
        # Save comparison
        comparison_path = os.path.join(output_dir, f"comparison_{name}.png")
        create_comparison_image(orig_latents, gen_latents, comparison_path, vae_scale_factor, 
                                f"Original: {name}", f"Generated (SSIM: {ssim_score:.4f})")
        
        result = {
            "name": name,
            "caption": caption,
            "ssim": ssim_score,
            "mse": mse_score,
            "gen_path": gen_path,
            "comparison_path": comparison_path
        }
        results.append(result)
        
        logger.info(f"   ✓ {name}: SSIM={ssim_score:.4f}, MSE={mse_score:.6f}")
    
    # --- Summary ---
    avg_ssim = sum(r["ssim"] for r in results) / len(results) if results else 0
    avg_mse = sum(r["mse"] for r in results) / len(results) if results else 0
    
    logger.info("-" * 40)
    logger.info(f"Reproduction Results:")
    logger.info(f"   Average SSIM: {avg_ssim:.4f}")
    logger.info(f"   Average MSE:  {avg_mse:.6f}")
    
    if avg_ssim >= 0.5:
        logger.info("   ✅ Reproduction test PASSED (SSIM >= 0.5)")
    else:
        logger.warning("   ⚠️ Reproduction test WARNING (SSIM < 0.5)")
    
    return {
        "samples": results,
        "avg_ssim": avg_ssim,
        "avg_mse": avg_mse,
        "final_loss": final_loss,
        "pass": avg_ssim >= 0.5
    }


def run_generalization_test(
    model: OmniFusionV2,
    test_prompts: List[str],
    output_dir: str,
    target_size: int = 128,
    num_gen_steps: int = 20,
    vae_scale_factor: float = 0.3611
) -> Dict[str, Any]:
    """
    Tests model's ability to generate images from new prompts.
    
    Args:
        model: Trained OmniFusionV2 model
        test_prompts: List of new prompts to test
        output_dir: Directory to save generated images
        target_size: Target image size
        num_gen_steps: Number of Euler generation steps
        vae_scale_factor: VAE scaling factor
        
    Returns:
        Dict with generalization results
    """
    logger.info("=" * 60)
    logger.info("GENERALIZATION TEST")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    results = []
    
    for i, prompt in enumerate(test_prompts):
        logger.info(f"   [{i+1}/{len(test_prompts)}] Generating: '{prompt[:40]}...'")
        
        prompt_ids = ascii_tokenize(prompt, length=64)
        
        gen_latents = generate_image_euler(
            model=model,
            prompt_ids=prompt_ids.squeeze(0),
            target_h=target_size,
            target_w=target_size,
            num_steps=num_gen_steps,
            device=str(device),
            dtype=dtype
        )
        
        # Save generated image
        safe_name = f"gen_{i:03d}"
        gen_path = os.path.join(output_dir, f"{safe_name}.png")
        save_latent_as_image(gen_latents, gen_path, vae_scale_factor)
        
        # Compute basic stats (variance as a proxy for non-trivial generation)
        variance = gen_latents.var().item()
        
        results.append({
            "prompt": prompt,
            "gen_path": gen_path,
            "variance": variance
        })
        
        logger.info(f"   ✓ Saved to {gen_path} (variance: {variance:.4f})")
    
    avg_variance = sum(r["variance"] for r in results) / len(results) if results else 0
    
    logger.info("-" * 40)
    logger.info(f"Generalization Results:")
    logger.info(f"   Generated {len(results)} images")
    logger.info(f"   Average variance: {avg_variance:.4f}")
    
    return {
        "samples": results,
        "avg_variance": avg_variance,
        "count": len(results)
    }


def create_comparison_image(
    original: torch.Tensor,
    generated: torch.Tensor,
    filepath: str,
    vae_scale_factor: float = 0.3611,
    label_orig: str = "Original",
    label_gen: str = "Generated"
):
    """Creates a side-by-side comparison image."""
    from PIL import ImageDraw, ImageFont
    
    def to_pil(latents):
        if latents.dim() == 4:
            latents = latents.squeeze(0)
        img = latents[:3].float().cpu()
        img = img / vae_scale_factor
        img = (img + 1.0) / 2.0
        img = img.clamp(0, 1)
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(img)
    
    orig_pil = to_pil(original)
    gen_pil = to_pil(generated)
    
    # Create side-by-side
    w, h = orig_pil.size
    comparison = Image.new('RGB', (w * 2 + 10, h + 30), color='white')
    comparison.paste(orig_pil, (0, 25))
    comparison.paste(gen_pil, (w + 10, 25))
    
    # Add labels
    draw = ImageDraw.Draw(comparison)
    draw.text((w // 4, 5), label_orig, fill='black')
    draw.text((w + 10 + w // 4, 5), label_gen, fill='black')
    
    comparison.save(filepath)


# =============================================================================
# Extended Smoke Test with Reproduction & SSIM
# =============================================================================

def run_smoke_test():
    """
    Comprehensive smoke test for the training backend.
    Tests:
    1. Model creation
    2. Data loading
    3. Forward pass
    4. Backward pass
    5. Optimizer step
    6. EMA update
    7. Checkpoint save/load
    8. Reproduction test with SSIM
    9. Generalization test
    """
    import tempfile
    from PIL import Image, ImageDraw
    
    logger.info("=" * 60)
    logger.info("TRAINING BACKEND SMOKE TEST (Extended)")
    logger.info("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy training data
        logger.info("[1/9] Creating dummy training data...")
        
        data_dir = os.path.join(tmpdir, "train_data")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(data_dir)
        os.makedirs(output_dir)
        
        test_samples = [
            ("red_square", (255, 0, 0), "a vibrant red square on black background"),
            ("blue_circle", (0, 0, 255), "a deep blue circle floating in space"),
            ("green_triangle", (0, 255, 0), "a bright green triangle pointing up"),
        ]
        
        for name, color, caption in test_samples:
            # Create image with distinct shape
            img = Image.new('RGB', (64, 64), color='black')
            draw = ImageDraw.Draw(img)
            if "square" in name:
                draw.rectangle([16, 16, 48, 48], fill=color)
            elif "circle" in name:
                draw.ellipse([16, 16, 48, 48], fill=color)
            else:
                draw.polygon([(32, 16), (16, 48), (48, 48)], fill=color)
            img.save(os.path.join(data_dir, f"{name}.png"))
            
            with open(os.path.join(data_dir, f"{name}.txt"), "w") as f:
                f.write(caption)
        
        logger.info("   ✓ Created 3 training samples")
        
        # Create config
        logger.info("[2/9] Creating model and config...")
        
        config = TrainingConfig(
            data_paths=[data_dir],
            d_model=128,
            n_layers=2,
            n_heads=4,
            head_dim=32,
            patch_size=8,
            in_channels=16,
            batch_size=2,
            epochs=10,  # More epochs for better reproduction
            learning_rate=5e-3,  # Higher LR for fast overfitting
            warmup_steps=5,
            save_dir=os.path.join(tmpdir, "checkpoints"),
            tensorboard_dir=os.path.join(tmpdir, "runs"),
            use_ema=True,
            log_interval=5,
            save_interval=50,
            grad_checkpointing=False,
        )
        
        model = create_model(config)
        logger.info(f"   ✓ Created model with {sum(p.numel() for p in model.parameters()):,} params")
        
        # Create trainer
        logger.info("[3/9] Creating trainer with dataloader...")
        
        trainer = create_trainer(model, config, [data_dir])
        logger.info(f"   ✓ Trainer created, dataloader has {len(trainer.dataloader)} batches")
        
        # Training step test
        logger.info("[4/9] Testing training step...")
        
        for batch in trainer.dataloader:
            loss = trainer._training_step(batch)
            if loss is not None:
                logger.info(f"   ✓ Training step completed. Loss: {loss.item():.6f}")
                break
        
        # Optimizer step test
        logger.info("[5/9] Testing optimizer step...")
        trainer.amp_manager.step(trainer.optimizer, trainer.model, config.max_grad_norm)
        trainer.optimizer.zero_grad()
        logger.info("   ✓ Optimizer step completed")
        
        # EMA test
        logger.info("[6/9] Testing EMA update...")
        
        if trainer.ema is not None:
            trainer.ema.update(model)
            logger.info("   ✓ EMA updated successfully")
        
        # Checkpoint test
        logger.info("[7/9] Testing checkpoint save/load...")
        
        trainer.save_checkpoint(epoch=0, is_best=True)
        
        new_model = create_model(config)
        new_trainer = Trainer(new_model, config)
        
        ckpt_path = os.path.join(config.save_dir, "best_model.pt")
        new_trainer.load_checkpoint(ckpt_path)
        logger.info("   ✓ Checkpoint save/load verified")
        
        # Fixed noise overfit test (proper reproduction validation)
        logger.info("[8/9] Running fixed noise overfit test (proper SSIM validation)...")
        
        fresh_model = create_model(config)
        fresh_model.to(config.device)
        if DTYPE == torch.bfloat16:
            fresh_model = fresh_model.bfloat16()
        
        overfit_results = run_fixed_noise_overfit_test(
            model=fresh_model,
            test_samples=test_samples,
            data_dir=data_dir,
            output_dir=output_dir,
            num_steps=500,  # Enough steps to overfit
            learning_rate=1e-3,
            num_gen_steps=20,
            vae_scale_factor=config.vae_scale_factor
        )
        
        # Generalization test
        logger.info("[9/9] Running generalization test...")
        
        gen_prompts = [
            "a yellow star shining bright",
            "a purple hexagon with dots",
            "an orange diamond pattern",
        ]
        
        gen_results = run_generalization_test(
            model=fresh_model,
            test_prompts=gen_prompts,
            output_dir=output_dir,
            target_size=64,
            num_gen_steps=20,
            vae_scale_factor=config.vae_scale_factor
        )
        
    # Final Summary
    logger.info("=" * 60)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"   Fixed Noise Overfit Avg SSIM: {overfit_results['avg_ssim']:.4f}")
    logger.info(f"   Fixed Noise Overfit Avg MSE:  {overfit_results['avg_mse']:.6f}")
    logger.info(f"   Generalization Images: {gen_results['count']}")
    logger.info(f"   Generalization Variance: {gen_results['avg_variance']:.4f}")
    
    if overfit_results['pass']:
        logger.info("✅ ALL SMOKE TESTS PASSED!")
    else:
        logger.warning("⚠️ OVERFIT SSIM < 0.5 - Model may need more training steps")
    
    logger.info("=" * 60)


# =============================================================================
# Train_Img Test (Similar to minimal_overfit_test.py)
# =============================================================================

def load_train_img_data(
    data_dir: str = "Train_Img",
    patch_size: int = 8,
    in_channels: int = 16,
    max_size: int = 256,
    vae_scale_factor: float = 0.3611,
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE
) -> List[Dict[str, Any]]:
    """
    Loads training data from Train_Img folder.
    
    Creates dummy data if folder is empty.
    Returns list of dicts with:
        - name: str
        - latents: [1, C, H, W] tensor
        - input_ids: [1, L] tensor
        - text: str
        - h: latent height
        - w: latent width
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Find image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(image_extensions)]
    
    # Create dummy data if empty
    if not files:
        logger.info("Creating dummy training data in Train_Img...")
        
        samples = [
            ("green_image", "green", "an image displaying a field of uniform green"),
            ("blue_image", "blue", "a deep immersive background of solid blue"),
            ("red_image", "red", "a vibrant canvas of pure solid red"),
        ]
        
        for name, color, caption in samples:
            img = Image.new('RGB', (256, 256), color=color)
            img.save(os.path.join(data_dir, f"{name}.png"))
            with open(os.path.join(data_dir, f"{name}.txt"), "w") as f:
                f.write(caption)
        
        files = [f"{name}.png" for name, _, _ in samples]
    
    data_items = []
    
    for fname in files:
        base_name = os.path.splitext(fname)[0]
        img_path = os.path.join(data_dir, fname)
        
        # Try multiple caption formats
        txt_path = os.path.join(data_dir, base_name + ".txt")
        json_path = os.path.join(data_dir, base_name + ".json")
        
        try:
            # Load image
            raw_img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = raw_img.size
            
            # Resize if too large
            scale = min(max_size / orig_w, max_size / orig_h)
            if scale < 1.0:
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                raw_img = raw_img.resize((new_w, new_h), resample=Image.LANCZOS)
                orig_w, orig_h = raw_img.size
            
            # Align to patch size
            target_h = (orig_h // patch_size) * patch_size
            target_w = (orig_w // patch_size) * patch_size
            target_h = max(patch_size, target_h)
            target_w = max(patch_size, target_w)
            
            raw_img = raw_img.resize((target_w, target_h), resample=Image.LANCZOS)
            
            # Convert to tensor
            img_tensor = torch.tensor(np.array(raw_img)).float().permute(2, 0, 1) / 127.5 - 1.0
            
            # Expand channels if needed
            if img_tensor.shape[0] < in_channels:
                repeats = (in_channels // img_tensor.shape[0]) + 1
                img_tensor = img_tensor.repeat(repeats, 1, 1)[:in_channels]
            
            vae_instance = get_vae() # Get the actual VAE instance
            with torch.no_grad():
                latents = vae_instance.encode(img_tensor.unsqueeze(0).to(device, dtype))

            # Then normalize the 16-channel output for the Flow Matching model
            latents = (latents - config.vae_shift_factor) * config.vae_scale_factor
            
            # Load caption
            text_content = f"image of {base_name}"  # Default
            
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    text_content = f.read().strip()
            elif os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Try common keys
                    text_content = data.get("caption", data.get("text", data.get("prompt", text_content)))
            
            input_ids = ascii_tokenize(text_content, length=64)
            
            data_items.append({
                "name": base_name,
                "latents": latents,
                "input_ids": input_ids,
                "text": text_content,
                "h": target_h // patch_size,
                "w": target_w // patch_size,
                "original_path": img_path
            })
            
            logger.info(f"Loaded: {fname} -> Latent: {target_w}x{target_h}, Caption: '{text_content[:50]}...'")
            
        except Exception as e:
            logger.warning(f"Failed to load {fname}: {e}")
    
    return data_items


def run_train_img_test(
    num_steps: int = 500,
    learning_rate: float = 1e-3,
    num_gen_steps: int = 20,
    max_samples: int = 5,
    output_dir: str = "train_img_outputs"
):
    """
    Runs fixed-noise SSIM tests on Train_Img folder.
    
    Similar to minimal_overfit_test.py workflow:
    1. Loads images from Train_Img
    2. Trains with fixed noise per sample
    3. Generates reproduction from same fixed noise
    4. Computes and reports SSIM
    
    Args:
        num_steps: Training steps per sample
        learning_rate: Learning rate for overfitting
        num_gen_steps: Euler generation steps
        max_samples: Maximum samples to process
        output_dir: Directory for output images
    """
    logger.info("=" * 60)
    logger.info("TRAIN_IMG SIMILARITY TEST")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    config = TrainingConfig(
        d_model=384,
        n_layers=6,
        n_heads=6,
        head_dim=64,
        patch_size=8,
        in_channels=16,
        vae_scale_factor=VAE_SCALE_FACTOR,
        grad_checkpointing=False,
    )
    
    model = create_model(config)
    model.to(DEVICE)
    if DTYPE == torch.bfloat16:
        model = model.bfloat16()
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load training data
    data_items = load_train_img_data(
        data_dir="Train_Img",
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        vae_scale_factor=config.vae_scale_factor
    )
    
    if not data_items:
        logger.error("No training data found!")
        return
    
    # Limit samples
    if len(data_items) > max_samples:
        data_items = data_items[:max_samples]
        logger.info(f"Limited to {max_samples} samples for testing")
    
    results = []
    
    for idx, item in enumerate(data_items):
        name = item["name"]
        target_latents = item["latents"]
        prompt_ids = item["input_ids"]
        text = item["text"]
        h_lat = item["h"]
        w_lat = item["w"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[{idx+1}/{len(data_items)}] Processing: {name}")
        logger.info(f"   Caption: {text[:60]}...")
        logger.info(f"   Latent size: {w_lat}x{h_lat}")
        logger.info("=" * 60)
        
        # Generate fixed noise
        torch.manual_seed(42 + idx)
        fixed_noise = torch.randn_like(target_latents)
        
        # === PHASE 1: Train with Fixed Noise ===
        logger.info(f"[Phase 1] Training for {num_steps} steps...")
        
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        patch_size = config.patch_size
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Sample random timestep
            t_val = torch.rand(1, device=DEVICE, dtype=DTYPE)
            
            # Interpolate: x_t = (1-t) * noise + t * target
            x_t = (1.0 - t_val) * fixed_noise + t_val * target_latents
            
            # Target velocity: v = target - noise
            v_target = target_latents - fixed_noise
            
            # Forward pass
            res = model([prompt_ids[0]], [x_t[0]], t_val)
            
            # Extract predicted velocity
            pred_v_packed = res["image"]
            mod_mask = res["modality_mask"]
            pred_img = pred_v_packed[mod_mask == 1.0]
            
            # Patchify target velocity
            p = patch_size
            patches = v_target.unfold(2, p, p).unfold(3, p, p)
            gh, gw = patches.shape[2], patches.shape[3]
            target_flat = patches.permute(0, 2, 3, 1, 4, 5).reshape(gh * gw, -1)
            
            # MSE loss
            loss = F.mse_loss(pred_img, target_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 100 == 0 or step == num_steps - 1:
                logger.info(f"   Step {step}: Loss = {loss.item():.6f}")
        
        final_loss = loss.item()
        
        # === PHASE 2: Generate from Fixed Noise ===
        logger.info(f"[Phase 2] Generating reproduction...")
        
        gen_latents = generate_image_euler(
            model=model,
            prompt_ids=prompt_ids.squeeze(0),
            target_h=target_latents.shape[2],
            target_w=target_latents.shape[3],
            num_steps=num_gen_steps,
            init_latents=fixed_noise,
            device=DEVICE,
            dtype=DTYPE
        )
        
        # === PHASE 3: Compute Metrics ===
        ssim_score = compute_ssim(gen_latents[:, :3], target_latents[:, :3]).item()
        mse_score = F.mse_loss(gen_latents, target_latents).item()
        
        logger.info(f"[Phase 3] Results:")
        logger.info(f"   SSIM: {ssim_score:.4f}")
        logger.info(f"   MSE:  {mse_score:.6f}")
        logger.info(f"   Loss: {final_loss:.6f}")
        
        # Save outputs
        gen_path = os.path.join(output_dir, f"repro_{name}.png")
        save_latent_as_image(gen_latents, gen_path, config.vae_scale_factor)
        
        comparison_path = os.path.join(output_dir, f"compare_{name}.png")
        create_comparison_image(
            target_latents, gen_latents, comparison_path, config.vae_scale_factor,
            f"Original: {name}", f"Generated (SSIM: {ssim_score:.4f})"
        )
        
        results.append({
            "name": name,
            "ssim": ssim_score,
            "mse": mse_score,
            "loss": final_loss,
            "gen_path": gen_path,
            "comparison_path": comparison_path
        })
        
        if ssim_score >= 0.5:
            logger.info(f"   ✅ PASSED (SSIM >= 0.5)")
        else:
            logger.warning(f"   ⚠️ WARNING (SSIM < 0.5)")
        
        # Cleanup
        del optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # === Final Summary ===
    logger.info("\n" + "=" * 60)
    logger.info("TRAIN_IMG TEST SUMMARY")
    logger.info("=" * 60)
    
    avg_ssim = sum(r["ssim"] for r in results) / len(results) if results else 0
    avg_mse = sum(r["mse"] for r in results) / len(results) if results else 0
    
    logger.info(f"\nPer-Sample Results:")
    for r in results:
        status = "✅" if r["ssim"] >= 0.5 else "⚠️"
        logger.info(f"   {status} {r['name']}: SSIM={r['ssim']:.4f}, MSE={r['mse']:.6f}")
    
    logger.info(f"\nAggregated:")
    logger.info(f"   Average SSIM: {avg_ssim:.4f}")
    logger.info(f"   Average MSE:  {avg_mse:.6f}")
    logger.info(f"   Samples Tested: {len(results)}")
    logger.info(f"   Output Directory: {output_dir}")
    
    passed = sum(1 for r in results if r["ssim"] >= 0.5)
    logger.info(f"\n   Passed: {passed}/{len(results)}")
    
    if avg_ssim >= 0.5:
        logger.info("\n✅ OVERALL: PASSED (Average SSIM >= 0.5)")
    else:
        logger.warning("\n⚠️ OVERALL: NEEDS MORE TRAINING (Average SSIM < 0.5)")
    
    logger.info("=" * 60)
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OmniFusion-X V2 Training Backend")
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "train_img"],
                        help="Test mode: 'smoke' for quick smoke test, 'train_img' for Train_Img SSIM test")
    parser.add_argument("--steps", type=int, default=500, help="Training steps per sample (train_img mode)")
    parser.add_argument("--max-samples", type=int, default=3, help="Max samples to test (train_img mode)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    
    if args.mode == "train_img":
        run_train_img_test(
            num_steps=args.steps,
            learning_rate=args.lr,
            max_samples=args.max_samples
        )
    else:
        run_smoke_test()
