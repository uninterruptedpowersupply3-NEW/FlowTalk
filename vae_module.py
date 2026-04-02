import os
import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class FluxVAE(nn.Module):
    """
    Corrected Wrapper for Flux VAE (AutoencoderKL).
    FLUX uses 16 channels and a 16x downsampling factor.
    """
    def __init__(self, model_id="black-forest-labs/FLUX.1-schnell", dtype=torch.bfloat16, local_files_only: bool | None = None):
        super().__init__()
        self.model_id = model_id
        self.dtype = dtype
        if local_files_only is None:
            # If HF_HUB_OFFLINE is enabled, enforce local-only loads to avoid any network calls.
            local_files_only = os.environ.get("HF_HUB_OFFLINE", "0") not in ("0", "", "false", "False")
        
        # Flux VAE Constants
        self.scaling_factor = 0.3611
        self.shift_factor = 0.1159
        
        try:
            self.vae = AutoencoderKL.from_pretrained(
                model_id, 
                subfolder="vae", 
                torch_dtype=dtype,
                local_files_only=local_files_only,
            )
            self.vae.requires_grad_(False)
            self.vae.eval()
            print(f"Successfully loaded Flux VAE from {model_id}")
            
        except OSError:
            print(f"Error: Could not load {model_id}. Fallback init is not recommended for Flux due to 16-channel requirement.")
            # If you absolutely must fallback, you MUST use latent_channels=16
            raise RuntimeError("Flux VAE must be loaded from a valid pretrained path to ensure 16-channel architecture.")

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encodes image to latents and applies Flux normalization.
        Input Image: [B, 3, H, W] in range [-1, 1]
        Returns: [B, 16, H/16, W/16] normalized latents
        """
        if self.vae.device != image.device:
            self.vae.to(image.device)
            
        with torch.no_grad():
            # 1. Get raw latents from posterior sample
            # Flux VAE output is [B, 16, H/16, W/16]
            latents = self.vae.encode(image.to(self.dtype)).latent_dist.sample()
            
            # 2. Apply Flux-specific scaling and shift
            # Formula: (latent - shift) * scale
            latents = (latents - self.shift_factor) * self.scaling_factor
            
            return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Undoes Flux normalization and decodes latents to image.
        Input Latents: [B, 16, H/16, W/16]
        Returns: [B, 3, H, W] in range [-1, 1]
        """
        if self.vae.device != latents.device:
            self.vae.to(latents.device)
            
        with torch.no_grad():
            # 1. Undo Flux-specific scaling and shift
            # Formula: (latent / scale) + shift
            latents = (latents / self.scaling_factor) + self.shift_factor
            
            # 2. Decode to RGB
            image = self.vae.decode(latents.to(self.dtype)).sample
            return image
