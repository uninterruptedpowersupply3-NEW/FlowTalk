
import os
import sys
import torch
import logging
import argparse
import inspect
from PIL import Image
import numpy as np
import torch.nn.functional as F
import glob

# Import from working script (but NOT the non-existent VAE constants)
from test_dataset_generalization import DatasetGeneralizationTest, TestConfig, DEVICE, DTYPE, encode_prompt_tokens, empty_prompt_tokens
from data_manager import TiktokenTokenizer, IMAGE_TOKEN

# VAE constants from FluxVAE (vae_module.py)
VAE_SCALE_FACTOR = 0.3611
VAE_SHIFT_FACTOR = 0.1159

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Inference")

def _chatml_one_turn(user_text: str, assistant_text: str) -> str:
    return (
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_text}<|im_end|>"
    )


def _detect_cache_chatml(cache_dir: str, tokenizer: TiktokenTokenizer, *, sample: int = 8) -> tuple[bool, str]:
    """
    Best-effort detection of whether cached tokens were stored in ChatML format.

    Returns:
      (is_chatml, suggested_user_prompt)
    """
    try:
        index_path = os.path.join(cache_dir, "index.json")
        if not os.path.exists(index_path):
            return False, "Describe this image."

        import json
        import random
        import struct

        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        keys = list(index.keys())
        if not keys:
            return False, "Describe this image."

        rng = random.Random(0)
        if len(keys) > int(sample):
            keys = rng.sample(keys, int(sample))

        user_counts: dict[str, int] = {}

        for k in keys:
            info = index.get(k)
            if not isinstance(info, dict):
                continue
            shard_path = os.path.join(cache_dir, f"shard_{int(info['shard']):04d}.bin")
            if not os.path.exists(shard_path):
                continue

            with open(shard_path, "rb") as f:
                f.seek(int(info["offset"]))
                latent_len = struct.unpack("<I", f.read(4))[0]
                f.read(latent_len)
                token_len = struct.unpack("<I", f.read(4))[0]
                token_bytes = f.read(token_len)

            t_dtype = np.int32 if info.get("token_dtype", "int32") == "int32" else np.int64
            token_ids = np.frombuffer(token_bytes, dtype=t_dtype)
            try:
                text = tokenizer.decode(token_ids.tolist())
            except Exception:
                continue

            if "<|im_start|>" not in text or "<|im_end|>" not in text:
                continue

            # Extract the user block if present to pick a reasonable default wrapper prompt.
            # Expected pattern:
            #   <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
            u_start = text.find("<|im_start|>user\n")
            if u_start >= 0:
                u_start += len("<|im_start|>user\n")
                u_end = text.find("<|im_end|>", u_start)
                if u_end > u_start:
                    user_text = " ".join(text[u_start:u_end].strip().split())
                    if user_text:
                        user_counts[user_text] = user_counts.get(user_text, 0) + 1

            # If any sampled item contains ChatML tokens, treat the cache as ChatML.
            suggested = "Describe this image."
            if user_counts:
                suggested = max(user_counts.items(), key=lambda kv: kv[1])[0]
            return True, suggested

        return False, "Describe this image."
    except Exception:
        return False, "Describe this image."


class InferenceModel(DatasetGeneralizationTest):
    """
    Robust Inference Solution wrapping the verified DatasetGeneralizationTest environment.
    """
    def __init__(self, model_path=None):
        # Config matching the training/test setup (512 d_model, 8 layers/heads from logs)
        # Note: We hardcode 512/8/8 as verified by debug scripts.
        config = TestConfig(d_model=512, n_layers=8, n_heads=8)
        
        # Initialize basic props needed by parent
        self.config = config
        self.device = DEVICE
        self.dtype = DTYPE
        
        # Manually verify path
        if model_path is None:
            model_path = os.path.join("dataset_gen_checkpoints", "trained_model.pt")
            
        if not os.path.exists(model_path):
             # Try auto-discovery
            import glob
            search_pattern = os.path.join(os.path.dirname(model_path), "*.pt")
            files = glob.glob(search_pattern)
            if files:
                model_path = max(files, key=os.path.getctime)
                logger.info(f"Using latest checkpoint found: {model_path}")
            else:
                 raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        logger.info(f"Loading checkpoint metadata from: {model_path}")
        
        # 1. Load checkpoint FIRST to inspect shapes.
        # Some checkpoints are raw state_dicts, others are wrapped dicts with {model_state_dict, config, ...}.
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=True)
        ckpt_config = None
        state_dict = ckpt
        if isinstance(ckpt, dict) and ("model_state_dict" in ckpt or "model" in ckpt):
            ckpt_config = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else None
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt["model"]
        
        # Strip compile prefixes
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        # 2. INTROSPECTION: Deduce Config from Weights
        # d_model from patch_embed
        if "patch_embed.weight" in state_dict:
            d_model = state_dict["patch_embed.weight"].shape[0] # [D, C, H, W]
        elif "text_embed.weight" in state_dict:
            d_model = state_dict["text_embed.weight"].shape[1] # [V, D]
        else:
            raise ValueError("Could not deduce d_model from state_dict (no patch_embed or text_embed)")
            
        # n_layers from blocks
        # Keys look like "blocks.0.xxx", "blocks.1.xxx"
        layer_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith("blocks.")]
        n_layers = max(layer_indices) + 1 if layer_indices else 0
        
        # vocab_size
        vocab_size = 100352 # Default
        if "text_embed.weight" in state_dict:
            vocab_size = state_dict["text_embed.weight"].shape[0]
            
        # n_heads
        # Assumption: head_dim is 64 (standard for this architecture)
        head_dim = 64
        n_heads = d_model // head_dim
        
        # Verify assumption if possible via q_proj
        # q_proj weight: [n_heads * head_dim, d_model]
        if "blocks.0.attn.q_proj.weight" in state_dict:
            q_out_dim = state_dict["blocks.0.attn.q_proj.weight"].shape[0]
            if q_out_dim != d_model:
                # If Q projection is not d_model, calculating n_heads might differ
                # But usually q_dim = d_model in this codebase
                logger.info(f"   Note: q_proj dim {q_out_dim} != d_model {d_model}")
                n_heads = q_out_dim // head_dim

        logger.info(f"   [AUTO-DETECT] d_model={d_model}, n_layers={n_layers}, n_heads={n_heads} (head_dim={head_dim}), vocab={vocab_size}")

        # Detect optional architecture features from checkpoint keys.
        # Newer checkpoints may include a learned text attention pooling module for pooled conditioning.
        text_pooling = "attn" if any(k.startswith("text_attn_pool.") for k in state_dict.keys()) else "mean"
        if ckpt_config is not None:
            cfg_pool = ckpt_config.get("text_pooling")
            if cfg_pool in ("mean", "attn") and cfg_pool != text_pooling:
                logger.warning(
                    f"Checkpoint config text_pooling={cfg_pool!r} disagrees with weight keys; using {text_pooling!r}."
                )
        if text_pooling == "attn":
            logger.info("   [AUTO-DETECT] text_pooling=attn (learned attention pooling present in checkpoint)")

        # 3. Create Config
        config = TestConfig(
            d_model=d_model, 
            n_layers=n_layers, 
            n_heads=n_heads,
            # We assume patch_size/in_channels didn't change (16/16) or strictly hardcoded in codebase defaults
        )
        
        # Initialize basic props needed by parent
        self.config = config
        self.device = DEVICE
        self.dtype = DTYPE
        
        # Init Model (OmniFusionV2)
        from omni_model_v2 import OmniFusionV2, OmniConfigV2
        
        # Explicitly replicate the config used in robust_gen / test_dataset
        pooled_scale = 1.0
        pooled_drop = 0.0
        if ckpt_config is not None:
            try:
                pooled_scale = float(ckpt_config.get("pooled_text_cond_scale", pooled_scale))
            except Exception:
                pooled_scale = 1.0
            try:
                pooled_drop = float(ckpt_config.get("pooled_text_drop_prob", pooled_drop))
            except Exception:
                pooled_drop = 0.0

        self.omni_config = OmniConfigV2(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            vocab_size=vocab_size,     # Use detected vocab size
            qk_norm=True,             # Critical: Test script uses True
            attention_logit_cap=50.0,  # Critical: Test script uses 50.0
            text_pooling=text_pooling,
            pooled_text_cond_scale=pooled_scale,
            pooled_text_drop_prob=pooled_drop,
        )
        
        self.model = OmniFusionV2(self.omni_config).to(self.device)
        self.model.eval()
            
        # Load weights. We validate missing/unexpected keys to avoid silently running with
        # partially-random parameters (which can look like "prompt is ignored" / "blob prior").
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if missing or unexpected:
            msg = (
                f"Checkpoint load had mismatched keys: missing={len(missing)} unexpected={len(unexpected)}. "
                f"Examples missing={missing[:5]} unexpected={unexpected[:5]}"
            )
            allow_partial = os.environ.get("OMNIFUSION_ALLOW_PARTIAL_LOAD", "0") not in ("0", "", "false", "False")
            if allow_partial:
                logger.warning(msg)
            else:
                raise RuntimeError(
                    msg
                    + " | Refusing to continue with partial load. "
                    + "Set OMNIFUSION_ALLOW_PARTIAL_LOAD=1 to override (not recommended)."
                )
        if hasattr(self.model, "zero_padding_embedding"):
            self.model.zero_padding_embedding()
        
        if self.dtype == torch.bfloat16:
            self.model.bfloat16()
            
        # Initialize VAE (Flux)
        from vae_module import FluxVAE

        # Default to offline Hub usage for the GUI to avoid any Hugging Face network calls at runtime.
        # Users who explicitly want online behavior (e.g. first-time download) can set:
        #   set OMNIFUSION_HF_OFFLINE=0
        force_offline = os.environ.get("OMNIFUSION_HF_OFFLINE", "1") not in ("0", "", "false", "False")
        if force_offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        self.vae = FluxVAE(local_files_only=force_offline).to(self.device).eval()
        
        # Initialize Tokenizer
        from data_manager import TiktokenTokenizer
        self.tokenizer = TiktokenTokenizer()
        
        # Mock dataset object for parent compatibility if we call parent methods (optional)
        self.dataset = type('obj', (object,), {'vae': self.vae, 'tokenizer': self.tokenizer})

        # Prompt-format compatibility:
        # Some latent caches (notably JSON multi-caption datasets) store tokens as ChatML.
        # If a checkpoint was trained on ChatML tokens, encoding a plain prompt string at
        # inference is out-of-distribution and often looks like "prompt ignored / blob prior".
        self._wrap_prompt_chatml = False
        self._chatml_user_prompt = "Describe this image."
        if ckpt_config is not None:
            cache_dir = ckpt_config.get("cache_dir")
            if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
                is_chatml, user_prompt = _detect_cache_chatml(cache_dir, self.tokenizer, sample=8)
                self._wrap_prompt_chatml = bool(is_chatml)
                self._chatml_user_prompt = str(user_prompt or self._chatml_user_prompt)
                if self._wrap_prompt_chatml:
                    logger.info(f"   [PROMPT FORMAT] Detected ChatML cache tokens (cache_dir={cache_dir}).")
                    logger.info(f"   [PROMPT FORMAT] Wrapping image-generation prompts with user='{self._chatml_user_prompt}'.")

        

    

    def generate_image(self, prompt, output_path="output.png", steps=50, cfg=4.0, width=128, height=128):
        """
        Stream Image Generation using valid CFG logic and mathematically correct VAE scaling.
        """
        logger.info(f" Generating Image | Prompt: '{prompt}' | Steps: {steps} | Size: {width}x{height} | CFG: {cfg}")
        
        # Match training tokenization / format.
        prompt_text = str(prompt)
        if self._wrap_prompt_chatml and "<|im_start|>" not in prompt_text:
            prompt_text = _chatml_one_turn(self._chatml_user_prompt, prompt_text)
        prompt_ids = encode_prompt_tokens(self.tokenizer, prompt_text).to(self.device)
        # IMPORTANT: Keep unconditional prompt length identical to conditional so image-token
        # RoPE temporal positions align for CFG subtraction.
        uncond_ids = prompt_ids.new_full(prompt_ids.shape, int(self.tokenizer.pad_token))
        uncond_ids[-1] = int(self.tokenizer.eot_token)
        
        h_lat = height // 8
        w_lat = width // 8
        in_channels = self.config.in_channels
        
        latents_gen = torch.randn(in_channels, h_lat, w_lat, device=self.device, dtype=self.dtype)
        
        dt = 1.0 / steps
        patch_size = self.config.patch_size
        
        with torch.no_grad():
            for step in range(steps):
                t_curr = step * dt
                # Batch of 2 for CFG (Cond + Uncond)
                t_batch = torch.full((2,), t_curr, device=self.device, dtype=torch.float32)
                
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    txt_input = [prompt_ids, uncond_ids]
                    img_input = [latents_gen, latents_gen]

                    try:
                        # Fast path: single forward with (cond + uncond) batched together.
                        # Requires correct document isolation (FlexAttention) for packed sequences.
                        out = self.model(txt_input, images=img_input, timesteps=t_batch, causal_text=True)

                        pred_v_packed = out["image"]
                        mod_mask = out["modality_mask"]
                        cu_seqlens = out["cu_seqlens"]

                        # 1. Extract Conditional Output
                        start_c, end_c = cu_seqlens[0], cu_seqlens[1]
                        img_tok_c = pred_v_packed[start_c:end_c][mod_mask[start_c:end_c] == 1.0]

                        # 2. Extract Unconditional Output
                        start_u, end_u = cu_seqlens[1], cu_seqlens[2]
                        img_tok_u = pred_v_packed[start_u:end_u][mod_mask[start_u:end_u] == 1.0]
                    except RuntimeError as e:
                        msg = str(e)
                        # Correctness-first fallback:
                        # If FlexAttention is unavailable/failed for packed sequences, run the two passes
                        # separately (B=1 each). This avoids cross-document leakage entirely.
                        if "FlexAttention" not in msg or "packed" not in msg:
                            raise

                        t_one = torch.full((1,), t_curr, device=self.device, dtype=torch.float32)
                        out_c = self.model([prompt_ids], images=[latents_gen], timesteps=t_one, causal_text=True)
                        out_u = self.model([uncond_ids], images=[latents_gen], timesteps=t_one, causal_text=True)

                        pred_c = out_c["image"]
                        mask_c = out_c["modality_mask"]
                        img_tok_c = pred_c[mask_c == 1.0]

                        pred_u = out_u["image"]
                        mask_u = out_u["modality_mask"]
                        img_tok_u = pred_u[mask_u == 1.0]
                    
                    def unpatchify(img_tokens):
                        if img_tokens.dim() == 3: img_tokens = img_tokens.view(-1, img_tokens.shape[-1])
                        fold_input = img_tokens.transpose(0, 1).unsqueeze(0)
                        return F.fold(fold_input, output_size=(h_lat, w_lat), kernel_size=patch_size, stride=patch_size).squeeze(0)
                    
                    v_cond = unpatchify(img_tok_c)
                    v_uncond = unpatchify(img_tok_u)
                    
                    # 3. Apply Classifier-Free Guidance Math
                    v_pred = v_uncond + cfg * (v_cond - v_uncond)
                
                latents_gen = latents_gen + v_pred * dt
                
                # Preview logic
                if step % 5 == 0 or step == steps - 1:
                    preview_rgb = self.vae.decode(latents_gen.unsqueeze(0).to(torch.float32))[0]
                    img_arr = preview_rgb.to(torch.float32).permute(1, 2, 0).cpu().numpy()
                    img_arr = np.clip((img_arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
                    yield {"type": "image_preview", "image": Image.fromarray(img_arr), "step": step+1, "total_steps": steps}
        
        # Final Decode
        with torch.no_grad():
            rgb = self.vae.decode(latents_gen.unsqueeze(0).to(torch.float32))[0]
        
        img = rgb.permute(1, 2, 0).float().cpu().numpy()
        img = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
        
        final_img = Image.fromarray(img)
        final_img.save(output_path)
        yield {"type": "final_image", "image": final_img, "path": output_path}
                
    def generate_text_completion(self, prompt, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9, min_p=0.0, repetition_penalty=1.0):
        """
        Stream Text generation using a generator.
        """
        logger.info(f" Completing Text | Prompt: '{prompt}'")
        
        prompt_ids = torch.tensor(self.tokenizer.encode(prompt, add_pad=False, add_eot=False), device=self.device, dtype=torch.long).unsqueeze(0)
        
        generated = []
        curr_ids = prompt_ids
        
        import time
        start_time = time.time()
        
        # [NEW] Real-time Streaming
        yield {"type": "metrics", "text": "Starting generation..."}
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                txt_input = [curr_ids[0]]
                t_batch = torch.full((1,), 1.0, device=self.device, dtype=torch.float32)
                
                with torch.amp.autocast('cuda', dtype=self.dtype):
                     self.model.set_allow_cross_attention(True)
                     res = self.model(txt_input, images=None, timesteps=t_batch, causal_text=True)
                     self.model.set_allow_cross_attention(False)
                
                logits = res["text"]
                valid_len = curr_ids.shape[1]
                
                if logits.dim() == 2:
                    next_token_logits = logits[valid_len - 1, :].unsqueeze(0) # [1, V]
                else:
                    next_token_logits = logits[0, valid_len - 1, :].unsqueeze(0)

                next_token = self.model._sample_next_token(
                    next_token_logits, 
                    temperature=temperature, 
                    top_k=top_k, 
                    top_p=top_p, 
                    min_p=min_p, 
                    repetition_penalty=repetition_penalty, 
                    generated_tokens=curr_ids
                ).item()
                
                if next_token == self.tokenizer.eot_token or next_token == 100295:  # <|im_end|>
                    break
                
                generated.append(next_token)
                curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
                
                # Instantly yield the decoded chunk using robust token decoding
                text_chunk = self.tokenizer.decode([next_token])
                yield {"type": "token", "text": text_chunk, "token_id": next_token}
                
                # Progress Metrics Update every 20 tokens
                if step % 20 == 0 and step > 0:
                    elapsed = time.time() - start_time
                    tok_per_sec = len(generated) / elapsed
                    yield {"type": "metrics", "text": f"... [{step}/{max_new_tokens}] {tok_per_sec:.1f} it/s ..."}
        
        elapsed_total = time.time() - start_time
        tok_per_sec = len(generated) / elapsed_total if elapsed_total > 0 else 0
        
        decoded_text = self.tokenizer.decode(generated)
        logger.info(f"Generated {len(generated)} tokens in {elapsed_total:.2f}s ({tok_per_sec:.1f} tok/s)")
        
        yield {"type": "final_text", "text": decoded_text, "metrics": f"{tok_per_sec:.1f} tok/s"}

    def generate_multimodal_with_images(self, prompt: str, image_paths: list = None, 
                                         max_new_tokens=256, temperature=0.7, top_k=50, top_p=0.9, min_p=0.0, repetition_penalty=1.0):
        """
        Stream text completion with multiple images in context via generator.
        """
        from data_manager import IMAGE_TOKEN
        from omni_model_v2 import KVCache
        
        num_images = len(image_paths) if image_paths else 0
        logger.info(f"Multi-Image Inference | Images: {num_images} | Prompt: '{prompt[:50]}...'")
        
        existing_placeholders = prompt.count("<image>")
        
        if num_images > 0 and existing_placeholders == 0:
            image_prefix = " ".join(["<image>"] * num_images) + "\n"
            if "<|im_start|>user" in prompt:
                prompt = prompt.replace("<|im_start|>user\n", f"<|im_start|>user\n{image_prefix}")
            else:
                prompt = image_prefix + prompt
        
        elif existing_placeholders < num_images:
            extra_needed = num_images - existing_placeholders
            extra = " ".join(["<image>"] * extra_needed)
            if "<|im_start|>user\n" in prompt:
                parts = prompt.split("<|im_start|>user\n", 1)
                prompt = parts[0] + "<|im_start|>user\n" + extra + " " + parts[1]
            else:
                prompt = extra + " " + prompt
        
        # ===== EFFICIENT IMAGE LOADING =====
        yield {"type": "metrics", "text": "Loading and processing images..."}
        image_tensors = []
        
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    pil_img = Image.open(img_path).convert("RGB")
                    pil_img = pil_img.resize((128, 128), Image.Resampling.LANCZOS)
                    
                    img_tensor = torch.from_numpy(np.array(pil_img)).float()
                    img_tensor = img_tensor.permute(2, 0, 1) / 127.5 - 1.0
                    img_tensor = img_tensor.unsqueeze(0).to(self.device, dtype=torch.float32)
                    
                    with torch.no_grad():
                        latent = self.vae.encode(img_tensor)
                    
                    image_tensors.append(latent.squeeze(0).to(self.dtype))
        
        prompt_ids = torch.tensor(self.tokenizer.encode(prompt, add_pad=False, add_eot=False), device=self.device, dtype=torch.long)
        
        image_positions = []
        token_list = prompt_ids.tolist()
        for pos, token_id in enumerate(token_list):
            if token_id == IMAGE_TOKEN:
                image_positions.append(pos)
        
        if len(image_positions) != len(image_tensors):
            while len(image_tensors) < len(image_positions):
                image_tensors.append(None)
            image_tensors = image_tensors[:len(image_positions)]
        
        # ===== GENERATION LOOP STREAMING (KV-CACHE PREFILL + DECODE) =====
        # This avoids O(N^2) recomputation of the entire context for every generated token.
        generated = []
        curr_ids = prompt_ids.unsqueeze(0)  # [1, L_prompt]

        import time
        start_time = time.time()
        yield {"type": "metrics", "text": "Starting multi-modal generation (KV-cache)..."}

        # Prefill: run the full prompt once to populate the KV cache.
        kv_cache = KVCache.empty(len(self.model.blocks))
        t_batch = torch.full((1,), 1.0, device=self.device, dtype=torch.float32)

        if image_tensors:
            img_input = [image_tensors]
            img_pos_input = [image_positions]
        else:
            img_input = None
            img_pos_input = None

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=self.dtype):
            self.model.set_allow_cross_attention(True)
            res = self.model(
                [prompt_ids],
                images=img_input,
                timesteps=t_batch,
                causal_text=True,
                kv_cache=kv_cache,
                image_positions=img_pos_input,
            )
            self.model.set_allow_cross_attention(False)

        # Track the next RoPE position for text tokens, mirroring OmniFusionV2.generate_multimodal.
        kv_cache.seq_len = int(prompt_ids.shape[0])

        # Use the last TEXT token in the packed sequence for next-token sampling.
        logits = res["text"]
        mod_mask = res.get("modality_mask", None)
        if mod_mask is not None and logits.dim() == 2:
            is_text = (mod_mask == 0.0)
            text_idx = torch.nonzero(is_text, as_tuple=False).squeeze(-1)
            last_text_idx = int(text_idx[-1].item()) if text_idx.numel() > 0 else (logits.shape[0] - 1)
            next_token_logits = logits[last_text_idx, :].unsqueeze(0)  # [1, V]
        else:
            # Fallback: last position in logits
            if logits.dim() == 2:
                next_token_logits = logits[-1:, :]
            else:
                next_token_logits = logits[0, -1:, :]

        with torch.no_grad():
            for step in range(max_new_tokens):
                next_token = self.model._sample_next_token(
                    next_token_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    generated_tokens=curr_ids,
                ).item()

                if next_token == self.tokenizer.eot_token or next_token == 100259:
                    break

                generated.append(next_token)
                curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]], device=self.device)], dim=1)

                text_chunk = self.tokenizer.decode([next_token])
                yield {"type": "token", "text": text_chunk, "token_id": next_token}

                # Decode: run ONE token using KV-cache.
                new_pos = kv_cache.seq_len
                t_step = torch.full((1,), 1.0, device=self.device, dtype=torch.float32)
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    self.model.set_allow_cross_attention(True)
                    out = self.model(
                        [torch.tensor([next_token], device=self.device, dtype=torch.long)],
                        timesteps=t_step,
                        causal_text=True,
                        kv_cache=kv_cache,
                        text_pos_offset=[new_pos],
                    )
                    self.model.set_allow_cross_attention(False)

                kv_cache.seq_len += 1
                logits_step = out["text"]
                if logits_step.dim() == 2:
                    next_token_logits = logits_step[-1:, :]
                else:
                    next_token_logits = logits_step[0, -1:, :]
        
        elapsed_total = time.time() - start_time
        tok_per_sec = len(generated) / elapsed_total if elapsed_total > 0 else 0
        decoded_text = self.tokenizer.decode(generated)
        
        yield {"type": "final_text", "text": decoded_text, "metrics": f"{tok_per_sec:.1f} tok/s"}

def main():
    parser = argparse.ArgumentParser(description="OmniFusion Infinity Inference")
    
    # [FIX] Use --input-model flag instead of positional arg for consistency
    parser.add_argument("--input-model", type=str, default=r"C:\Users\chatr\Documents\Tech\VLLM\New folder\dataset_gen_checkpoints\trained_model.pt", help="Path to .pt checkpoint")
    
    parser.add_argument("--imggen", type=str, help="Prompt for image generation")
    parser.add_argument("--textgen", type=str, help="Prompt for text completion")
    parser.add_argument("--output", "-o", type=str, default="output.png", help="Output filename for image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation (Default: 42). Use 42+i for dataset reproduction.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--width", type=int, default=256, help="Image width (default 256)")
    parser.add_argument("--height", type=int, default=256, help="Image height (default 256)")
    
    # Text generation settings
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens for text generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    
    # Advanced Sampling
    parser.add_argument("--top-k", type=int, default=40, help="Top-K sampling (default 40)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P Nucleus sampling (default 0.9)")
    parser.add_argument("--min-p", type=float, default=0.05, help="Min-P sampling (default 0.05)")
    parser.add_argument("--rep-penalty", type=float, default=1.2, help="Repetition Penalty (default 1.2)")
    
    # Multi-Image Mode
    parser.add_argument("--images", type=str, nargs='+', default=[], 
                        help="Image file paths for multi-image inference (use <image> placeholders in prompt)")
    
    args = parser.parse_args()
    
    # Setup Optimizations (From test script)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    engine = InferenceModel(args.input_model)
    
    if args.imggen:
        # Set seed manually before generation
        torch.manual_seed(args.seed)
        final_path = None
        for update in engine.generate_image(args.imggen, args.output, steps=args.steps, width=args.width, height=args.height):
            if update.get("type") == "final_image":
                final_path = update.get("path")
        if final_path:
            logger.info(f"Image saved to: {final_path}")
        
    if args.textgen:
        # Text gen might also use seed?
        torch.manual_seed(args.seed)
        
        # [UPDATE] Use model's native generate_multimodal for advanced sampling
        logger.info(f" Completing Text | Prompt: '{args.textgen}' | Temp: {args.temperature} | RepPen: {args.rep_penalty}")
        
        # [FEATURE] Auto-apply ChatML format
        prompt_text = args.textgen
        if "<|im_start|>" not in prompt_text:
             prompt_text = f"<|im_start|>system\nYou are a math assistant. Solve the following problem concisely.\n<|im_end|>\n<|im_start|>user\n{prompt_text}\n<|im_end|>\n<|im_start|>assistant\n"
             logger.info("   Auto-formatted prompt to ChatML.")
        
        # Check if multi-image mode
        if args.images:
            logger.info(f"   Multi-image mode with {len(args.images)} images")
            stream = engine.generate_multimodal_with_images(
                prompt_text,
                image_paths=args.images,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                min_p=args.min_p,
                repetition_penalty=args.rep_penalty,
            )
            final_text = ""
            for update in stream:
                if update.get("type") == "final_text":
                    final_text = update.get("text", "")
            logger.info(f"Result: {final_text}")
        else:
            # Standard text-only generation
            prompt_ids = engine.tokenizer.encode(prompt_text, add_pad=False, add_eot=False).unsqueeze(0).to(engine.device)
            
            try:
                res = engine.model.generate_multimodal(
                    prompt_ids,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    min_p=args.min_p,
                    repetition_penalty=args.rep_penalty
                )
                
                # Decode result
                text_ids = res["text"]
                decoded = engine.tokenizer.decode(text_ids)
                logger.info(f"Result: {decoded}")
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")

if __name__ == "__main__":
    main()

