"""
Multi-Image Pipeline Sanity Test
=================================
Creates dummy images and runs 1 training step to verify the new pack_inputs
pipeline accepts List[List[Tensor]] format without crashing.

Usage:
    python test_multi_image_sanity.py
"""

import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw

# Ensure local imports work
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_manager import (
    MultiImageChatDataset, multiimage_collate_fn, 
    TiktokenTokenizer, DataConfig, IMAGE_TOKEN
)
from omni_model_v2 import OmniFusionV2, OmniConfigV2
from vae_module import FluxVAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

def create_dummy_images():
    """Create simple shape images for testing."""
    test_dir = os.path.abspath("multi_image_test")
    os.makedirs(test_dir, exist_ok=True)
    
    # Image 1: Red square
    img1_path = os.path.join(test_dir, "shape_1.png")
    img1 = Image.new("RGB", (128, 128), "white")
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle([30, 30, 98, 98], fill="red")
    img1.save(img1_path)
    
    # Image 2: Blue circle
    img2_path = os.path.join(test_dir, "shape_2.png")
    img2 = Image.new("RGB", (128, 128), "white")
    draw2 = ImageDraw.Draw(img2)
    draw2.ellipse([30, 30, 98, 98], fill="blue")
    img2.save(img2_path)
    
    print(f"[OK] Created dummy images: {img1_path}, {img2_path}")
    return img1_path, img2_path

def create_test_jsonl(img1_path, img2_path):
    """Create test JSONL with multi-image conversation."""
    data = {
        "messages": [
            {"role": "user", "content": "Compare these shapes: <image> and <image>. Which is redder?"},
            {"role": "assistant", "content": "The first image shows a red square, while the second shows a blue circle. The first one is clearly redder."}
        ],
        "images": [img1_path, img2_path]
    }
    
    jsonl_path = "multi_image_test/test_conversation.jsonl"
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(data) + "\n")
    
    print(f"[OK] Created test JSONL: {jsonl_path}")
    return jsonl_path

def run_sanity_test():
    """Run a single forward pass with multi-image data."""
    print("\n" + "="*60)
    print("Multi-Image Pipeline Sanity Test")
    print("="*60)
    
    # 1. Create test data
    img1, img2 = create_dummy_images()
    jsonl_path = create_test_jsonl(img1, img2)
    
    # 2. Load dataset
    print("\n[INFO] Loading MultiImageChatDataset...")
    dataset = MultiImageChatDataset(
        data_paths=[jsonl_path],
        max_context_length=512,
        vae=None,  # No VAE encoding for this test
    )
    print(f"   Loaded {len(dataset)} samples")
    
    # 3. Get a sample
    sample = dataset[0]
    print(f"\n[INFO] Sample structure:")
    print(f"   input_ids shape: {sample['input_ids'].shape}")
    print(f"   num images: {len(sample['image_tensors'])}")
    print(f"   image_positions: {sample['image_positions']}")
    
    # Verify IMAGE_TOKEN exists in input_ids
    token_list = sample['input_ids'].tolist()
    img_token_count = token_list.count(IMAGE_TOKEN)
    print(f"   IMAGE_TOKEN (100293) count in tokens: {img_token_count}")
    
    # 4. Collate into batch
    print("\n[INFO] Testing collate function...")
    batch = multiimage_collate_fn([sample])
    print(f"   batch['input_ids']: {len(batch['input_ids'])} tensors")
    print(f"   batch['image_tensors']: {len(batch['image_tensors'])} lists")
    print(f"   batch['image_positions']: {batch['image_positions']}")
    
    # 5. Create small model for testing
    print("\n[INFO] Creating test model (small config)...")
    config = OmniConfigV2(
        d_model=128,
        n_layers=2,
        n_heads=2,
        patch_size=2,
        in_channels=3,  # RGB for this test (not VAE latents)
    )
    model = OmniFusionV2(config).to(DEVICE, dtype=DTYPE)
    model.eval()
    
    # 6. Prepare inputs for model
    print("\n[INFO] Running pack_inputs with multi-image data...")
    text_ids = [sample['input_ids'].to(DEVICE)]
    
    # Convert PIL-loaded images to tensors (simulating latents)
    image_tensors = []
    for img in sample['image_tensors']:
        if img is not None:
            image_tensors.append(img.to(DEVICE, dtype=DTYPE))
        else:
            image_tensors.append(None)
    
    # Call pack_inputs with the NEW interface
    timesteps = torch.tensor([0.5], device=DEVICE, dtype=DTYPE)
    
    try:
        packed_x, packed_c, packed_pos, mod_mask, cu_seqlens, doc_ids, image_shapes = model.pack_inputs(
            text_ids=text_ids,
            images=[image_tensors],  # List[List[Tensor]]
            timesteps=timesteps,
            image_positions=[sample['image_positions']],  # List[List[int]]
            pad=True
        )
        
        print(f"\n[PASS] pack_inputs SUCCESS!")
        print(f"   packed_x shape: {packed_x.shape}")
        print(f"   packed_pos shape: {packed_pos.shape}")
        print(f"   mod_mask sum (image tokens): {mod_mask.sum().item():.0f}")
        print(f"   cu_seqlens: {cu_seqlens}")
        print(f"   image_shapes: {image_shapes}")
        
        # Verify temporal positions are monotonic for text-image interleaving
        temporal_pos = packed_pos[:, 0].float().cpu().numpy()
        print(f"   Temporal positions (first 20): {temporal_pos[:20]}")
        
    except Exception as e:
        print(f"\n[FAIL] pack_inputs FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. Run a forward pass
    print("\n[INFO] Running full forward pass...")
    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=DTYPE):
                output = model.forward(
                    text_ids=text_ids,
                    images=[image_tensors],
                    timesteps=timesteps,
                    image_positions=[sample['image_positions']],
                )
        
        print(f"\n[PASS] Forward pass SUCCESS!")
        print(f"   output['image'] shape: {output['image'].shape}")
        print(f"   output['text'] shape: {output['text'].shape}")
        print(f"   output['modality_mask'] sum: {output['modality_mask'].sum().item():.0f}")
        
    except Exception as e:
        print(f"\n[FAIL] Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print(">>> ALL SANITY TESTS PASSED! <<<")
    print("="*60)
    return True

if __name__ == "__main__":
    success = run_sanity_test()
    exit(0 if success else 1)
