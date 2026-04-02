import os
import sys
import shutil
import torch
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_dataset_generalization import TestConfig, ImageLatentDataset, LatentCache
from encoder_backend import EncoderRuntimeConfig, FastMultimodalEncoder

def test_latent_cache():
    print("=== Testing Latent Cache Functionality ===")
    
    # 1. Setup Data Directory
    test_dir = "test_cache_data"
    cache_dir = "test_cache_output"
    
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a dummy image and text
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (128, 128), color=(255, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([32, 32, 96, 96], fill=(0, 255, 0))
    
    img_path = os.path.join(test_dir, "test_image.png")
    txt_path = os.path.join(test_dir, "test_image.txt")
    
    img.save(img_path)
    with open(txt_path, "w") as f:
        f.write("A test square image")
        
    print(f"Created dummy data at {test_dir}")
    
    # 2. Precompute Latents (Simulating what launcher_gui does)
    print("\n--- Step 1: Precomputing Latents ---")
    enc_config = EncoderRuntimeConfig(
        cache_dir=cache_dir,
        batch_size=1,
        max_size=128,
        num_workers=0, # No multiprocessing for simple test
    )
    
    encoder = FastMultimodalEncoder(enc_config)
    stats = encoder.precompute_directory(test_dir)
    print(f"Precompute stats: {stats}")
    
    # Check if index exists
    index_path = os.path.join(cache_dir, "index.json")
    if os.path.exists(index_path):
        print(f"Index created at {index_path}")
    else:
        print("ERROR: Index not created!")
        return
        
    # 3. Load using ImageLatentDataset (Simulating training startup)
    print("\n--- Step 2: Loading via Dataset ---")
    config = TestConfig(
        cache_dir=cache_dir,
        use_cache=True,
        max_image_size=128,
        patch_size=2,
        lazy_load=True, # Prevent it from caching everything in RAM upfront
        parallel_encode=False
    )
    
    dataset = ImageLatentDataset(test_dir, config)
    print(f"Dataset initialized with {len(dataset)} samples")
    print(f"Cache enabled: {dataset.cache.enabled}")
    
    # 4. Verify Cache Hit
    print("\n--- Step 3: Fetching Sample ---")
    sample = dataset[0]
    
    print(f"Sample name: {sample['name']}")
    print(f"Latents shape: {sample['latents'].shape}")
    print(f"Latents dtype: {sample['latents'].dtype}")
    print(f"Tokens shape: {sample['input_ids'].shape}")
    
    # Cleanup
    print("\nCleaning up test files...")
    shutil.rmtree(test_dir, ignore_errors=True)
    shutil.rmtree(cache_dir, ignore_errors=True)
    print("Done!")

if __name__ == "__main__":
    test_latent_cache()
