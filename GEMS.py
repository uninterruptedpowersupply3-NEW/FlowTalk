import os
import json
import random
import glob

# Config
IMAGE_DIR = "Train_Img"  # Your existing image folder
OUTPUT_FILE = "multi_image_debug.jsonl"
NUM_SAMPLES = 100  # How many fake conversations to generate

# 1. Find Images
extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
images = []
for ext in extensions:
    images.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

if len(images) < 2:
    print(f"Error: Need at least 2 images in {IMAGE_DIR} to create pairs!")
    exit()

print(f"Found {len(images)} images. Generating {NUM_SAMPLES} multi-image conversations...")

# 2. Generate Conversations
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i in range(NUM_SAMPLES):
        # Pick 2-3 random images
        num_imgs = random.randint(2, 3)
        selected_imgs = random.sample(images, num_imgs)
        
        # Create LLaVA-style format
        # We use absolute paths to be safe, or relative if your loader handles it
        img_paths = [os.path.abspath(img) for img in selected_imgs]
        
        # Build a dummy conversation
        placeholders = " ".join(["<image>"] * num_imgs)
        
        entry = {
            "messages": [
                {
                    "role": "user", 
                    "content": f"Here are {num_imgs} images: {placeholders}. Describe them."
                },
                {
                    "role": "assistant", 
                    "content": f"I see {num_imgs} distinct images here. They appear to be from the training set."
                }
            ],
            "images": img_paths
        }
        
        f.write(json.dumps(entry) + "\n")

print(f"✅ Saved {OUTPUT_FILE}. Ready for training!")