#!/usr/bin/env python
"""
Synthetic Training API
======================
An API script that:
1. Generates synthetic training data (shapes with holdouts, math problems)
2. Calls backend training scripts with context packing enabled
3. Evaluates generalization: math, novel composition, captioning

Usage:
    python synthetic_training_api.py --run-all
    python synthetic_training_api.py --generate-data
    python synthetic_training_api.py --train
    python synthetic_training_api.py --evaluate
"""

import os
import sys
import json
import random
import subprocess
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Image generation
try:
    from PIL import Image, ImageDraw
except ImportError:
    print("PIL not found. Install with: pip install Pillow")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SyntheticAPI")

# =============================================================================
# Configuration
# =============================================================================

class SyntheticConfig:
    """Configuration for synthetic data generation and training."""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    SYNTHETIC_DATA_DIR = BASE_DIR / "synthetic_data"
    IMAGE_DIR = SYNTHETIC_DATA_DIR / "images"
    MATH_DIR = SYNTHETIC_DATA_DIR / "math"
    OUTPUT_DIR = SYNTHETIC_DATA_DIR / "outputs"
    CHECKPOINT_DIR = BASE_DIR / "dataset_gen_checkpoints"
    
    # Image generation
    IMAGE_SIZE = 256
    NUM_IMAGES_PER_COMBO = 5  # Number of images per shape+color combo
    
    # Colors and shapes for training
    COLORS = {
        "red": (255, 60, 60),
        "blue": (60, 100, 255),
        "green": (60, 200, 60),
        "yellow": (255, 255, 60),
        "orange": (255, 160, 60),
        "purple": (180, 60, 255),
        "white": (255, 255, 255),
    }
    
    # Training set combinations (holdout: red circle, yellow triangle)
    TRAINING_COMBOS = [
        # Circles (NO RED)
        ("circle", "blue"), ("circle", "green"), ("circle", "yellow"),
        ("circle", "orange"), ("circle", "purple"),
        # Squares (ALL)
        ("square", "red"), ("square", "blue"), ("square", "green"),
        ("square", "yellow"), ("square", "orange"),
        # Triangles (NO YELLOW)
        ("triangle", "red"), ("triangle", "blue"), ("triangle", "green"),
        ("triangle", "orange"), ("triangle", "purple"),
        # Stars
        ("star", "red"), ("star", "yellow"), ("star", "purple"), ("star", "white"),
    ]
    
    # Holdout combos (never trained on - for testing generalization)
    HOLDOUT_COMBOS = [
        ("circle", "red"),      # Red circle
        ("triangle", "yellow"), # Yellow triangle
    ]
    
    # Math dataset
    NUM_MATH_SAMPLES = 500
    
    # Training command template
    TRAINING_SCRIPT = "test_dataset_generalization.py"
    PYTHON_EXE = sys.executable


# =============================================================================
# Image Generation
# =============================================================================

class ShapeGenerator:
    """Generate shape images with captions."""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.size = config.IMAGE_SIZE
        
    def draw_circle(self, draw: ImageDraw, color: Tuple[int, int, int], size_factor: float = 0.6):
        """Draw a circle in the center."""
        margin = int(self.size * (1 - size_factor) / 2)
        draw.ellipse([margin, margin, self.size - margin, self.size - margin], fill=color)
        
    def draw_square(self, draw: ImageDraw, color: Tuple[int, int, int], size_factor: float = 0.6):
        """Draw a square in the center."""
        margin = int(self.size * (1 - size_factor) / 2)
        draw.rectangle([margin, margin, self.size - margin, self.size - margin], fill=color)
        
    def draw_triangle(self, draw: ImageDraw, color: Tuple[int, int, int], size_factor: float = 0.6):
        """Draw a triangle in the center."""
        margin = int(self.size * (1 - size_factor) / 2)
        points = [
            (self.size // 2, margin),  # Top
            (margin, self.size - margin),  # Bottom left
            (self.size - margin, self.size - margin),  # Bottom right
        ]
        draw.polygon(points, fill=color)
        
    def draw_star(self, draw: ImageDraw, color: Tuple[int, int, int], size_factor: float = 0.6):
        """Draw a 5-pointed star in the center."""
        import math
        cx, cy = self.size // 2, self.size // 2
        outer_r = int(self.size * size_factor / 2)
        inner_r = outer_r // 2
        
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            r = outer_r if i % 2 == 0 else inner_r
            x = cx + int(r * math.cos(angle))
            y = cy - int(r * math.sin(angle))
            points.append((x, y))
        draw.polygon(points, fill=color)
        
    def generate_image(self, shape: str, color_name: str, variation: int = 0) -> Image.Image:
        """Generate an image with the specified shape and color."""
        # Dark background with slight variation
        bg_val = 20 + random.randint(-5, 5)
        img = Image.new('RGB', (self.size, self.size), (bg_val, bg_val, bg_val))
        draw = ImageDraw.Draw(img)
        
        color = self.config.COLORS[color_name]
        # Add slight color variation
        color = tuple(max(0, min(255, c + random.randint(-10, 10))) for c in color)
        
        # Size variation
        size_factor = 0.5 + random.uniform(-0.1, 0.15)
        
        if shape == "circle":
            self.draw_circle(draw, color, size_factor)
        elif shape == "square":
            self.draw_square(draw, color, size_factor)
        elif shape == "triangle":
            self.draw_triangle(draw, color, size_factor)
        elif shape == "star":
            self.draw_star(draw, color, size_factor)
            
        return img
    
    def generate_caption(self, shape: str, color: str) -> str:
        """Generate a ChatML caption for the image."""
        templates = [
            f"A {color} {shape} on a dark background.",
            f"This image shows a {color} {shape}.",
            f"A {color} colored {shape}.",
        ]
        return random.choice(templates)


def generate_shape_dataset(config: SyntheticConfig, combos: List[Tuple[str, str]], is_holdout: bool = False):
    """Generate shape images with captions."""
    generator = ShapeGenerator(config)
    
    output_dir = config.IMAGE_DIR / ("holdout" if is_holdout else "train")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for shape, color in combos:
        for i in range(config.NUM_IMAGES_PER_COMBO):
            img = generator.generate_image(shape, color, variation=i)
            caption = generator.generate_caption(shape, color)
            
            # Save image
            filename = f"{shape}_{color}_{i:02d}"
            img_path = output_dir / f"{filename}.png"
            txt_path = output_dir / f"{filename}.txt"
            
            img.save(img_path)
            
            # Caption in ChatML format
            chatml_caption = f"""<|im_start|>user
Describe this image.<|im_end|>
<|im_start|>assistant
{caption}<|im_end|>"""
            txt_path.write_text(chatml_caption)
            count += 1
            
    logger.info(f"Generated {count} {'holdout' if is_holdout else 'training'} images in {output_dir}")
    return output_dir


# =============================================================================
# Math Data Generation
# =============================================================================

def generate_math_dataset(config: SyntheticConfig):
    """Generate simple math problems in ChatML format."""
    config.MATH_DIR.mkdir(parents=True, exist_ok=True)
    
    problems = []
    
    for _ in range(config.NUM_MATH_SAMPLES):
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        op = random.choice(["+", "-", "*"])
        
        if op == "+":
            answer = a + b
        elif op == "-":
            answer = a - b
        else:  # *
            a = random.randint(1, 12)  # Keep multiplication simpler
            b = random.randint(1, 12)
            answer = a * b
            
        question = f"What is {a} {op} {b}?"
        
        chatml = f"""<|im_start|>system
You are a math assistant. Solve the problem and give only the numerical answer.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""
        
        problems.append({
            "text": chatml,
            "question": question,
            "answer": answer,
        })
    
    # Save as text files (one per problem)
    for i, prob in enumerate(problems):
        path = config.MATH_DIR / f"math_{i:04d}.txt"
        path.write_text(prob["text"])
    
    # Also save as JSON for evaluation
    eval_path = config.SYNTHETIC_DATA_DIR / "math_eval.json"
    eval_data = [{"q": p["question"], "a": p["answer"]} for p in problems[-50:]]  # Last 50 for eval
    eval_path.write_text(json.dumps(eval_data, indent=2))
    
    logger.info(f"Generated {len(problems)} math problems in {config.MATH_DIR}")
    return config.MATH_DIR


# =============================================================================
# Training Orchestration
# =============================================================================

def find_vcvars64() -> Optional[Path]:
    """Find vcvars64.bat for Visual Studio environment setup."""
    possible_paths = [
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"),
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None


def run_training(config: SyntheticConfig, epochs: int = 200, checkpoint_name: str = "synthetic_model", 
                 no_compile: bool = False, workers: int = 4):
    """Run training via subprocess with all optimizations.
    
    Args:
        config: Configuration object
        epochs: Number of training epochs
        checkpoint_name: Name for the checkpoint file
        no_compile: If True, disable torch.compile to avoid needing MSVC compiler
        workers: Number of data loading workers (0 = main thread only)
    """
    
    image_dir = config.IMAGE_DIR / "train"
    math_dir = config.MATH_DIR
    
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        logger.error("Run with --generate-data first!")
        return None
    
    # Find Visual Studio environment
    vcvars_path = find_vcvars64()
    
    if vcvars_path is None and not no_compile:
        logger.warning("=" * 60)
        logger.warning("Visual Studio vcvars64.bat not found!")
        logger.warning("torch.compile requires Visual Studio C++ compiler (cl.exe)")
        logger.warning("Please install Visual Studio 2022 with C++ tools or use --no-compile")
        logger.warning("=" * 60)
        return None
    
    # Build training command arguments (not including python executable for now)
    script_path = str(config.BASE_DIR / config.TRAINING_SCRIPT)
    
    # When context-pack is enabled, workers must be 0 (due to pickling issues)
    # But we can use workers for non-packed mode
    actual_workers = 0  # Context packing requires workers=0
    
    training_args = [
        f'"{script_path}"',
        f'--data-dir "{image_dir}"',
        f'--text-data-dirs "{math_dir}"',
        "--batch-size 4",
        "--lr 2e-4",
        "--max-size 256",
        "--grad-accum-steps 2",
        "--d-model 512",
        "--n-heads 8",
        "--n-layers 8",
        f"--output-name {checkpoint_name}",
        "--save-every 500",
        f"--epochs {epochs}",
        "--max-steps 0",
        "--lambda-img 5.0",
        "--alpha-ntp 0.01",
        "--alpha-ntp-text-only 1.0",
        f"--workers {actual_workers}",
        "--graph-batch-size 4",
        "--ema-every 50",
        "--prefetch-factor 2",
        "--log-every 10",
        # NOTE: --use-min-snr removed - conflicts with logit-normal sampling per SD3 paper
        # NOTE: --parallel-encode removed - not useful with workers=0
        "--multi-image",
        "--context-pack",
        "--max-context-length 16384",
        "--image-ratio 0.5",
        # NOTE: NOT using --freeze-text since we want multimodal training
    ]
    
    if no_compile:
        training_args.append("--no-compile")
    
    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)
    
    # Build arguments string
    args_str = " ".join(training_args)
    python_exe = f'"{config.PYTHON_EXE}"'
    full_cmd = f'{python_exe} {args_str}'
    
    try:
        if vcvars_path and not no_compile:
            # Run training through cmd.exe with vcvars64.bat to set up compiler environment
            logger.info(f"Visual Studio environment: {vcvars_path}")
            logger.info(f"Training command: python {args_str[:100]}...")
            
            # Create a batch script to run
            batch_script = config.BASE_DIR / "_run_training.bat"
            batch_content = f'''@echo off
echo Setting up Visual Studio environment...
call "{vcvars_path}"
if errorlevel 1 (
    echo Failed to set up Visual Studio environment
    exit /b 1
)
echo Starting training...
{full_cmd}
'''
            batch_script.write_text(batch_content)
            logger.info(f"Created batch script: {batch_script}")
            
            # Run the batch script
            result = subprocess.run(
                ['cmd', '/c', str(batch_script)],
                cwd=str(config.BASE_DIR),
            )
            
            # Clean up batch script
            try:
                batch_script.unlink()
            except:
                pass
        else:
            # Run directly without vcvars (--no-compile mode)
            logger.info(f"Running with --no-compile (no Visual Studio required)")
            logger.info(f"Command: python {args_str[:100]}...")
            
            cmd_list = [config.PYTHON_EXE, script_path] + [
                arg for pair in [
                    ["--data-dir", str(image_dir)],
                    ["--text-data-dirs", str(math_dir)],
                    ["--batch-size", "4"],
                    ["--lr", "2e-4"],
                    ["--max-size", "256"],
                    ["--grad-accum-steps", "2"],
                    ["--d-model", "512"],
                    ["--n-heads", "8"],
                    ["--n-layers", "8"],
                    ["--output-name", checkpoint_name],
                    ["--save-every", "500"],
                    ["--epochs", str(epochs)],
                    ["--max-steps", "0"],
                    ["--lambda-img", "5.0"],
                    ["--alpha-ntp", "0.01"],
                    ["--alpha-ntp-text-only", "1.0"],
                    ["--workers", str(actual_workers)],
                    ["--graph-batch-size", "4"],
                    ["--ema-every", "50"],
                    ["--prefetch-factor", "2"],
                    ["--log-every", "10"],
                ] for arg in pair
            ] + [
                # NOTE: Removed --use-min-snr and --parallel-encode (see vcvars path)
                "--multi-image",
                "--context-pack",
                "--max-context-length", "16384",
                "--image-ratio", "0.5",
                # NOTE: NOT using --freeze-text since we want multimodal training
                "--no-compile",
            ]
            
            result = subprocess.run(cmd_list, cwd=str(config.BASE_DIR))
            
        if result.returncode == 0:
            logger.info("Training completed successfully!")
            return config.CHECKPOINT_DIR / f"{checkpoint_name}.pt"
        else:
            logger.error(f"Training failed with return code {result.returncode}")
            return None
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_math(config: SyntheticConfig, checkpoint_path: Path, num_tests: int = 20):
    """Evaluate math problem solving."""
    logger.info("\n" + "=" * 60)
    logger.info("[EVAL] Math Problem Solving")
    logger.info("=" * 60)
    
    # Load evaluation data
    eval_path = config.SYNTHETIC_DATA_DIR / "math_eval.json"
    if not eval_path.exists():
        logger.error("Math evaluation data not found!")
        return 0.0
        
    eval_data = json.loads(eval_path.read_text())
    test_cases = random.sample(eval_data, min(num_tests, len(eval_data)))
    
    # Import inference backend
    try:
        from inference_backend import InferenceModel
        model = InferenceModel(str(checkpoint_path))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 0.0
    
    correct = 0
    for tc in test_cases:
        prompt = f"""<|im_start|>system
You are a math assistant. Solve the problem and give only the numerical answer.<|im_end|>
<|im_start|>user
{tc['q']}<|im_end|>
<|im_start|>assistant
"""
        try:
            response = model.generate_text_completion(prompt, max_new_tokens=10, temperature=0.1)
            # Extract number from response
            predicted = ''.join(c for c in response.split()[0] if c.isdigit() or c == '-')
            if predicted and int(predicted) == tc['a']:
                correct += 1
                logger.info(f"  ✓ {tc['q']} = {tc['a']} (got: {predicted})")
            else:
                logger.info(f"  ✗ {tc['q']} = {tc['a']} (got: {response[:20]})")
        except Exception as e:
            logger.warning(f"  Error on {tc['q']}: {e}")
            
    accuracy = correct / len(test_cases) * 100
    logger.info(f"\nMath Accuracy: {correct}/{len(test_cases)} = {accuracy:.1f}%")
    return accuracy


def evaluate_novel_composition(config: SyntheticConfig, checkpoint_path: Path):
    """Generate novel compositions (red circle, yellow triangle)."""
    logger.info("\n" + "=" * 60)
    logger.info("[EVAL] Novel Composition (Generalization)")
    logger.info("=" * 60)
    
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        from inference_backend import InferenceModel
        model = InferenceModel(str(checkpoint_path))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    novel_prompts = [
        ("red circle", "A red circle on a dark background."),
        ("yellow triangle", "A yellow triangle on a dark background."),
    ]
    
    for name, prompt in novel_prompts:
        logger.info(f"\n  Generating: '{name}'")
        try:
            output_path = config.OUTPUT_DIR / f"novel_{name.replace(' ', '_')}.png"
            model.generate_image(
                prompt=prompt,
                output_path=str(output_path),
                steps=50,
                width=256,
                height=256
            )
            logger.info(f"  ✓ Saved: {output_path}")
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")


def evaluate_captioning(config: SyntheticConfig, checkpoint_path: Path):
    """Evaluate image captioning on holdout images."""
    logger.info("\n" + "=" * 60)
    logger.info("[EVAL] Image Captioning")
    logger.info("=" * 60)
    
    holdout_dir = config.IMAGE_DIR / "holdout"
    if not holdout_dir.exists():
        logger.warning("Holdout directory not found, using training images")
        holdout_dir = config.IMAGE_DIR / "train"
        
    try:
        from inference_backend import InferenceModel
        model = InferenceModel(str(checkpoint_path))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Get sample images
    images = list(holdout_dir.glob("*.png"))[:5]
    
    for img_path in images:
        logger.info(f"\n  Image: {img_path.name}")
        prompt = "Describe this image."
        try:
            response = model.generate_multimodal_with_images(
                prompt=prompt,
                image_paths=[str(img_path)],
                max_new_tokens=50,
                temperature=0.7
            )
            logger.info(f"  Caption: {response[:100]}")
        except Exception as e:
            logger.error(f"  Error: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Synthetic Training API")
    parser.add_argument("--run-all", action="store_true", help="Run full pipeline: generate -> train -> evaluate")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic data only")
    parser.add_argument("--train", action="store_true", help="Run training only")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation only")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for evaluation")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile (avoids MSVC compiler requirement)")
    args = parser.parse_args()
    
    config = SyntheticConfig()
    
    if args.generate_data or args.run_all:
        logger.info("=" * 60)
        logger.info("Generating Synthetic Data")
        logger.info("=" * 60)
        
        # Training images
        generate_shape_dataset(config, config.TRAINING_COMBOS, is_holdout=False)
        
        # Holdout images (for evaluation)
        generate_shape_dataset(config, config.HOLDOUT_COMBOS, is_holdout=True)
        
        # Math problems
        generate_math_dataset(config)
        
    checkpoint_path = None
    
    if args.train or args.run_all:
        checkpoint_path = run_training(config, epochs=args.epochs, no_compile=args.no_compile)
        
    if args.evaluate or args.run_all:
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        elif checkpoint_path is None:
            # Try to find latest checkpoint
            checkpoint_path = config.CHECKPOINT_DIR / "synthetic_model.pt"
            
        if checkpoint_path and checkpoint_path.exists():
            evaluate_math(config, checkpoint_path)
            evaluate_novel_composition(config, checkpoint_path)
            evaluate_captioning(config, checkpoint_path)
        else:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            
    if not any([args.run_all, args.generate_data, args.train, args.evaluate]):
        parser.print_help()


if __name__ == "__main__":
    main()
