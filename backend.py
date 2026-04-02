"""
backend.py - OmniFusion-X V2 GUI Backend
========================================
Wrapper around test_dataset_generalization.py to provide GUI hooks,
callbacks, and configuration management while leveraging the
optimized training loop and Z-Turbo fixes.
"""
import logging
import os
import torch
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field

# Reuse logic from test_dataset_generalization (Single Source of Truth)
from test_dataset_generalization import (
    TestConfig,
    DatasetGeneralizationTest,
    DEVICE,
    DTYPE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backend")

TrainingCallback = Callable[[Dict[str, Any]], None]

@dataclass
class TrainingConfig(TestConfig):
    """Extends TestConfig with GUI specific fields."""
    data_paths: List[str] = field(default_factory=lambda: ["Train_Img"])
    # Note: vocab_size is hardcoded to 100352 in DatasetGeneralizationTest for consistency
    vocab_size: int = 100352 
    
    # GUI specific fields (not in TestConfig)
    min_image_size: int = 64
    optimizer_type: str = "adamw8bit"  # Mapped to use_8bit_adam in logic
    save_every_n_epochs: int = 10
    checkpoint_dir: str = "checkpoints"
    
    # =========================================================================
    # DEFAULTS FROM USER'S WORKING run_training_v2.bat
    # =========================================================================
    learning_rate: float = 2e-4
    epochs: int = 900
    warmup_steps: int = 100
    lambda_img: float = 5.0
    alpha_ntp: float = 0.01
    ema_update_every: int = 50
    
    # Text Training Support
    text_data_dirs: str = ""  # Comma-separated list of text directories
    alpha_ntp_text_only: float = 1.0  # Full weight for text-only samples
    
    # Disable torch.compile for GUI to prevent Windows/Inductor freeze
    compile_model: bool = False

class GuITrainer(DatasetGeneralizationTest):
    """
    GUI-compatible trainer.
    Inherits the valid Z-Turbo training loop and adds stop-flag handling via callback injection.
    """
    def __init__(self, config: TrainingConfig, callback: TrainingCallback = None):
        # FIX: Explicit, safe handling for data paths
        if config.data_paths and len(config.data_paths) > 0:
            data_dir = config.data_paths[0]
            if len(config.data_paths) > 1:
                logger.warning("Multiple data paths provided, but currently only the first one is used.")
        else:
            data_dir = "Train_Img"
            logger.warning("No data path provided, defaulting to 'Train_Img'.")

        super().__init__(config, data_dir=data_dir, callback=callback)

def train_model(paths: List[str], config_dict: Dict, callback: TrainingCallback = None, stop_flag: Dict = None):
    """
    Main entry point for GUI training.
    """
    logger.info("Initializing Training Backend...")
    
    # 1. Filter config_dict to match TrainingConfig fields
    valid_keys = set(TrainingConfig.__dataclass_fields__.keys())
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    
    # 2. Create Configuration
    try:
        config = TrainingConfig(**filtered_config)
    except Exception as e:
        logger.error(f"Configuration Error: {e}")
        if callback:
            callback({"status": "error", "message": f"Config Error: {e}"})
        return {}
    
    config.data_paths = paths
    
    # Map GUI fields to TestConfig fields
    if hasattr(config, "optimizer_type"):
        config.use_8bit_adam = (config.optimizer_type == "adamw8bit")
    
    # Windows/CUDA DataLoader compatibility
    if os.name == 'nt' and config.num_workers > 0:
        logger.warning(f"Windows detected with num_workers={config.num_workers}. If training freezes, set workers to 0.")
    
    # 3. Create Wrapped Callback for Interruption
    def wrapped_callback(data):
        if stop_flag and stop_flag.get("stop", False):
            raise InterruptedError("Training stopped by user")
        if callback:
            callback(data)

    # 4. Initialize and Run Trainer
    try:
        trainer = GuITrainer(config, callback=wrapped_callback)
        trainer.run()
        return getattr(trainer, "results", {})
        
    except InterruptedError:
        logger.info("Training interrupted by user.")
        if callback:
            callback({"status": "stopped", "message": "Training stopped by user."})
        return getattr(trainer, "results", {}) if 'trainer' in locals() else {}
    except Exception as e:
        logger.error(f"Training Failed: {e}", exc_info=True)
        if callback:
            callback({"status": "error", "message": str(e)})
        raise e

if __name__ == "__main__":
    print("Running in DIAGNOSTIC MODE (Direct Execution)...")
    try:
        from test_model_diagnostics import run_extensive_diagnostics
        run_extensive_diagnostics()
    except ImportError:
        print("Error: Could not import test_model_diagnostics.py. Please check the file exists.")
    except Exception as e:
        print(f"Diagnostic Run Failed: {e}")
