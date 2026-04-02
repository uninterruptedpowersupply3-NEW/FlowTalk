import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Local imports
from omni_model_v2 import OmniFusionV2, OmniConfigV2
from data_manager import TiktokenTokenizer

class ExtremeModelDiagnostic:
    def __init__(self, model_path: str = "checkpoints/omni_v2_final.pt"):
        print("Initializing Extreme Model Diagnostic Suite...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = TiktokenTokenizer()
        
        # Load Model
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # Robust config extraction
            if "config" in checkpoint:
                config_data = checkpoint["config"]
                if isinstance(config_data, dict):
                    # Filter keys to only match OmniConfigV2 fields
                    import inspect
                    valid_keys = inspect.signature(OmniConfigV2).parameters.keys()
                    filtered_config = {k: v for k, v in config_data.items() if k in valid_keys}
                    self.config = OmniConfigV2(**filtered_config)
                else:
                    self.config = config_data
            else:
                print("No config found in checkpoint. Using default OmniConfigV2.")
                self.config = OmniConfigV2()
                
            # Ensure in_channels exists (common source of errors in older checkpoints)
            if not hasattr(self.config, "in_channels"):
                self.config.in_channels = 16 # Default for Flux VAE latents
                
            self.model = OmniFusionV2(self.config)
            
            # Extract state dict (handle different naming conventions)
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model")
            if not state_dict:
                 # Check if the checkpoint is just the state dict itself
                 if any("backbone" in k for k in checkpoint.keys()):
                     state_dict = checkpoint
            
            if state_dict:
                # [FIX] Handle torch.compile "_orig_mod." prefix
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("_orig_mod."):
                        new_state_dict[k.replace("_orig_mod.", "")] = v
                    else:
                        new_state_dict[k] = v
                
                # Check for config mismatch (e.g. layers)
                checkpoint_layers = 0
                for k in new_state_dict.keys():
                    if k.startswith("blocks.") and k.endswith(".attn.q_proj.weight"):
                        checkpoint_layers += 1
                
                if checkpoint_layers > 0 and checkpoint_layers != self.config.n_layers:
                    print(f"⚠️ Config/Checkpoint Mismatch: Config has {self.config.n_layers} layers, Checkpoint has {checkpoint_layers}. Updating config.")
                    self.config.n_layers = checkpoint_layers
                    self.model = OmniFusionV2(self.config)
                    self.model.to(self.device)

                self.model.load_state_dict(new_state_dict)
            else:
                raise KeyError("Could not find model state dict in checkpoint.")
                
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            import traceback
            print(f"Error loading model: {e}")
            traceback.print_exc()
            self.model = None

    def check_nans_and_infs(self) -> Dict[str, bool]:
        """Checks every parameter for NaN or Inf values."""
        print("\n[Diagnostic] Scanning weights for NaNs and Infs...")
        results = {"has_nan": False, "has_inf": False, "bad_params": []}
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                results["has_nan"] = True
                results["bad_params"].append(f"{name} (NaN)")
            if torch.isinf(param).any():
                results["has_inf"] = True
                results["bad_params"].append(f"{name} (Inf)")
        
        if not results["bad_params"]:
            print("✅ No NaNs or Infs found in model weights.")
        else:
            print(f"❌ Found issues in {len(results['bad_params'])} parameters!")
            for p in results["bad_params"]:
                print(f"   - {p}")
        return results

    def analyze_signal_power(self, input_text: str = "A professional anime portrait."):
        """Hooks into the model to monitor signal power (mean/std) per layer."""
        print(f"\n[Diagnostic] Analyzing signal power for prompt: '{input_text}'")
        stats = []

        def hook_fn(module, input, output):
            # output can be a tensor or a tuple
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            
            # Detach and move to CPU
            data = out.detach().float().cpu()
            stats.append({
                "layer": str(module.__class__.__name__),
                "mean": data.mean().item(),
                "std": data.std().item(),
                "max": data.max().item(),
                "abs_mean": data.abs().mean().item()
            })

        # Register hooks for blocks
        hooks = []
        # Use blocks instead of backbone
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            print("❌ Could not find 'blocks' attribute in model.")
            return stats

        for i, block in enumerate(blocks):
            hooks.append(block.register_forward_hook(hook_fn))

        # Run forward pass
        text_ids = self.tokenizer.encode(input_text).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.model(text_ids, images=None, timesteps=torch.zeros(1, device=self.device))

        # Cleanup hooks
        for h in hooks:
            h.remove()

        # Print report
        print(f"{'Layer Index':<12} | {'Mean':<10} | {'Std':<10} | {'Max':<10}")
        print("-" * 50)
        for i, s in enumerate(stats):
            print(f"Block {i:<6} | {s['mean']:<10.4f} | {s['std']:<10.4f} | {s['max']:<10.4f}")
            if s['std'] < 1e-4:
                print(f"   ⚠️ WARNING: Signal Collapse detected in Block {i} (Std is near zero!)")
            if s['std'] > 10.0:
                print(f"   ⚠️ WARNING: Signal Explosion detected in Block {i} (Std is very high!)")

        return stats

    def diagnose_rope_frequencies(self):
        """Analyzes RoPE frequencies to ensure they are rotating correctly."""
        print("\n[Diagnostic] Analyzing RoPE Frequency Spectrum...")
        # Access the RoPE module (AxialRoPE3D)
        rope = None
        for m in self.model.modules():
            if "AxialRoPE3D" in str(m.__class__.__name__):
                rope = m
                break
        
        if rope is None:
            print("❌ Could not locate AxialRoPE3D module.")
            return

        # inv_freq is usually [dim/2]
        inv_freq = rope.inv_freq.detach().cpu()
        dim = inv_freq.shape[0] * 2
        base = rope.base if hasattr(rope, 'base') else 10000.0
        
        # Calculate wavelengths
        wavelengths = 2 * np.pi / inv_freq.numpy()
        
        print(f"   RoPE Dim: {dim}")
        print(f"   Base: {base}")
        print(f"   Max Wavelength: {wavelengths[-1]:.2f} tokens")
        print(f"   Min Wavelength: {wavelengths[0]:.2f} tokens")

        # Visualization
        plt.figure(figsize=(10, 5))
        plt.plot(wavelengths, marker='o')
        plt.axhline(y=512, color='r', linestyle='--', label='Standard Context (512)')
        plt.title("RoPE Wavelength Spectrum")
        plt.xlabel("Frequency Index")
        plt.ylabel("Wavelength (Tokens)")
        plt.legend()
        plt.grid(True)
        plt.savefig("rope_freqs_diagnostic.png")
        print("✅ Saved RoPE frequency diagnostic to 'rope_freqs_diagnostic.png'")

    def run_all(self):
        if self.model is None:
            print("Diagnostic aborted: Model not loaded.")
            return

        report = {}
        report["weights"] = self.check_nans_and_infs()
        report["signal"] = self.analyze_signal_power()
        self.diagnose_rope_frequencies()
        
        # Save JSON report
        with open("extreme_model_test_suite_report.json", "w") as f:
            json.dump(report, f, indent=4)
        print("\n[Diagnostic Complete] Full report saved to 'extreme_model_test_suite_report.json'")

if __name__ == "__main__":
    import sys
    checkpoint_path = None
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        # Search priority
        search_dirs = ["checkpoints", "dataset_gen_checkpoints"]
        found = []
        for d in search_dirs:
            if os.path.exists(d):
                found.extend(glob.glob(f"{d}/*.pt"))
        
        if found:
            # Sort by time, get newest
            found.sort(key=os.path.getmtime, reverse=True)
            checkpoint_path = found[0]
            print(f"Auto-selected newest checkpoint: {checkpoint_path}")
        else:
            print("No checkpoints found. Please provide path: python extreme_model_test_suite.py <path>")
            exit()

    suite = ExtremeModelDiagnostic(checkpoint_path)
    suite.run_all()
