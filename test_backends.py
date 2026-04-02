"""
Backend Integration Test Script
================================
Tests that all backend files work correctly with the trained model.
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

results = {}

def run_test(name, test_func):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)
    try:
        test_func()
        results[name] = "PASSED"
        print(f"PASSED: {name}")
    except Exception as e:
        results[name] = f"FAILED: {e}"
        print(f"FAILED: {name}")
        traceback.print_exc()

def test_data_manager():
    from data_manager import DataConfig, TiktokenTokenizer, CaptionExtractor, RobustCaptionSelector
    
    print("  Testing DataConfig...")
    config = DataConfig()
    assert config.max_image_size == 512
    print("    DataConfig OK")
    
    print("  Testing TiktokenTokenizer...")
    tokenizer = TiktokenTokenizer(max_length=64)
    tokens = tokenizer.encode("Hello, this is a test!")
    assert len(tokens) == 64
    decoded = tokenizer.decode(tokens)
    assert "Hello" in decoded
    print(f"    Tokenizer OK: encoded/decoded successfully")
    
    print("  Testing CaptionExtractor...")
    test_data = {"florence": {"more_detailed_caption": "A sunset"}, "blip": {"caption": "sunset"}}
    caption = CaptionExtractor.extract(test_data)
    assert caption == "A sunset"
    print(f"    CaptionExtractor OK")
    
    print("  Testing RobustCaptionSelector...")
    selector = RobustCaptionSelector()
    caption = selector.select(test_data)
    assert caption is not None
    print(f"    RobustCaptionSelector OK")

def test_inference_backend():
    from inference_backend import InferenceEngine, SamplingConfig, ImageGenConfig, get_tokenizer
    
    MODEL_PATH = r"C:\Users\chatr\Documents\Tech\VLLM\New folder\dataset_gen_checkpoints\trained_model.pt"
    
    print("  Testing tokenizer functions...")
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode("Test prompt", max_len=32)
    assert tokens is not None
    print("    Tokenizer OK")
    
    print("  Testing configs...")
    sampling = SamplingConfig()
    assert sampling.temperature == 0.85
    img_config = ImageGenConfig()
    assert img_config.num_steps == 20
    print("    Configs OK")
    
    print(f"  Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    engine = InferenceEngine(model_path=MODEL_PATH, load_vae=True)
    assert engine.model is not None
    print(f"    Model loaded: d_model={engine.omni_config.d_model}, layers={engine.omni_config.n_layers}")

def test_training_backend():
    from training_backend import TrainingConfig, CosineWarmupScheduler, AMPManager, create_model
    import torch
    
    print("  Testing TrainingConfig...")
    config = TrainingConfig()
    assert config.batch_size == 1
    print(f"    TrainingConfig OK")
    
    print("  Testing to_omni_config...")
    omni_config = config.to_omni_config()
    assert omni_config.d_model == config.d_model
    print(f"    OmniConfig OK")
    
    print("  Testing CosineWarmupScheduler...")
    dummy_param = torch.nn.Parameter(torch.zeros(10))
    optimizer = torch.optim.AdamW([dummy_param], lr=1e-4)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10, total_steps=100)
    scheduler.step()
    assert scheduler.current_step == 1
    print(f"    Scheduler OK")
    
    print("  Testing create_model...")
    config.d_model = 128
    config.n_layers = 2
    config.n_heads = 4
    model = create_model(config)
    assert model is not None
    print(f"    Model created OK")

def test_backend():
    from backend import TrainingConfig as GUITrainingConfig
    
    print("  Testing GUITrainingConfig...")
    config = GUITrainingConfig()
    assert config.epochs == 900
    assert config.learning_rate == 2e-4
    print(f"    GUITrainingConfig OK: epochs={config.epochs}, lr={config.learning_rate}")

def test_model_inference():
    import torch
    from omni_model_v2 import OmniFusionV2, OmniConfigV2
    from data_manager import TiktokenTokenizer
    
    MODEL_PATH = r"C:\Users\chatr\Documents\Tech\VLLM\New folder\dataset_gen_checkpoints\trained_model.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    print(f"  Device: {DEVICE}, Dtype: {DTYPE}")
    print(f"  Loading checkpoint...")
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
    
    # Detect config
    d_model, n_layers, n_heads = 384, 6, 6
    if "text_embed.weight" in state_dict:
        vocab_size, d_model = state_dict["text_embed.weight"].shape
        print(f"    Detected: vocab={vocab_size}, d_model={d_model}")
    
    for key in state_dict.keys():
        if key.startswith("blocks."):
            try:
                idx = int(key.split(".")[1])
                n_layers = max(n_layers, idx + 1)
            except: pass
    
    if "blocks.0.attn.q_proj.weight" in state_dict:
        n_heads = state_dict["blocks.0.attn.q_proj.weight"].shape[0] // 64
    
    print(f"    Config: d_model={d_model}, layers={n_layers}, heads={n_heads}")
    
    config = OmniConfigV2(d_model=d_model, n_layers=n_layers, n_heads=n_heads, vocab_size=100352, patch_size=2, in_channels=16)
    model = OmniFusionV2(config).to(DEVICE)
    if DTYPE == torch.bfloat16:
        model = model.bfloat16()
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("    Model loaded")
    
    # Test forward pass
    print("  Testing forward pass...")
    tokenizer = TiktokenTokenizer(max_length=32)
    tokens = tokenizer.encode("test prompt").to(DEVICE)
    dummy_latent = torch.randn(16, 16, 16, device=DEVICE, dtype=DTYPE)
    t = torch.tensor([0.5], device=DEVICE, dtype=DTYPE)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=DTYPE):
            output = model([tokens], [dummy_latent], t)
    
    assert "image" in output and "text" in output
    print(f"    Forward pass OK: image={output['image'].shape}, text={output['text'].shape}")

def main():
    print("\n" + "="*60)
    print("BACKEND INTEGRATION TESTS")
    print("="*60)
    
    run_test("1. data_manager.py", test_data_manager)
    run_test("2. inference_backend.py", test_inference_backend)
    run_test("3. training_backend.py", test_training_backend)
    run_test("4. backend.py", test_backend)
    run_test("5. Model Inference", test_model_inference)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed, failed = 0, 0
    for name, result in results.items():
        status = "PASSED" if "PASSED" in result else "FAILED"
        print(f"  {name}: {status}")
        if "PASSED" in result: passed += 1
        else: failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
