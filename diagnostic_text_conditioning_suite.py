import torch
import torch.nn as nn
from omni_model_v2 import OmniFusionV2, OmniConfigV2, FlowMatchingLoss
import os

def run_diagnostics():
    print("="*60)
    print("DIAGNOSTIC TEST SUITE: TEXT CONDITIONING COLLAPSE")
    print("="*60)
    
    # 1. Initialize Model
    config = OmniConfigV2()
    # Keep it small for testing
    config.d_model = 128
    config.n_heads = 4
    config.n_layers = 2
    config.patch_size = 16
    config.vocab_size = 100500
    config.regional_compile = False # Disable compilation for raw gradient access
    config.grad_checkpointing = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = OmniFusionV2(config).to(device)
    model.set_allow_cross_attention(True) # Fix FlexAttention failure
    model.train()
    
    print("\n[1/6] DATAFLOW & SHAPES VERIFICATION")
    # Dummy inputs
    # Batch size 2
    text_ids = [torch.randint(0, config.vocab_size, (10,), device=device), 
                torch.randint(0, config.vocab_size, (15,), device=device)]
    
    # Image lists
    images = [[torch.randn(config.in_channels, 1, 64, 64, device=device)],
              [torch.randn(config.in_channels, 1, 64, 64, device=device)]]
              
    timesteps = torch.tensor([0.5, 0.8], device=device)
    
    try:
        x, c, pos, mod_mask, cu_seqlens, doc_ids, image_shapes = model.pack_inputs(
            text_ids, images, timesteps
        )
        print("  ✓ pack_inputs succeeded")
        print(f"  - Packed X shape: {x.shape}")
        print(f"  - Packed C shape: {c.shape}")
        print(f"  - Modality Mask shape: {mod_mask.shape}")
        print(f"  - CU_Seqlens: {cu_seqlens}")
    except Exception as e:
        print(f"  ✗ pack_inputs failed: {e}")
        return

    print("\n[2/6] ADALN GATE INITIALIZATION & ACTIVATION")
    # Check if AdaLN gates are initialized correctly or if they are dead
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'adaLN'):
            gate_bias = block.adaLN.proj_up.bias
            gate_msa = gate_bias[2*config.d_model : 3*config.d_model].mean().item()
            gate_mlp = gate_bias[5*config.d_model : 6*config.d_model].mean().item()
            print(f"  - Block {i} AdaLN proj_up bias Gate_MSA mean: {gate_msa:.4f}")
            print(f"  - Block {i} AdaLN proj_up bias Gate_MLP mean: {gate_mlp:.4f}")
            
            # Check weight scale
            w_std = block.adaLN.proj_up.weight.std().item()
            print(f"  - Block {i} AdaLN proj_up weight std: {w_std:.4f}")

    print("\n[3/6] FORWARD PASS & ATTENTION INTEGRITY")
    
    # We will hook into the attention to check entropy/saturation
    attn_entropies = []
    
    def attn_hook(module, input, output):
        # Output of attention is just the tensor, we can't easily get the scores 
        # unless we modify the module. For now, let's just ensure no NaNs
        pass
        
    for block in model.blocks:
        block.attn.register_forward_hook(attn_hook)

    try:
        out = model(text_ids, images, timesteps, causal_text=False)
        print("  ✓ Forward pass succeeded")
        print(f"  - Image Head output shape: {out['image'].shape}")
        print(f"  - Text Head output shape: {out['text'].shape}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[4/6] LOSS MECHANICS & GRADIENT FLOW")
    loss_fn = FlowMatchingLoss(model)
    
    # Compute loss
    try:
        loss_dict = loss_fn(text_ids, images)
        print("  ✓ Loss computation succeeded")
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"  - {k}: {v.item():.4f}")
    except Exception as e:
        print(f"  ✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Backward pass
    total_loss = loss_dict['loss']
    total_loss.backward()
    
    print("\n[5/6] GRADIENT MAGNITUDES")
    def check_grad(name, tensor):
        if tensor.grad is not None:
            gnorm = tensor.grad.norm().item()
            print(f"  - {name} grad norm: {gnorm:.6f}")
        else:
            print(f"  - {name} grad: NONE")

    check_grad("text_embed.weight", model.text_embed.weight)
    for i, p in enumerate(model.text_pool_proj.parameters()):
        if p.requires_grad:
            check_grad(f"text_pool_proj[{i}]", p)
            
    check_grad("time_embed.mlp[0].weight", model.time_embed.mlp[0].weight)
    check_grad("image_head.weight", model.image_head.weight)
    check_grad("text_head.weight", model.text_head.weight)

    for i, block in enumerate(model.blocks):
        if hasattr(block, 'adaLN'):
            check_grad(f"block[{i}].adaLN.proj_down.weight", block.adaLN.proj_down.weight)
            check_grad(f"block[{i}].adaLN.proj_up.weight", block.adaLN.proj_up.weight)

    print("\n[6/6] WEIGHT DECAY VULNERABILITY ANALYSIS")
    print("  Checking parameter grouping logic for decay...")
    
    # Simulate test_dataset_generalization.py logic
    decay = []
    no_decay = []
    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if id(param) in seen: continue
        seen.add(id(param))
        
        if param.ndim < 2 or "bias" in name or "norm" in name or "embed" in name:
            no_decay.append(name)
        else:
            decay.append(name)
            
    vulnerable_params = []
    for name in decay:
        if "adaLN" in name or "text_pool" in name:
            vulnerable_params.append(name)
            
    if vulnerable_params:
        print("  [WARNING] The following conditioning components are subject to weight decay:")
        for vp in vulnerable_params:
            print(f"    - {vp}")
        print("  If gradient signal is sparse (e.g. text alignment in large datasets),")
        print("  weight decay will push these to zero, cutting off text conditioning completely.")
    else:
        print("  ✓ Conditioning components are protected from weight decay.")

    # Let's also check the relative ratio of AdaLN gradient vs Time_embed gradient
    time_grad_norm = model.time_embed.mlp[0].weight.grad.norm().item() if model.time_embed.mlp[0].weight.grad is not None else 1e-8
    ada_grad_norm = model.blocks[0].adaLN.proj_down.weight.grad.norm().item() if model.blocks[0].adaLN.proj_down.weight.grad is not None else 0
    text_pool_grad_norm = list(model.text_pool_proj.parameters())[1].grad.norm().item() if list(model.text_pool_proj.parameters())[1].grad is not None else 0

    print(f"\n  Signal Strength Analysis:")
    print(f"  - Time Embed Grad Norm: {time_grad_norm:.6f}")
    print(f"  - Text Pool Grad Norm: {text_pool_grad_norm:.6f}")
    if text_pool_grad_norm < time_grad_norm * 0.1:
        print("  [WARNING] Text conditioning gradient is significantly weaker than Time conditioning.")
        print("  This increases the risk of the optimizer relying only on Time (and Seed).")

if __name__ == "__main__":
    run_diagnostics()
