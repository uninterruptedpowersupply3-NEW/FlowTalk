"""
Diagnostic script to determine WHY "red" and "blue" prompts produce identical images.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Force UTF-8 output for Windows console
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn.functional as F
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
from data_manager import TiktokenTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

def main():
    tokenizer = TiktokenTokenizer()
    
    print("=" * 70)
    print("TEST 1: Tokenizer Output Comparison")
    print("=" * 70)
    
    tokens_red_padded = tokenizer.encode("red")
    tokens_blue_padded = tokenizer.encode("blue")
    tokens_empty_padded = tokenizer.encode("")
    
    pad_token = 100258
    red_real = (tokens_red_padded != pad_token).sum().item()
    blue_real = (tokens_blue_padded != pad_token).sum().item()
    
    print(f"  'red'  real tokens: {red_real} / {len(tokens_red_padded)} ({red_real/len(tokens_red_padded)*100:.1f}% real)")
    print(f"  'blue' real tokens: {blue_real} / {len(tokens_blue_padded)} ({blue_real/len(tokens_blue_padded)*100:.1f}% real)")
    print(f"  First 5 of 'red':  {tokens_red_padded[:5].tolist()}")
    print(f"  First 5 of 'blue': {tokens_blue_padded[:5].tolist()}")
    
    tokens_match = torch.equal(tokens_red_padded, tokens_blue_padded)
    print(f"\n  'red' == 'blue' tokens identical? {tokens_match}")
    if not tokens_match:
        diff_positions = (tokens_red_padded != tokens_blue_padded).nonzero().squeeze()
        if diff_positions.dim() == 0:
            diff_positions = diff_positions.unsqueeze(0)
        print(f"  Differ at {len(diff_positions)} position(s): {diff_positions.tolist()}")
        for pos in diff_positions:
            p = pos.item()
            print(f"    pos={p}: red={tokens_red_padded[p].item()}, blue={tokens_blue_padded[p].item()}")
    
    print("\n" + "=" * 70)
    print("TEST 2: Load Model and Check Text Conditioning Pathway")
    print("=" * 70)
    
    try:
        from inference_backend import InferenceModel
        import glob
        
        ckpt_path = os.path.join("dataset_gen_checkpoints", "trained_model.pt")
        if not os.path.exists(ckpt_path):
            files = glob.glob(os.path.join("dataset_gen_checkpoints", "*.pt"))
            if files:
                ckpt_path = max(files, key=os.path.getctime)
            else:
                print("  No checkpoint found. Skipping model tests.")
                return
        
        print(f"  Loading model from: {ckpt_path}")
        engine = InferenceModel(ckpt_path)
        model = engine.model
        model.eval()
        
        print(f"  Model loaded. d_model={model.config.d_model}, n_layers={model.config.n_layers}")
        
        # 2a: Text Embedding Similarity
        print("\n--- 2a: Text Embedding Layer ---")
        with torch.no_grad():
            emb_red = model.text_embed(tokens_red_padded.to(DEVICE))
            emb_blue = model.text_embed(tokens_blue_padded.to(DEVICE))
            
            # Full embedding cosine sim (ALL 512 tokens including pads)
            cos_sim_full = F.cosine_similarity(
                emb_red.flatten().unsqueeze(0).float(), 
                emb_blue.flatten().unsqueeze(0).float()
            ).item()
            
            # Only non-pad tokens
            red_mask = (tokens_red_padded != pad_token)
            blue_mask = (tokens_blue_padded != pad_token)
            
            emb_red_real = emb_red[red_mask.to(DEVICE)]
            emb_blue_real = emb_blue[blue_mask.to(DEVICE)]
            
            # Mean pooling (as done in pack_inputs)
            red_pooled = emb_red_real.float().mean(dim=0)
            blue_pooled = emb_blue_real.float().mean(dim=0)
            
            cos_sim_pooled = F.cosine_similarity(
                red_pooled.unsqueeze(0), 
                blue_pooled.unsqueeze(0)
            ).item()
            
            print(f"  Full embedding cosine sim (incl pads): {cos_sim_full:.6f}")
            print(f"  Mean-pooled (non-pad) cosine sim: {cos_sim_pooled:.6f}")
        
        # 2b: text_pool_proj output
        print("\n--- 2b: text_pool_proj Output ---")
        with torch.no_grad():
            text_cond_red = model.text_pool_proj(red_pooled.to(DTYPE).unsqueeze(0))
            text_cond_blue = model.text_pool_proj(blue_pooled.to(DTYPE).unsqueeze(0))
            
            cos_text_cond = F.cosine_similarity(
                text_cond_red.float(), text_cond_blue.float()
            ).item()
            
            print(f"  text_pool_proj cosine sim (red vs blue): {cos_text_cond:.6f}")
            print(f"  text_cond 'red'  norm:  {text_cond_red.float().norm().item():.4f}")
            print(f"  text_cond 'blue' norm:  {text_cond_blue.float().norm().item():.4f}")
            
            # Compare to timestep embedding
            t = torch.tensor([0.5], device=DEVICE, dtype=DTYPE)
            t_emb = model.time_embed(t)
            ratio_red = text_cond_red.float().norm().item() / max(t_emb.float().norm().item(), 1e-8)
            ratio_blue = text_cond_blue.float().norm().item() / max(t_emb.float().norm().item(), 1e-8)
            print(f"  timestep_emb norm (t=0.5): {t_emb.float().norm().item():.4f}")
            print(f"  Ratio text_cond/t_emb (red):  {ratio_red:.4f}")
            print(f"  Ratio text_cond/t_emb (blue): {ratio_blue:.4f}")
        
        # 2c: Full model forward comparison
        print("\n--- 2c: Full Forward Pass Comparison ---")
        with torch.no_grad():
            torch.manual_seed(42)
            latent = torch.randn(16, 32, 32, device=DEVICE, dtype=DTYPE)
            
            out_red = model.forward(
                [tokens_red_padded.to(DEVICE)], 
                [latent.clone()], 
                torch.tensor([0.5], device=DEVICE, dtype=DTYPE), 
                causal_text=True
            )
            
            out_blue = model.forward(
                [tokens_blue_padded.to(DEVICE)], 
                [latent.clone()], 
                torch.tensor([0.5], device=DEVICE, dtype=DTYPE), 
                causal_text=True
            )
            
            img_pred_red = out_red["image"]
            img_pred_blue = out_blue["image"]
            mod_mask = out_red["modality_mask"]
            
            img_mask = (mod_mask == 1.0)
            img_red = img_pred_red[img_mask].float()
            img_blue = img_pred_blue[img_mask].float()
            
            if img_red.numel() > 0 and img_blue.numel() > 0:
                cos_img = F.cosine_similarity(
                    img_red.flatten().unsqueeze(0), 
                    img_blue.flatten().unsqueeze(0)
                ).item()
                
                mse_img = F.mse_loss(img_red, img_blue).item()
                max_diff = (img_red - img_blue).abs().max().item()
                
                print(f"  Image prediction cosine sim:   {cos_img:.6f}")
                print(f"  Image prediction MSE:          {mse_img:.10f}")
                print(f"  Image prediction max |diff|:   {max_diff:.10f}")
                
                if cos_img > 0.999:
                    print(f"\n  CRITICAL: Image predictions are NEARLY IDENTICAL (cos={cos_img:.6f})")
                    print(f"     >>> Model IS ignoring the text prompt <<<")
                elif cos_img > 0.99:
                    print(f"\n  WARNING: Image predictions are VERY SIMILAR (cos={cos_img:.6f})")
                    print(f"     Text conditioning has minimal effect.")
                else:
                    print(f"\n  OK: Image predictions differ meaningfully (cos={cos_img:.6f})")
                    print(f"     Text conditioning IS working. Bug may be elsewhere.")
        
        # 2d: AdaLN gate analysis
        print("\n--- 2d: AdaLN Gate Analysis ---")
        with torch.no_grad():
            for i, block in enumerate(model.blocks[:3]):
                gate_w = block.adaLN.proj_up.weight
                d = model.config.d_model
                gate_msa_w = gate_w[2*d:3*d]
                gate_mlp_w = gate_w[5*d:6*d]
                
                print(f"  Block {i}: gate_msa_w norm={gate_msa_w.float().norm():.4f}, "
                      f"gate_mlp_w norm={gate_mlp_w.float().norm():.4f}")
        
        # 2e: Check if pad tokens are embedding as zero or meaningful
        print("\n--- 2e: Pad Token Embedding Analysis ---")
        with torch.no_grad():
            pad_emb = model.text_embed(torch.tensor([pad_token], device=DEVICE))
            eot_emb = model.text_embed(torch.tensor([100257], device=DEVICE))  # EOT
            red_tok_emb = model.text_embed(torch.tensor([1171], device=DEVICE))  # "red"
            blue_tok_emb = model.text_embed(torch.tensor([12481], device=DEVICE))  # "blue"
            
            print(f"  Pad  token embedding norm: {pad_emb.float().norm().item():.4f}")
            print(f"  EOT  token embedding norm: {eot_emb.float().norm().item():.4f}")
            print(f"  'red' token embedding norm: {red_tok_emb.float().norm().item():.4f}")
            print(f"  'blue' token embedding norm: {blue_tok_emb.float().norm().item():.4f}")
            
            cos_red_blue_emb = F.cosine_similarity(
                red_tok_emb.float(), blue_tok_emb.float()
            ).item()
            print(f"  Cosine sim raw 'red' vs 'blue' embedding: {cos_red_blue_emb:.6f}")
        
        print("\n" + "=" * 70)
        print("DIAGNOSIS SUMMARY")
        print("=" * 70)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Error: {e}")

if __name__ == "__main__":
    main()
