
import sys
import os
import torch
import logging
from test_dataset_generalization import DEVICE

# Use the InferenceModel wrapper (we can import it or reimplement it)
# Since I can't import `inference_backend.py` easily if it's not a module (it's a script), 
# I'll just reimplement the inference logic here for standalone verification.

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from omni_model_v2 import OmniFusionV2, OmniConfigV2
from data_manager import TiktokenTokenizer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("VerifyText")

def verify():
    model_path = r"C:\Users\chatr\Documents\Tech\VLLM\New folder\dataset_gen_checkpoints\smalles_default_model.pt"
    
    if not os.path.exists(model_path):
        # Try finding the latest checkpoint if exact name varies (e.g. step number suffix)
        import glob
        files = glob.glob(r"C:\Users\chatr\Documents\Tech\VLLM\New folder\dataset_gen_checkpoints\smalles_default_model*.pt")
        if files:
            model_path = max(files, key=os.path.getctime)
            logger.info(f"Found checkpoint: {model_path}")
        else:
            logger.error(f"Model not found at {model_path}. Did training run?")
            return

    logger.info("Initializing Model for Verification...")
    
    # 1. Config (Must match training defaults in test_dataset_generalization.py)
    omni_config = OmniConfigV2(
        d_model=384, n_layers=6, n_heads=6, head_dim=64,
        vocab_size=100352, qk_norm=True, attention_logit_cap=50.0
    )
    
    model = OmniFusionV2(omni_config).to(DEVICE)
    
    # Load Weights
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    tokenizer = TiktokenTokenizer()
    
    # 2. Test Questions from SMALLESTMATH.jsonl
    questions = [
        "Calculate: -831.1 * 923.735",
        "Calculate: -666.411 / 324.667",
        "Calculate: -536.109 + 298.664",
        "Solve for x in the equation: 19x + 1 = 33"
    ]
    
    expected_snippets = [
        "-767716",
        "-2.0526",
        "-237.445",
        "1.68421"
    ]
    
    logger.info("Starting Verification Generations...\n")
    
    passed = 0
    for i, q in enumerate(questions):
        # Add system prompt to match training data distribution
        prompt = f"<|im_start|>system\nYou are a math assistant. Solve the following problem concisely.\n<|im_end|>\n<|im_start|>user\n{q}\n<|im_end|>\n<|im_start|>assistant\n"
        
        # Encode
        input_ids = tokenizer.encode(prompt, add_pad=False, add_eot=False).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # Generate mechanism
            curr_ids = input_ids
            generated = []
            
            for step in range(50): # Max new tokens
                # Model forward
                # Note: expects list of inputs
                txt_input = [curr_ids[0]]
                t_batch = torch.tensor([1.0], device=DEVICE).to(curr_ids.dtype) if curr_ids.dtype.is_floating_point else torch.tensor([1.0], device=DEVICE).float()
                if torch.cuda.is_available():
                     t_batch = t_batch.half() # Float16

                # Autocast
                with torch.amp.autocast('cuda', dtype=torch.float16):
                     res = model(txt_input, images=None, timesteps=t_batch, causal_text=True)
                
                logits = res["text"]
                # Handle flattened output capabilities
                
                # Input length (valid tokens)
                valid_len = curr_ids.shape[1]
                
                if logits.dim() == 2:
                    # [Total_Padded_Tokens, V] -> take index (valid_len - 1)
                    # Note: logits[i] predicts token at i+1
                    next_token_logits = logits[valid_len - 1, :]
                else:
                    # [B, Padded_L, V]
                    next_token_logits = logits[0, valid_len - 1, :]

                next_token = torch.argmax(next_token_logits).item()
                
                if next_token == tokenizer.eot_token or next_token == 100259: # <|im_end|>
                    break
                
                generated.append(next_token)
                curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]], device=DEVICE)], dim=1)
        
        output_text = tokenizer.decode(generated)
        logger.info(f"Q: {q}")
        logger.info(f"A: {output_text.strip()}")
        
        # Check if expected answer is in output
        expected = expected_snippets[i]
        if expected in output_text:
            logger.info(f"✅ PASSED (Found {expected})")
            passed += 1
        else:
            logger.info(f"❌ FAILED (Expected {expected})")
        logger.info("-" * 40)

    if passed == len(questions):
        logger.info(f"Result: SUCCESS ({passed}/{len(questions)} passed)")
    else:
        logger.info(f"Result: MIXED ({passed}/{len(questions)} passed)")

if __name__ == "__main__":
    verify()
