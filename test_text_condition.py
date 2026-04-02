import torch
from test_dataset_generalization import TestConfig, DatasetGeneralizationTest

print("--- Testing Text Conditioning ---")
config = TestConfig(batch_size=1, use_bucketed_batching=False, num_workers=0)
tester = DatasetGeneralizationTest(config)

# Load checkpoint
checkpoint_path = "dataset_gen_checkpoints/my_model_step_1000.pt"
import glob
import os
if not os.path.exists(checkpoint_path):
    pts = glob.glob("dataset_gen_checkpoints/*.pt")
    if pts:
        pts.sort(key=os.path.getmtime, reverse=True)
        checkpoint_path = pts[0]

print(f"Loading {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu")
tester.model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
tester.model.eval()

# Two different prompts
prompt1 = "blue tint"
prompt2 = "green tint"

from test_dataset_generalization import encode_prompt_tokens
p1_ids = encode_prompt_tokens(tester.dataset.tokenizer, prompt1).to(tester.device)
p2_ids = encode_prompt_tokens(tester.dataset.tokenizer, prompt2).to(tester.device)

# Same noise
torch.manual_seed(42)
fixed_noise = torch.randn(16, 16, 16, device=tester.device, dtype=tester.dtype)
t_batch = torch.full((1,), 0.5, device=tester.device, dtype=tester.dtype)

with torch.no_grad():
    out1 = tester.model.forward([p1_ids], [fixed_noise], t_batch, causal_text=True)
    v1 = out1["image"][out1["modality_mask"] == 1.0]

    out2 = tester.model.forward([p2_ids], [fixed_noise], t_batch, causal_text=True)
    v2 = out2["image"][out2["modality_mask"] == 1.0]

diff = (v1 - v2).abs().mean().item()
print(f"Mean Absolute Difference in v_pred between '{prompt1}' and '{prompt2}': {diff:.6f}")

if diff < 1e-4:
    print("Conclusion: The model is COMPLETELY IGNORING the text condition.")
else:
    print("Conclusion: The model IS responding to the text condition, but it might be generating similar images due to overfitting.")
