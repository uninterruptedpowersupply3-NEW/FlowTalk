# Flow-Matching Multimodal (Prototype, Very Broken)

This repository is a **research prototype** for a single-transformer multimodal model that can:

- Generate images with **flow-matching** in **VAE latent space**
- Generate text autoregressively (next-token prediction)
- Train on **packed contexts** (mixed image + text sequences)

## Read This First (Seriously)

This project is **not production-ready**. It is **not stable**. It is **not reproducible by default**.

If you are looking for “a working SDXL alternative” or a clean training stack, this repo is the wrong place.

## Expected failure modes:

- Training runs that “work” but learn the wrong thing (especially if you reuse a latent cache incorrectly)
- Models that appear to **ignore prompts** (often due to prompt-format mismatch)
- Text generation that is repetitive or nonsensical
- Huge GPU time spent debugging silent footguns
- Windows-specific pain around compilation/backends (Triton/FlexAttention/torch.compile variability)

## This checkpoint often needs longer / more specific prompts to noticeably change structure (known limitation).

![Screenshot 2026-04-01 193854](https://github.com/user-attachments/assets/e73dbedc-bf78-4872-85a5-57b8a50a87e8)

Here is a comparision between using a short and long prompt
The prompt for the first image was "butterflyer, butterflies, butterflylike, butterfly, browntail, brownwort" and the prompt for the second image was "butterflyer, butterflies, butterflylike, butterfly, browntail, brownwort, microlepidopteran, microlepidopterous, bugwort, butterflying, butterwort, mariposas, lepidopteran, lepidopterous, beetleweed, featherleaf, leafhopper, leafhoppers, nutbrown, throatwort"

note the model has a bias for generating prompts that may result in blue or green areas, and it might struggle with other scenes even with similar quality prompts

![Screenshot 2026-04-01 193556](https://github.com/user-attachments/assets/5077a522-506e-44ee-bd67-9c039daebda2)

EXAMPLE OF THIS BIAS

The first prompt for the first image was "butterflyer, butterflies, butterflylike, butterfly, browntail, brownwort" while the second was "dogger, dogs, doglike, dog, canine, canid"

## What’s In Here (No Packaging, Just Scripts)

Key files (non-exhaustive):

- `omni_model_v2.py`: model definition (multimodal transformer, attention, conditioning)
- `vae_module.py`: VAE loading/encode/decode helpers (Flux VAE is commonly used)
- `test_dataset_generalization.py`: training + evaluation harness (SSIM, generalization probes, save/load)
- `training_backend.py`: training backend utilities
- `inference_backend.py`: inference logic used by the GUI
- `gui_app.py`: simple GUI for chat + image generation
- `precompute_metadata.py`, `precompute_latents.py`: utilities for building `.latent_cache`
- `launcher_gui.py`:simple GUI for building `.latent_cache`

There is **no** stable CLI/API contract here. Scripts are edited frequently.

## Prompt Format Matters (A Lot)

This model learns **the exact token distribution** of whatever you train it on.

If your captions/prompts in `.latent_cache` are ChatML-ish (tokens like `<|im_start|>user`, `<|im_end|>`),
then giving the model a plain prompt like:

`green trees, flowers`

is **out-of-distribution**. In practice, this can look like:

- different prompts produce nearly the same “blob”
- prompt changes only affect tint/texture

The inference backend tries to detect this situation and auto-wrap prompts, but this is **best-effort**.
If you see logs printing a ChatML prompt, the wrapper is active.

## Latent Cache Footgun Warning

Training can use a precomputed latent cache (`.latent_cache/`). This can save a lot of time, but it is also
the easiest way to accidentally train on the wrong data.

Rules of thumb:

- If you switch datasets, use a **different** `--cache-dir` (or delete the old one).
- If results “don’t change” when you point training at a new `--data-dir`, assume you are still training on
  an old cache until proven otherwise.
- Keep `--data-dir` and `--cache-dir` paired. Do not share one cache across unrelated folders.

## Quick Start (Minimal, Not Guaranteed)

1. Create a venv
2. Install PyTorch + CUDA that matches your GPU (recomended pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128)
3. Install flash attn2 wheels from here https://wildminder.github.io/AI-windows-whl/
4. Install ``uv pip install uv`` (Optional for speed)
5. Install the requirements ``uv pip install -r req.txt --reinstall`` uv is optional for speed
6. Run scripts directly

Example training entrypoint:

```powershell
python -u test_dataset_generalization.py ^
  --data-dir "C:\\path\\to\\images_with_txt_captions" ^
  --cache-dir "C:\\path\\to\\.latent_cache" ^
  --batch-size 1 --grad-accum-steps 8 ^
  --d-model 768 --n-heads 12 --n-layers 12 ^
  --max-size 256 --max-context-length 4096 ^
  --text-pooling attn --pooled-text-scale 1.0 --pooled-text-dropout 0.1
```

Example GUI:

```powershell
python -u gui_app.py
```

The model currenly at this hugging face page 

UPShf/FlowTalk

was pre trained on a small sample of image net 

( https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256) EVEN A SMALL SAMPLE OF THIS

That is only contained 256x256 753,088 image text pairs or 376,544 image only samples and all of then where exatctly 256x256. It was trained for 5 epochs using the following arguments

```
python -u test_dataset_generalization.py --data-dir "C:\Users\chatr\Pictures\aaa\imagenet-1k-256x256\CaptinedIMGNET" --cache-dir "C:\Users\chatr\Documents\Tech\VLLM\New folder\.latent_cache" --batch-size 1 --lr 2e-4 --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --max-size 256 --grad-accum-steps 8 --d-model 768 --n-heads 12 --n-layers 12 --warmup-steps 900 --text-pooling attn --pooled-text-scale 1.0 --pooled-text-dropout 0.1 --output-name IMAGENET_FIXED --save-every 1000 --epochs 5 --max-steps 0 --lambda-img 1.0 --alpha-ntp 0.5 --alpha-ntp-text-only 1.0 --workers 0 --graph-batch-size 1 --ema-every 8 --prefetch-factor 2 --log-every 10 --parallel-encode --grad-checkpointing --context-pack --max-context-length 4096 --image-ratio 0.5
```

## Offline Mode (No Hugging Face Network Calls)

If you have already downloaded required model files, you can force offline behavior:

- Set `HF_HUB_OFFLINE=1`
- Ensure the VAE and any HF assets are present in your local HF cache

If the repo still tries to contact Hugging Face, assume your local cache is missing required files.

## License / Data Disclaimer

Code in this repository is intended to be released under **Apache-2.0** (see `LICENSE`).

This repository does **not** include ImageNet or other copyrighted datasets.
If you train on ImageNet-derived data, you are responsible for complying with ImageNet’s terms and any
underlying image copyrights.

## Non-Goals

- Safety filtering
- A general-purpose assistant
- A stable training framework
- A clean, well-tested library API
