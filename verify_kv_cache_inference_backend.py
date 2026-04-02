import torch


class _SimpleTokenizer:
    # Pick a value that won't be produced by our dummy model unless we choose to.
    eot_token = 99999

    def encode(self, text: str, add_pad: bool = False, add_eot: bool = False):
        # Return a fixed short prompt tokenization.
        return [11, 22, 33, 44]

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if isinstance(tokens, int):
            tokens = [tokens]
        return " ".join(str(t) for t in tokens)


class _DummyModel:
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.blocks = [object(), object()]  # length only
        self.calls = []
        self._next = 101

    def set_allow_cross_attention(self, allow: bool):
        return None

    def _sample_next_token(
        self,
        next_token_logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        generated_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Deterministic "sampling" for verification.
        return torch.argmax(next_token_logits, dim=-1)

    def __call__(
        self,
        text_ids,
        images=None,
        timesteps=None,
        causal_text: bool = False,
        kv_cache=None,
        text_pos_offset=None,
        image_positions=None,
    ):
        # Record call structure for assertions.
        assert isinstance(text_ids, list) and len(text_ids) == 1
        tok = text_ids[0]
        assert isinstance(tok, torch.Tensor) and tok.dim() == 1
        self.calls.append(
            dict(
                L=int(tok.shape[0]),
                kv_cache_is_none=(kv_cache is None),
                text_pos_offset=text_pos_offset,
                has_images=images is not None,
                has_image_positions=image_positions is not None,
            )
        )

        # Produce logits for each provided token position.
        L = int(tok.shape[0])
        logits = torch.zeros(L, self.vocab_size, device=tok.device, dtype=torch.float32)

        # Make the last position predict a deterministic next token id (within vocab range).
        next_id = self._next % self.vocab_size
        logits[-1, next_id] = 1000.0
        self._next += 1

        mod_mask = torch.zeros(L, device=tok.device, dtype=torch.float32)  # all "text"
        return {"text": logits, "modality_mask": mod_mask}


def main():
    import inference_backend as ib

    if not torch.cuda.is_available():
        raise RuntimeError("This verification expects CUDA to match the inference backend autocast path.")

    # Build an InferenceModel instance without running its heavy __init__ (VAE/HF downloads).
    engine = ib.InferenceModel.__new__(ib.InferenceModel)
    engine.device = "cuda"
    engine.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    engine.tokenizer = _SimpleTokenizer()
    engine.vae = None
    engine.model = _DummyModel(vocab_size=256)

    # Run a short generation and exhaust the generator.
    out = list(
        engine.generate_multimodal_with_images(
            prompt="hello world",
            image_paths=None,
            max_new_tokens=8,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            min_p=0.0,
            repetition_penalty=1.0,
        )
    )

    # Assert KV-cache behavior: 1 prefill call with L>1, then decode calls with L==1.
    calls = engine.model.calls
    assert len(calls) >= 2, f"Expected at least 2 model calls (prefill+decode), got {len(calls)}"

    prefill = calls[0]
    assert prefill["L"] > 1, f"Prefill should pass the full prompt (L>1), got L={prefill['L']}"
    assert prefill["kv_cache_is_none"] is False, "Prefill should pass kv_cache (not None)"

    for i, c in enumerate(calls[1:], start=1):
        assert c["L"] == 1, f"Decode call {i} should pass a single token (L==1), got L={c['L']}"
        assert c["kv_cache_is_none"] is False, f"Decode call {i} should pass kv_cache (not None)"
        assert c["text_pos_offset"] is not None and len(c["text_pos_offset"]) == 1, (
            f"Decode call {i} should pass text_pos_offset=[pos], got {c['text_pos_offset']}"
        )
        assert c["has_images"] is False, f"Decode call {i} should not resend images; got images={c['has_images']}"
        assert c["has_image_positions"] is False, (
            f"Decode call {i} should not resend image_positions; got {c['has_image_positions']}"
        )

    assert any(u.get("type") == "final_text" for u in out), "Expected a final_text update from generator"
    print("OK: inference_backend.generate_multimodal_with_images now uses KV-cache prefill+decode (no O(N^2) loop).")


if __name__ == "__main__":
    main()

