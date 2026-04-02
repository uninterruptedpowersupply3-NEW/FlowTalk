"""
Regression test for deterministic caption-source selection.

This proves that setting:
  OMNIFUSION_CAPTION_SAMPLING=deterministic
causes RobustCaptionSelector to pick a stable caption source per sample identity, so the same
image is not trained with contradictory captions across epochs.
"""

from __future__ import annotations

import os

from data_manager import RobustCaptionSelector


def main() -> None:
    os.environ["OMNIFUSION_CAPTION_SAMPLING"] = "deterministic"
    os.environ.pop("OMNIFUSION_CAPTION_KEY", None)  # ensure we're not forcing a single key

    sel = RobustCaptionSelector()
    assert sel.sampling_mode == "deterministic", f"Expected deterministic mode, got {sel.sampling_mode}"

    # A sample with multiple mutually-different captions.
    sample = {
        "image_filename": "example_000001.jpg",
        "wd_tagger": {"caption": "1girl, solo, blue hair, blue eyes"},
        "blip": {"caption": "anime girl with red hair"},
        "florence": {"more_detailed_caption": "A cartoon drawing of a girl with long green hair and yellow eyes."},
        "question_used_for_image": "Can you describe the general tone?",
    }

    first = sel.select(sample)
    assert isinstance(first, str) and first.strip(), "Selector returned empty caption in deterministic mode"

    # Run multiple times to ensure stability.
    for i in range(50):
        nxt = sel.select(sample)
        assert nxt == first, f"Non-deterministic caption selection at iter={i}: '{first}' vs '{nxt}'"

    # Different identity should be allowed to map to a different source/caption.
    sample2 = dict(sample)
    sample2["image_filename"] = "example_000002.jpg"
    other = sel.select(sample2)
    assert isinstance(other, str) and other.strip()
    # Not guaranteed different (hash could land in same bucket), but it's very likely with different ids.
    # We still print it for visibility when running manually.
    print("OK deterministic caption selection.")
    print("sample1:", first)
    print("sample2:", other)


if __name__ == "__main__":
    main()

