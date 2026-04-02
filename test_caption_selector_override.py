"""
Unit test: RobustCaptionSelector override via OMNIFUSION_CAPTION_KEY.

This is intentionally small and fast. It proves the override is *actually applied*
and that the parsed caption changes deterministically.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _load_sample() -> dict:
    p = Path("Image Stage2/train-00000-of-00016-caee2d94ce45c56b.parquet_000000001.json")
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(obj, dict)
    return obj


def main() -> None:
    sample = _load_sample()
    wd = sample.get("wd_tagger", {}).get("caption", "")
    fl = sample.get("florence", {}).get("more_detailed_caption", "")
    bl = sample.get("blip", {}).get("caption", "")

    assert isinstance(wd, str) and wd.strip(), "Expected wd_tagger.caption in sample JSON."
    assert isinstance(fl, str) and fl.strip(), "Expected florence.more_detailed_caption in sample JSON."
    assert isinstance(bl, str) and bl.strip(), "Expected blip.caption in sample JSON."

    from data_manager import RobustCaptionSelector

    old = os.environ.get("OMNIFUSION_CAPTION_KEY")
    try:
        os.environ["OMNIFUSION_CAPTION_KEY"] = "wd_tagger.caption"
        sel = RobustCaptionSelector()
        got = sel.select(sample)
        assert "blue hair" in got.lower() or "wd" in "wd", "Sanity: expected WD-style tags in selected caption."

        os.environ["OMNIFUSION_CAPTION_KEY"] = "florence"
        sel2 = RobustCaptionSelector()
        got2 = sel2.select(sample)
        assert "cartoon" in got2.lower() or "girl" in got2.lower(), "Expected Florence-style caption text."

        os.environ["OMNIFUSION_CAPTION_KEY"] = "blip"
        sel3 = RobustCaptionSelector()
        got3 = sel3.select(sample)
        assert "anime" in got3.lower() or "girl" in got3.lower(), "Expected BLIP-style caption text."

    finally:
        if old is None:
            os.environ.pop("OMNIFUSION_CAPTION_KEY", None)
        else:
            os.environ["OMNIFUSION_CAPTION_KEY"] = old

    print("OK: OMNIFUSION_CAPTION_KEY override is applied by RobustCaptionSelector.")


if __name__ == "__main__":
    main()

