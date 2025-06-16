#!/usr/bin/env python
"""
Clean OCR text outputs produced by Google Vision API.

For each `.txt` file in the `ocr/original_ocr/` directory, the script:
1. Joins lines into a single space-separated line
2. Strips excess whitespace
3. Ensures terminal punctuation ('。')
4. Removes noisy characters like repeated 'w' (e.g., "www")

Cleaned text is saved to `ocr/cleaned_ocr/` under the same filename.

Notes
-----
* Input: UTF-8 `.txt` files, one per video thumbnail
* Output: Cleaned `.txt` files with same base name
"""

import os
import re
from pathlib import Path
from tqdm import tqdm

def preprocess_ocr(raw_text: str) -> str:
    """Perform rule-based cleaning of OCR output."""
    text = raw_text.replace("\n", " ")                      # Join lines with space
    text = re.sub(r"\s+", " ", text).strip()                # Collapse whitespace
    if not text.endswith("。"):
        text += "。"                                           # Ensure final punctuation
    text = re.sub(r"[wｗ]{2,}", "", text)                    # Remove noisy repeated 'w'
    return text

def main():
    mnt_path = Path(os.getenv("MNT_PATH", ""))
    ocr_dir = mnt_path / "ocr" / "original_ocr"
    output_dir = mnt_path / "ocr" / "cleaned_ocr"
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(sorted(ocr_dir.glob("*.txt")), desc="Cleaning OCR text"):
        with file_path.open("r", encoding="utf-8") as f:
            text = f.read()
        cleaned = preprocess_ocr(text)
        with (output_dir / file_path.name).open("w", encoding="utf-8") as f:
            f.write(cleaned)

if __name__ == "__main__":
    main()
