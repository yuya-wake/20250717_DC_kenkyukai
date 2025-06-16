#!/usr/bin/env python
"""
Apply OCR to YouTube video thumbnails using Google Cloud Vision API.

For each image in the `thumbnails_20250105` directory, this script extracts text
via the Cloud Vision API and saves the detected full text into a `.txt` file
under the `ocr` directory.

Notes for Reproducibility
-------------------------
* Requires Google Cloud credentials with `GOOGLE_APPLICATION_CREDENTIALS` env var.
* Assumes Cloud Vision API is enabled for your project.
* Output format is plain UTF-8 `.txt` files named by video_id.
"""

import io
import os
from pathlib import Path
from tqdm import tqdm
from google.cloud import vision

def detect_text(path: Path, client: vision.ImageAnnotatorClient) -> str:
    """Run OCR using Google Cloud Vision and return the full text."""
    with path.open("rb") as f:
        content = f.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)
    return response.text_annotations[0].description if response.text_annotations else ""

def main():
    mnt_path = Path(os.getenv("MNT_PATH", ""))
    thumbnail_dir = mnt_path / "thumbnails_20250105"
    ocr_dir = mnt_path / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)

    client = vision.ImageAnnotatorClient()
    image_files = sorted(thumbnail_dir.glob("*.jpg")) + sorted(thumbnail_dir.glob("*.png"))

    for image_path in tqdm(image_files, desc="OCR Thumbnails"):
        try:
            extracted_text = detect_text(image_path, client)
            video_id = image_path.stem
            with (ocr_dir / f"{video_id}.txt").open("w", encoding="utf-8") as f:
                f.write(extracted_text)
        except Exception as e:
            print(f"Failed to process {image_path.name}: {e}")

if __name__ == "__main__":
    main()
