"""Tesseract OCR wrapper for the extraction pipeline.

Provides a simple OCRExtractor that runs image preprocessing (via OpenCV)
and invokes pytesseract to extract plain text and structured word-level
results (with bounding boxes and confidence scores).

Note: Requires Tesseract to be installed on the host (https://github.com/tesseract-ocr/tesseract).
On Windows, ensure tesseract.exe is on PATH or set pytesseract.pytesseract.tesseract_cmd.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List

# Ensure project root is on sys.path so imports like 'pipeline' resolve when
# running this file directly (python pipeline/extract/ocr.py).
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env when this module is executed as a script so TESSERACT_CMD from .env is available.
# This is safe when imported by the FastAPI app as well.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv is optional; if missing, environment variables must be set externally
    pass

try:
    import cv2
    import pytesseract
    from pytesseract import Output
except Exception:  # pragma: no cover - runtime dependency
    cv2 = None  # type: ignore
    pytesseract = None  # type: ignore
    Output = None  # type: ignore


class OCRExtractor:
    """Simple OCR extractor using pytesseract.

    Methods
    - extract_from_path(path, use_preprocess=True, upscale=None): returns dict with 'text' and 'words' list
    - extract_from_image(image, upscale=None): accepts an OpenCV BGR image (numpy array)

    The extractor can optionally apply preprocessing from pipeline.extract.cv_preprocessing
    when extracting from a file path. The CLI exposes a --raw flag to skip preprocessing
    and compare results. Upscaling can be applied automatically for small images.
    """

    def __init__(self, tesseract_cmd: str | None = None, default_upscale: int | None = None) -> None:
        # Prefer explicit arg, otherwise read from TESSERACT_CMD env var
        cmd = tesseract_cmd if tesseract_cmd is not None else os.getenv("TESSERACT_CMD")
        if cmd and pytesseract is not None:
            pytesseract.pytesseract.tesseract_cmd = cmd
        self.available = pytesseract is not None and cv2 is not None
        # default_upscale: if None, the extractor will auto-upscale images under threshold
        self.default_upscale = default_upscale

    def _ensure_available(self) -> None:
        if not self.available:
            raise RuntimeError(
                "pytesseract and opencv-python are required for OCRExtractor. Install them and ensure Tesseract is available."
            )

    def _maybe_upscale(self, img, upscale: int | None) :
        """Upscale image by factor if requested or if image is small and default_upscale applies."""
        try:
            h, w = img.shape[:2]
        except Exception:
            return img
        # If explicit upscale passed, use it; otherwise, use default_upscale when image is under 1000px
        factor = None
        if upscale and isinstance(upscale, int) and upscale > 1:
            factor = upscale
        elif self.default_upscale and isinstance(self.default_upscale, int) and self.default_upscale > 1 and w < 1000:
            factor = self.default_upscale
        elif upscale is None and self.default_upscale is None and w < 1000:
            # implicit sensible default for small images
            factor = 2
        if factor and factor > 1:
            new_w = int(w * factor)
            new_h = int(h * factor)
            print(f"Upscaling image from {w}x{h} to {new_w}x{new_h} (factor {factor}x)")
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return img

    def extract_from_path(self, path: str, use_preprocess: bool = True, upscale: int | None = None) -> Dict[str, Any]:
        """Load an image from path, optionally preprocess it, upscale, and run OCR.

        use_preprocess=True (default) applies pipeline.extract.cv_preprocessing.preprocess_image
        if available. If use_preprocess=False the raw image bytes are passed to OCR.
        upscale: integer factor to resample image (2 means 2x). If None, auto-upscale small images.
        """
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found or cannot be read: {path}")

        # Upscale if requested or auto-triggered
        try:
            img = self._maybe_upscale(img, upscale)
        except Exception as exc:
            print(f"Warning: upscale failed: {exc}")

        if use_preprocess:
            # Attempt to import the preprocessing convenience function
            try:
                from pipeline.extract.cv_preprocessing import preprocess_image

                img = preprocess_image(img)
            except Exception as exc:
                # If preprocessing fails, warn and continue with raw image
                print(f"Warning: preprocessing failed or not available: {exc}")

        return self.extract_from_image(img)

    def extract_from_image(self, image, upscale: int | None = None) -> Dict[str, Any]:
        """Run pytesseract on an OpenCV BGR image.

        Returns:
          { 'text': str, 'words': [ {text, left, top, width, height, conf} ] }
        """
        # apply upscale to in-memory image as well
        if upscale:
            try:
                image = self._maybe_upscale(image, upscale)
            except Exception as exc:
                print(f"Warning: upscale failed: {exc}")
        self._ensure_available()

        # Convert BGR to RGB for pytesseract
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use pytesseract to get plain text
        raw_text = pytesseract.image_to_string(rgb)
        rotated_text = self._extract_rotated_text(rgb)
        additional_text = self._get_new_rotated_text(raw_text, rotated_text)
        if additional_text:
            raw_text = f"{raw_text.rstrip()}\n[additional rotated-text pass]: {additional_text}"

        # Use the TSV output to obtain word-level boxes and confidences
        data = pytesseract.image_to_data(rgb, output_type=Output.DICT)
        words: List[Dict[str, Any]] = []
        n_boxes = len(data.get("level", []))
        for i in range(n_boxes):
            text = data.get("text", [])[i]
            if not text or text.strip() == "":
                continue
            try:
                conf = float(data.get("conf", [])[i])
            except Exception:
                conf = None
            word = {
                "text": text,
                "left": int(data.get("left", [0])[i]),
                "top": int(data.get("top", [0])[i]),
                "width": int(data.get("width", [0])[i]),
                "height": int(data.get("height", [0])[i]),
                "conf": conf,
            }
            words.append(word)

        return {"text": raw_text, "words": words}

    @staticmethod
    def _get_new_rotated_text(main_text: str, rotated_text: str) -> str:
        main_words = set(re.findall(r"[a-z0-9]+", main_text.casefold()))
        additional_lines: list[str] = []
        seen_words = set(main_words)
        for line in rotated_text.splitlines():
            new_words = []
            for word in re.findall(r"[a-z0-9]+", line, flags=re.IGNORECASE):
                normalized = word.casefold()
                if len(normalized) > 1 and normalized not in seen_words:
                    seen_words.add(normalized)
                    new_words.append(word)
            if new_words:
                additional_lines.append(" ".join(new_words))
        return "\n".join(additional_lines)

    @staticmethod
    def _rotate_image(image):
        """Rotate an image -45 degrees, expanding the canvas to avoid clipping."""
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, -45, 1.0)
        abs_cos = abs(matrix[0, 0])
        abs_sin = abs(matrix[0, 1])
        new_width = int(height * abs_sin + width * abs_cos)
        new_height = int(height * abs_cos + width * abs_sin)
        matrix[0, 2] += new_width / 2 - center[0]
        matrix[1, 2] += new_height / 2 - center[1]
        return cv2.warpAffine(
            image,
            matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    def _extract_rotated_text(self, rgb_image) -> str:
        rotated_image = self._rotate_image(rgb_image)
        return pytesseract.image_to_string(rotated_image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OCR on an image file using Tesseract.")
    parser.add_argument("path", help="Path to image file")
    parser.add_argument("--tesseract-cmd", help="Path to tesseract executable", default=None)
    parser.add_argument("--raw", help="Skip preprocessing and run OCR on the raw image", action="store_true")
    parser.add_argument("--upscale", help="Upscale factor (integer). If omitted, small images are auto-upscaled 2x.", type=int, default=None)
    parser.add_argument("--out", help="Write extracted plain text to a file", default=None)
    args = parser.parse_args()

    extractor = OCRExtractor(tesseract_cmd=args.tesseract_cmd)
    res = extractor.extract_from_path(args.path, use_preprocess=not args.raw, upscale=args.upscale)

    text = res["text"]
    print("---TEXT---")
    print(text)
    print("---WORDS (sample 20)---")
    for w in res["words"][:20]:
        print(w)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote extracted text to {args.out}")
