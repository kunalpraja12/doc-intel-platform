"""Lightweight layout detection placeholder.

This module provides a small LayoutDetector class that currently exposes a
simple heuristic: it uses pytesseract's block/line information (if available)
and returns bounding boxes for text blocks. For production, replace this
with a trained layout model (e.g., Detectron2, LayoutLM based detectors,
or use PaddleOCR layout modules).
"""
from __future__ import annotations

from typing import Any, Dict, List

try:
    import cv2
    import pytesseract
    from pytesseract import Output
except Exception:  # pragma: no cover - optional
    cv2 = None  # type: ignore
    pytesseract = None  # type: ignore
    Output = None  # type: ignore


class LayoutDetector:
    """Very small layout detector using pytesseract's block-level output."""

    def __init__(self) -> None:
        self.available = pytesseract is not None and cv2 is not None

    def detect_blocks(self, image) -> List[Dict[str, Any]]:
        """Return a list of text block bounding boxes with basic metadata."""
        if not self.available:
            raise RuntimeError("pytesseract and opencv-python required for LayoutDetector")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(rgb, output_type=Output.DICT)
        blocks: List[Dict[str, Any]] = []
        n = len(data.get("level", []))
        for i in range(n):
            level = data.get("level", [])[i]
            # level 2 = block-level in many tesseract setups; be permissive
            if level == 2 or level == 1:
                text = data.get("text", [])[i]
                blocks.append(
                    {
                        "text": text,
                        "left": int(data.get("left", [0])[i]),
                        "top": int(data.get("top", [0])[i]),
                        "width": int(data.get("width", [0])[i]),
                        "height": int(data.get("height", [0])[i]),
                    }
                )
        return blocks
