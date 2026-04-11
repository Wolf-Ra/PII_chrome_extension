"""
preprocessor.py — Image preprocessing for scanned PDF pages.

Pipeline per page:
  1. Deskew   — detect dominant line angle via Hough, rotate to correct.
  2. Denoise  — Gaussian blur for mild noise; Non-Local Means for heavy noise.
  3. Binarise — Otsu global threshold (fast) or adaptive for uneven lighting.
"""

from __future__ import annotations
import math
import numpy as np
import cv2
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Deskew
# ──────────────────────────────────────────────────────────────────────────────

def _detect_skew_angle(gray: np.ndarray) -> float:
    """
    Detect skew angle using Hough line transform.
    Returns angle in degrees; positive = clockwise tilt.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Only trust near-horizontal lines (text baselines)
            if abs(angle) < 45:
                angles.append(angle)

    if not angles:
        return 0.0

    # Median is more robust than mean against stray lines
    return float(np.median(angles))


def deskew(image: np.ndarray) -> np.ndarray:
    """Rotate `image` to compensate for detected skew."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    angle = _detect_skew_angle(gray)

    if abs(angle) < 0.3:          # negligible — skip rotation
        return image

    h, w = image.shape[:2]
    centre = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


# ──────────────────────────────────────────────────────────────────────────────
# Denoise
# ──────────────────────────────────────────────────────────────────────────────

def denoise(image: np.ndarray, strength: str = "auto") -> np.ndarray:
    """
    strength:
      "light"  → Gaussian blur (fast, good for scanner noise)
      "heavy"  → Non-Local Means (slow, good for low-quality photo scans)
      "auto"   → estimate from variance; pick the right one automatically
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    if strength == "auto":
        # Estimate noise variance via Laplacian
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        strength = "heavy" if lap_var > 500 else "light"

    if strength == "light":
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
    else:
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    return denoised


# ──────────────────────────────────────────────────────────────────────────────
# Binarise
# ──────────────────────────────────────────────────────────────────────────────

def binarise(image: np.ndarray, method: str = "auto") -> np.ndarray:
    """
    Convert to binary (black text on white).

    method:
      "otsu"      → global Otsu threshold (fast, even lighting)
      "adaptive"  → adaptive mean threshold (uneven lighting / shadows)
      "auto"      → pick based on pixel-value std-dev
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    if method == "auto":
        std = gray.std()
        method = "adaptive" if std > 60 else "otsu"

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=10,
        )

    return binary


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_page(image: np.ndarray) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to one page image.
    Returns a binary (grayscale) image ready for OCR.
    Order: deskew → denoise → binarise.
    """
    image = deskew(image)
    image = denoise(image, strength="auto")
    image = binarise(image, method="auto")
    return image


def preprocess_pdf_pages(pages: list[np.ndarray]) -> list[np.ndarray]:
    """Process every page image in a list."""
    return [preprocess_page(p) for p in pages]