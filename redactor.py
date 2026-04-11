"""
redactor.py — Apply redaction bboxes to page images and export a clean PDF.

Two modes
---------
preview_page(image, bboxes)
    Returns an annotated copy of the page image with coloured highlight boxes —
    for the UI review step.  Nothing is permanently erased.

redact_page(image, bboxes)
    Returns a copy of the page image with solid black rectangles painted over
    every sensitive bbox.  The text layer is destroyed — no text can be recovered.

export_pdf(pages, output_path)
    Compose the list of redacted page images into a final PDF using fpdf2.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Tag → colour mapping  (BGR for OpenCV)
# Used in the preview overlay only
# ──────────────────────────────────────────────────────────────────────────────

_TAG_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "EMAIL":                  (34,  139, 34),    # green
    "PHONE":                  (0,   140, 255),   # orange
    "SSN":                    (60,  20,  220),   # crimson
    "CREDIT_CARD":            (211, 0,   148),   # purple
    "AADHAAR":                (128, 128, 0),     # teal
    "PAN":                    (200, 100, 0),     # blue
    "PERSON":                 (60,  20,  220),   # crimson
    "ORG":                    (0,   165, 255),   # amber
    "GPE":                    (237, 149, 100),   # cornflower
    "MONEY":                  (50,  205, 50),    # lime
    "LAW":                    (133, 21,  199),   # medvioletred
    "ACCOUNT_NUMBER":         (255, 191, 0),     # deep sky blue
    "SENSITIVE":              (128, 0,   128),   # purple (fallback)
    "ADDRESS":                (11,  134, 184),   # dark goldenrod
    "DOB":                    (147, 20,  255),   # deep pink
    "PASSPORT":               (255, 144, 30),    # dodger blue
    "DRIVER_LICENSE":         (170, 178, 32),    # light sea green
    "IP_ADDRESS":             (71,  99,  255),   # tomato
    "MEDICAL_HISTORY":        (0,   0,   139),   # dark red
    "CORPORATE_CONFIDENTIAL": (130, 0,   75),    # indigo
}

_DEFAULT_COLOR = (128, 128, 128)   # grey for unknown tags


def _color_for(tag: str) -> tuple[int, int, int]:
    return _TAG_COLORS_BGR.get(tag.upper(), _DEFAULT_COLOR)


# ──────────────────────────────────────────────────────────────────────────────
# Preview  (coloured transparent overlay)
# ──────────────────────────────────────────────────────────────────────────────

def preview_page(
    image: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    tag: str = "SENSITIVE",
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Overlay semi-transparent coloured rectangles over bboxes.
    Returns a copy — original is not modified.
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    color = _color_for(tag)
    overlay = vis.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)

    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    # Thin border for clarity
    for (x, y, w, h) in bboxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)

    return vis


def preview_all_entities(
    image: np.ndarray,
    entity_bboxes: list[tuple[str, tuple[int, int, int, int]]],
) -> np.ndarray:
    """
    Preview with per-tag colours.
    `entity_bboxes` is a list of (tag, bbox) tuples.
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    overlay = vis.copy()
    for tag, (x, y, w, h) in entity_bboxes:
        color = _color_for(tag)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
    for tag, (x, y, w, h) in entity_bboxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), _color_for(tag), 1)
    return vis


# ──────────────────────────────────────────────────────────────────────────────
# Redact  (permanent black boxes)
# ──────────────────────────────────────────────────────────────────────────────

def redact_page(
    image: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    padding: int = 3,
) -> np.ndarray:
    """
    Paint solid black over every bbox on the image.
    `padding` expands each box slightly to cover edge pixels.
    Returns a copy.
    """
    out = image.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    h_img, w_img = out.shape[:2]
    for (x, y, w, h) in bboxes:
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), -1)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Export redacted pages → PDF
# ──────────────────────────────────────────────────────────────────────────────

def export_pdf(
    pages: list[np.ndarray],
    output_path: str,
    dpi: int = 300,
) -> Path:
    """
    Compose a list of BGR page images into a single PDF.
    Uses fpdf2 for clean, non-recoverable output.
    """
    from fpdf import FPDF
    import tempfile, os

    pdf = FPDF(unit="pt")
    tmp_files = []

    try:
        for i, page_img in enumerate(pages):
            # Save page as temporary PNG using system temp directory
            import tempfile
            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, f"_redacted_page_{i}.png")
            tmp_files.append(tmp_path)
            cv2.imwrite(tmp_path, page_img)

            h_px, w_px = page_img.shape[:2]
            # Convert pixels → points  (1 inch = 72 pt; dpi pixels per inch)
            w_pt = w_px * 72.0 / dpi
            h_pt = h_px * 72.0 / dpi

            pdf.add_page(format=(w_pt, h_pt))
            pdf.image(tmp_path, x=0, y=0, w=w_pt, h=h_pt)

        output_path = str(output_path)
        pdf.output(output_path)
    finally:
        for tmp in tmp_files:
            try:
                os.remove(tmp)
            except OSError:
                pass

    return Path(output_path)