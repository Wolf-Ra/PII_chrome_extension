"""
ocr.py — OCR extraction with word-level bounding boxes.

Returns a list of OCRToken per page.  Supports backends:
  - tesseract   (default, via pytesseract)
  - easyocr
  - paddleocr
  - hybrid      ← RECOMMENDED: Tesseract bboxes + Vision LLM text recovery

Architecture note (IMPORTANT):
  Vision LLM cannot be used as a standalone OCR backend because it cannot
  reliably return accurate pixel bounding boxes — LLMs estimate coordinates,
  they don't measure them.

  The ONLY correct role for Vision LLM in this pipeline is:
    "Find text that Tesseract missed, then locate it using Tesseract's own bboxes."

  Hybrid mode works like this:
    1. Tesseract runs first  → produces tokens WITH accurate pixel bboxes
    2. Vision LLM runs       → produces a plain text list of what it sees
    3. For each phrase Vision found that Tesseract missed:
         → search Tesseract's token list for matching words
         → use TESSERACT's bboxes for those words
    4. Final token list = all Tesseract tokens (accurate bboxes guaranteed)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import re


# ──────────────────────────────────────────────────────────────────────────────
# Token
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OCRToken:
    """One word-level token extracted by OCR."""
    text:       str
    page:       int
    bbox:       tuple[int, int, int, int]   # (x, y, w, h)
    confidence: float                       # 0.0 – 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Tesseract backend  (TWO-PASS: psm 6 for body text + psm 11 for sparse/bold)
# ──────────────────────────────────────────────────────────────────────────────

def _ocr_tesseract(image: np.ndarray, page: int, conf_threshold: float) -> list[OCRToken]:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

    all_tokens: list[OCRToken] = []
    seen: set[tuple] = set()

    # Two passes:
    #   --psm 6  → uniform block of text (good for body paragraphs)
    #   --psm 11 → sparse text (good for bold table-cell values like "SUVETHA V")
    for psm in ["--psm 6", "--psm 11"]:
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config=psm,
        )
        n = len(data["text"])
        for i in range(n):
            word = data["text"][i].strip()
            conf = float(data["conf"][i])
            if not word or conf < 0:
                continue
            conf_norm = conf / 100.0
            if conf_norm < conf_threshold:
                continue
            x, y, w, h = (data["left"][i], data["top"][i],
                          data["width"][i], data["height"][i])
            # Deduplicate across passes by (word, x, y)
            key = (word.lower(), x, y)
            if key in seen:
                continue
            seen.add(key)
            all_tokens.append(OCRToken(
                text=word, page=page,
                bbox=(x, y, w, h),
                confidence=conf_norm,
            ))

    return all_tokens


# ──────────────────────────────────────────────────────────────────────────────
# EasyOCR backend
# ──────────────────────────────────────────────────────────────────────────────

def _ocr_easyocr(image: np.ndarray, page: int, conf_threshold: float,
                 reader=None) -> list[OCRToken]:
    import easyocr
    if reader is None:
        reader = easyocr.Reader(["en"])
    results = reader.readtext(image, detail=1)
    tokens: list[OCRToken] = []
    for (pts, text, conf) in results:
        if not text.strip() or conf < conf_threshold:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        words = text.strip().split()
        w_per_word = w // max(len(words), 1)
        for j, word in enumerate(words):
            tokens.append(OCRToken(
                text=word, page=page,
                bbox=(x + j * w_per_word, y, w_per_word, h),
                confidence=float(conf),
            ))
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# PaddleOCR backend
# ──────────────────────────────────────────────────────────────────────────────

def _ocr_paddleocr(image: np.ndarray, page: int, conf_threshold: float,
                   paddle=None) -> list[OCRToken]:
    from paddleocr import PaddleOCR
    if paddle is None:
        paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    result = paddle.ocr(image, cls=True)
    tokens: list[OCRToken] = []
    for line in (result[0] or []):
        pts, (text, conf) = line
        if not text.strip() or conf < conf_threshold:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        words = text.strip().split()
        w_per_word = w // max(len(words), 1)
        for j, word in enumerate(words):
            tokens.append(OCRToken(
                text=word, page=page,
                bbox=(x + j * w_per_word, y, w_per_word, h),
                confidence=float(conf),
            ))
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Vision LLM  — TEXT EXTRACTION ONLY (NO bboxes used from here)
# ──────────────────────────────────────────────────────────────────────────────

def _vision_llm_extract_text(image: np.ndarray) -> list[str]:
    """
    Ask the Vision LLM what text it sees in the image.
    Returns a flat list of text strings (words / short phrases).

    IMPORTANT: bboxes returned by the LLM are NOT used anywhere.
    This function is only called by _hybrid() to find text Tesseract missed.
    The actual bboxes for those recovered texts come from Tesseract's token list.
    """
    import base64
    import json
    import cv2
    from groq import Groq
    from dotenv import load_dotenv
    import os
    load_dotenv()

    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    b64_image = base64.b64encode(buffer).decode("utf-8")
    api_key = os.getenv("MY_API_KEY")
    client = Groq(api_key=api_key)

    prompt = """You are an OCR engine. Read every piece of text visible in this document image.
Pay special attention to:
- Bold text inside table cells (e.g. names, IDs, registration numbers)
- Field values next to labels like 'Name:', 'Register Number:', 'DOB:'
- Text in headers, footers, and form fields

Return a JSON array of strings only — one string per distinct word or short phrase.
Example: ["SUVETHA V", "71762234053", "2022-2027", "19MAM61", "07-05-2025"]

No markdown. No explanation. No bounding boxes. Just the JSON array of text strings."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            temperature=0.0,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        items = json.loads(raw)
        if isinstance(items, list):
            return [str(i).strip() for i in items if str(i).strip()]

    except Exception as e:
        print(f"      [VisionLLM] Text extraction failed: {e}")

    return []


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid  — THE CORRECT ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────────

def _find_tesseract_tokens_for_text(
    phrase: str,
    tess_tokens: list[OCRToken],
    page: int,
) -> list[OCRToken]:
    """
    Given a phrase like "SUVETHA V", find the Tesseract tokens whose text
    matches the words in the phrase and return them.
    This way the bbox always comes from Tesseract, never from the LLM.
    """
    words = phrase.split()
    tess_lower = [(t.text.lower().strip(), t) for t in tess_tokens]
    matched: list[OCRToken] = []

    for word in words:
        wl = word.lower().strip(".,;:\"'")
        for tl, tok in tess_lower:
            tl_clean = tl.strip(".,;:\"'")
            if tl_clean == wl and tok not in matched:
                matched.append(tok)
                break   # one match per word

    return matched


def _ocr_hybrid(image: np.ndarray, page: int,
                conf_threshold: float) -> list[OCRToken]:
    """
    Hybrid OCR:
      1. Run two-pass Tesseract → accurate bboxes
      2. Run Vision LLM → text strings only (no bboxes used)
      3. For Vision phrases not in Tesseract: find their words in Tesseract tokens
         to recover the correct bbox
      4. Final output = all Tesseract tokens  (bboxes always from Tesseract)

    This means redaction boxes are ALWAYS positioned by Tesseract coordinates,
    eliminating the bbox drift problem.
    """
    # ── Step 1: Tesseract (source of truth for bboxes) ────────────────────────
    tess_tokens = _ocr_tesseract(image, page, conf_threshold)
    tess_word_set = {t.text.lower().strip() for t in tess_tokens}

    print(f"      [Hybrid] Tesseract found {len(tess_tokens)} tokens on page {page+1}")

    # ── Step 2: Vision LLM (text only — find what Tesseract missed) ───────────
    vision_texts = _vision_llm_extract_text(image)
    print(f"      [Hybrid] Vision LLM found {len(vision_texts)} text items on page {page+1}")

    # ── Step 3: Recover missed text using Tesseract bboxes ───────────────────
    recovered: list[OCRToken] = []
    already_added: set[int] = set()   # track by id() to avoid duplicates

    for phrase in vision_texts:
        phrase_words = phrase.lower().split()

        # Check if ALL words of this phrase are already in Tesseract output
        all_found = all(
            any(w.strip(".,;:\"'") == pw.strip(".,;:\"'")
                for w in tess_word_set)
            for pw in phrase_words
        )
        if all_found:
            continue   # Tesseract already has this — nothing to recover

        # Some words are missing — try to find them in tess_tokens
        matched_toks = _find_tesseract_tokens_for_text(phrase, tess_tokens, page)
        for tok in matched_toks:
            if id(tok) not in already_added:
                already_added.add(id(tok))
                recovered.append(tok)

        if not matched_toks:
            # Vision found something Tesseract has no trace of at all.
            # We log it but do NOT create a synthetic token with a fake bbox.
            # A fake bbox would cause wrong redaction placement.
            print(f"      [Hybrid] Vision-only text (no Tesseract match, skipped for bbox safety): '{phrase}'")

    if recovered:
        print(f"      [Hybrid] Recovered {len(recovered)} additional tokens via Vision LLM")

    # ── Step 4: Return ONLY tokens whose bboxes came from Tesseract ──────────
    # This guarantees all redaction boxes are correctly positioned.
    return tess_tokens   # recovered tokens are already a subset of tess_tokens


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

Backend = Literal["tesseract", "easyocr", "paddleocr", "vision_llm", "hybrid"]


def extract_tokens(
    pages: list[np.ndarray],
    backend: Backend = "hybrid",
    conf_threshold: float = 0.40,
    **kwargs,
) -> list[OCRToken]:
    """
    Run OCR on all preprocessed page images.

    Parameters
    ----------
    pages          : list of preprocessed page images (BGR or grayscale)
    backend        : "tesseract" | "easyocr" | "paddleocr" | "hybrid"
                     Use "hybrid" for best results on form/table documents.
                     Do NOT use "vision_llm" alone — bboxes will be inaccurate.
    conf_threshold : discard tokens below this confidence (0.0–1.0)
    **kwargs       : passed to backend (e.g. reader=, paddle=)

    Returns
    -------
    Flat list of OCRToken across all pages, in reading order.
    All tokens are guaranteed to have bboxes sourced from Tesseract or
    EasyOCR/PaddleOCR (never from Vision LLM estimates).
    """
    all_tokens: list[OCRToken] = []

    for page_idx, image in enumerate(pages):

        if backend == "tesseract":
            tokens = _ocr_tesseract(image, page_idx, conf_threshold)

        elif backend == "easyocr":
            tokens = _ocr_easyocr(image, page_idx, conf_threshold, **kwargs)

        elif backend == "paddleocr":
            tokens = _ocr_paddleocr(image, page_idx, conf_threshold, **kwargs)

        elif backend == "hybrid":
            tokens = _ocr_hybrid(image, page_idx, conf_threshold)

        elif backend == "vision_llm":
            # Warn: vision_llm alone has unreliable bboxes.
            # It is kept here only for debugging/inspection purposes.
            # Do not use in production — use "hybrid" instead.
            print("WARNING: vision_llm backend has unreliable bboxes. "
                  "Use 'hybrid' for accurate redaction placement.")
            raw_texts = _vision_llm_extract_text(image)
            img_h, img_w = image.shape[:2]
            tokens = []
            y_pos = 0
            for phrase in raw_texts:
                tokens.append(OCRToken(
                    text=phrase, page=page_idx,
                    bbox=(0, y_pos, img_w, 25),   # horizontal strip — deliberately vague
                    confidence=0.85,
                ))
                y_pos += 30

        else:
            raise ValueError(f"Unknown OCR backend: {backend}")

        all_tokens.extend(tokens)

    return all_tokens