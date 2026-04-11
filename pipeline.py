"""
pipeline.py — Document Sanitization Pipeline orchestrator.

Usage (programmatic)
--------------------
    from pipeline import SanitizationPipeline

    pipe = SanitizationPipeline()
    result = pipe.ingest("scan.pdf")

    # Auto-detected entities
    for ent in result.entities:
        print(ent.id, ent.tag, ent.text)

    # User adds a custom field (ACCOUNT NUMBER → learned, persisted)
    result.add_field("ACCOUNT NUMBER")

    # User redacts a one-off word (ephemeral, not persisted)
    result.redact_word("Confidential")

    # User deselects entity id=3 (won't be redacted)
    result.deselect(3)

    # Export
    out_path = result.export("scan_redacted.pdf")

Usage (CLI)
-----------
    python pipeline.py scan.pdf
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from models import WordRegistry, SensitiveEntity
from preprocessor import preprocess_pdf_pages
from ocr import extract_tokens, OCRToken, Backend
from detector import Detector, extract_field_value
from redactor import redact_page, preview_all_entities, export_pdf


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    """Configuration class for the sanitization pipeline."""
    
    def __init__(
        self,
        registry_path: str = "registry.json",
        ocr_backend: Backend = "tesseract",
        dpi: int = 300,
        conf_threshold: float = 0.40,
    ):
        self.registry_path = registry_path
        self.ocr_backend = ocr_backend
        self.dpi = dpi
        self.conf_threshold = conf_threshold


# ──────────────────────────────────────────────────────────────────────────────
# PDF → page images
# ──────────────────────────────────────────────────────────────────────────────

def _pdf_to_images(pdf_path: str, dpi: int = 300) -> list[np.ndarray]:
    """Rasterise each page of a PDF to a NumPy BGR image array."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        images = []
        for page in doc:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            images.append(img)
        return images
    except ImportError:
        pass

    # Fallback: pdf2image (requires poppler)
    from pdf2image import convert_from_path
    import cv2
    pil_pages = convert_from_path(pdf_path, dpi=dpi)
    return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pil_pages]


# ──────────────────────────────────────────────────────────────────────────────
# ProcessingResult
# ──────────────────────────────────────────────────────────────────────────────

class ProcessingResult:
    """
    Holds everything the pipeline produced for one document.

    Attributes
    ----------
    entities       : all auto-detected SensitiveEntity objects
    tokens         : all OCRTokens (the complete word registry input)
    raw_pages      : original page images (before redaction)
    registry       : the shared WordRegistry
    """

    def __init__(
        self,
        entities: list[SensitiveEntity],
        tokens: list[OCRToken],
        raw_pages: list[np.ndarray],
        registry: WordRegistry,
    ):
        self.entities = entities
        self.tokens = tokens
        self.raw_pages = raw_pages
        self.registry = registry
        self._entity_map: dict[int, SensitiveEntity] = {e.id: e for e in entities}

    # ── User actions ──────────────────────────────────────────────────────────

    def deselect(self, entity_id: int) -> None:
        """User chose NOT to redact this entity."""
        if entity_id in self._entity_map:
            self._entity_map[entity_id].redact = False

    def select(self, entity_id: int) -> None:
        """Re-select an entity for redaction."""
        if entity_id in self._entity_map:
            self._entity_map[entity_id].redact = True

    def add_field(self, field_label: str) -> list[SensitiveEntity]:
        """
        User identifies an unknown field label (e.g. "ACCOUNT NUMBER").

        1. Derive a tag from the label (upper-snake-case).
        2. Find the value token(s) adjacent to the label in the token stream.
        3. Teach the registry → persisted to disk.
        4. Create new SensitiveEntity entries for all occurrences.
        5. Return the new entities so the UI can display them.
        """
        tag = field_label.upper().replace(" ", "_")

        # Find the value token adjacent to the label
        value_tok = extract_field_value(self.tokens, field_label)

        new_entities: list[SensitiveEntity] = []
        next_id = max((e.id for e in self.entities), default=-1) + 1

        if value_tok:
            # Register the value word as the sensitive item
            self.registry.learn_field(value_tok.text, tag=tag)
            # Also mark every occurrence of this value across the document
            for tok in self.tokens:
                if tok.text.lower().strip() == value_tok.text.lower().strip():
                    ent = SensitiveEntity(
                        id=next_id, text=tok.text, tag=tag,
                        source="user_field", page=tok.page,
                        bbox=tok.bbox, confidence=tok.confidence,
                        redact=True,
                    )
                    self.entities.append(ent)
                    self._entity_map[next_id] = ent
                    next_id += 1
                    new_entities.append(ent)
        else:
            # Label found but no adjacent value — teach the label itself
            self.registry.learn_field(field_label, tag=tag)

        return new_entities

    def redact_word(self, word: str) -> list[SensitiveEntity]:
        """
        User types a specific word to redact (ephemeral — NOT persisted).
        All occurrences of that exact word in the current document are redacted.
        """
        new_entities: list[SensitiveEntity] = []
        next_id = max((e.id for e in self.entities), default=-1) + 1
        w_lower = word.lower().strip()

        for tok in self.tokens:
            if tok.text.lower().strip() == w_lower:
                ent = SensitiveEntity(
                    id=next_id, text=tok.text, tag="USER_WORD",
                    source="user_word", page=tok.page,
                    bbox=tok.bbox, confidence=tok.confidence,
                    redact=True,
                )
                self.entities.append(ent)
                self._entity_map[next_id] = ent
                next_id += 1
                new_entities.append(ent)

        # Mark in registry for this session (not persisted)
        self.registry.redact_word_session(word)
        return new_entities

    # ── Summary for the UI ────────────────────────────────────────────────────

    def summary(self) -> dict:
        """
        Return a dict suitable for the review UI, grouped by tag.
        {
          "EMAIL":  [{"id": 0, "text": "...", "page": 0, "redact": True}, …],
          "PERSON": […],
          …
        }
        """
        groups: dict[str, list[dict]] = {}
        for ent in sorted(self.entities, key=lambda e: (e.page, e.bbox[1])):
            groups.setdefault(ent.tag, []).append({
                "id":     ent.id,
                "text":   ent.text,
                "page":   ent.page,
                "source": ent.source,
                "redact": ent.redact,
                "reason": getattr(ent, "reason", ""),
            })
        return groups

    # ── Export ────────────────────────────────────────────────────────────────

    def export(self, output_path: str) -> Path:
        """
        Redact all selected entities and write the sanitised PDF to `output_path`.
        """
        # Build per-page bbox lists for selected entities only
        per_page: dict[int, list[tuple[int, int, int, int]]] = {}
        for ent in self.entities:
            if ent.redact:
                per_page.setdefault(ent.page, []).append(ent.bbox)

        redacted_pages: list[np.ndarray] = []
        for page_idx, page_img in enumerate(self.raw_pages):
            bboxes = per_page.get(page_idx, [])
            redacted_pages.append(redact_page(page_img, bboxes))

        return export_pdf(redacted_pages, output_path)

    def preview_page(self, page_idx: int) -> np.ndarray:
        """Return a preview image of `page_idx` with colour-coded overlays."""
        img = self.raw_pages[page_idx]
        entity_bboxes = [
            (ent.tag, ent.bbox)
            for ent in self.entities
            if ent.page == page_idx and ent.redact
        ]
        return preview_all_entities(img, entity_bboxes)

    def export_pdf(self, output_path: str) -> Path:
        """Alias for export method to maintain server compatibility."""
        return self.export(output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class SanitizationPipeline:
    """
    End-to-end document sanitization pipeline.

    Parameters
    ----------
    registry_path : path to the persistent registry JSON file
    ocr_backend   : "tesseract" | "easyocr" | "paddleocr"
    dpi           : rasterisation resolution
    conf_threshold: discard OCR tokens below this confidence
    """

    def __init__(
        self,
        registry_path: str = "registry.json",
        ocr_backend: Backend = "hybrid",
        dpi: int = 300,
        conf_threshold: float = 0.40,
        purpose: str = "",
    ):
        self.registry = WordRegistry(registry_path)
        self.ocr_backend = ocr_backend
        self.dpi = dpi
        self.conf_threshold = conf_threshold
        self.purpose = purpose
        self.website_context = {}

    def set_website_context(self, context: dict):
        """Set website context for intelligent PII redaction."""
        self.website_context = context
        # Also update purpose for backward compatibility
        if context.get('description'):
            self.purpose = context['description']

    def ingest(self, pdf_path: str) -> ProcessingResult:
        """
        Full pipeline: ingest → preprocess → OCR → detect.
        Returns a ProcessingResult ready for user interaction.
        """
        # 1. Reset per-document registry state
        self.registry.reset_session()

        # 2. Rasterise
        print(f"[1/4] Rasterising {pdf_path} …")
        raw_pages = _pdf_to_images(pdf_path, dpi=self.dpi)

        # 3. Preprocess
        print(f"[2/4] Preprocessing {len(raw_pages)} page(s) …")
        processed_pages = preprocess_pdf_pages(raw_pages)

        # 4. OCR
        print(f"[3/4] Running OCR ({self.ocr_backend}) …")
        tokens = extract_tokens(
            processed_pages,
            backend=self.ocr_backend,
            conf_threshold=self.conf_threshold,
        )
        print(f"      → {len(tokens)} tokens extracted")

        # 5. Detect
        print("[4/4] Detecting PII and entities …")
        detector = Detector(self.registry, purpose=self.purpose, website_context=self.website_context)
        entities = detector.detect(tokens)
        print(f"      → {len(entities)} sensitive entities found")

        return ProcessingResult(
            entities=entities,
            tokens=tokens,
            raw_pages=raw_pages,
            registry=self.registry,
        )

    def process_document(self, pdf_path: str) -> ProcessingResult:
        """Process document with current website context settings."""
        return self.ingest(pdf_path)


# ──────────────────────────────────────────────────────────────────────────────
# Simple CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cli():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <input.pdf> [output.pdf]")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else input_pdf.replace(".pdf", "_redacted.pdf")

    pipe = SanitizationPipeline()
    result = pipe.ingest(input_pdf)

    # ── Review loop ───────────────────────────────────────────────────────────
    print("\n─── Detected sensitive entities ───")
    summary = result.summary()
    for tag, items in summary.items():
        print(f"\n  [{tag}]")
        for item in items:
            print(f"    #{item['id']:3d}  page {item['page']+1}  \"{item['text']}\"  (source: {item['source']})")

    print("\n─── User actions ───")
    print("Commands:")
    print("  deselect <id>          — don't redact this entity")
    print("  field <FIELD LABEL>    — teach system a new field (persistent)")
    print("  word <word>            — redact a word this session only")
    print("  done                   — export and exit")
    print()

    while True:
        try:
            cmd = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        if cmd == "done":
            break

        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if action == "deselect":
            try:
                eid = int(arg)
                result.deselect(eid)
                print(f"  Entity #{eid} deselected.")
            except ValueError:
                print("  Usage: deselect <id>")

        elif action == "select":
            try:
                eid = int(arg)
                result.select(eid)
                print(f"  Entity #{eid} selected for redaction.")
            except ValueError:
                print("  Usage: select <id>")

        elif action == "field":
            if not arg:
                print("  Usage: field ACCOUNT NUMBER")
                continue
            new_ents = result.add_field(arg)
            print(f"  Learned field '{arg}' → {len(new_ents)} value(s) will be redacted.")
            print(f"  This field is now saved and will auto-detect in future documents.")

        elif action == "word":
            if not arg:
                print("  Usage: word Confidential")
                continue
            new_ents = result.redact_word(arg)
            print(f"  Word '{arg}' → {len(new_ents)} occurrence(s) will be redacted (this session only).")

        else:
            print(f"  Unknown command: {action}")

    # ── Export ────────────────────────────────────────────────────────────────
    print(f"\nExporting redacted PDF to {output_pdf} …")
    out = result.export(output_pdf)
    print(f"Done → {out}")


if __name__ == "__main__":
    _cli()