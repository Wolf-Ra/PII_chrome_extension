"""
models.py — Core data structures for the Document Sanitization Pipeline.

WordRegistry  : persistent per-word store  (saved to registry.json)
SensitiveEntity: one detected hit on a page
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Occurrence  —  one place a word appears in a document
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Occurrence:
    """
    A single location where a word token appears on a page.

    bbox  : (x, y, w, h) in page-pixel coordinates as returned by OCR.
    tag   : PII tag if assigned ("EMAIL", "PHONE", "PERSON", "ACCOUNT_NUMBER", …)
            or None for plain-word redaction requests.
    sensitive : True  → always redact (field learned from a previous session).
                False → seen before but user chose NOT to redact it.
    page  : 0-based page index inside the current document.
    """
    bbox: tuple[int, int, int, int]
    page: int
    tag: Optional[str] = None
    sensitive: bool = False

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "page": self.page,
            "tag": self.tag,
            "sensitive": self.sensitive,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Occurrence":
        return cls(
            bbox=tuple(d["bbox"]),
            page=d["page"],
            tag=d.get("tag"),
            sensitive=d.get("sensitive", False),
        )


# ──────────────────────────────────────────────────────────────────────────────
# WordEntry  —  everything the system knows about one word/phrase key
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WordEntry:
    """
    Registry record for a single normalised word (lower-stripped).

    occurrences   : list of every place this word has appeared so far in the
                    *current* processing session.  Cleared between documents.
    is_learned_field : True when the user explicitly taught the system that
                        this field label (e.g. "ACCOUNT NUMBER") is sensitive.
                        These are persisted to disk so the next document benefits.
    """
    word: str
    occurrences: list[Occurrence] = field(default_factory=list)
    is_learned_field: bool = False      # persisted
    default_tag: Optional[str] = None  # persisted ("ACCOUNT_NUMBER", etc.)

    # ── occurrence helpers ────────────────────────────────────────────────────

    def add_occurrence(self, occ: Occurrence) -> None:
        self.occurrences.append(occ)

    def mark_all_sensitive(self, tag: Optional[str] = None) -> None:
        for occ in self.occurrences:
            occ.sensitive = True
            if tag:
                occ.tag = tag

    def sensitive_bboxes(self) -> list[tuple[int, tuple[int, int, int, int]]]:
        """Return (page, bbox) for every occurrence flagged sensitive."""
        return [(o.page, o.bbox) for o in self.occurrences if o.sensitive]

    def all_bboxes(self) -> list[tuple[int, tuple[int, int, int, int]]]:
        """Return (page, bbox) for every occurrence regardless of sensitivity."""
        return [(o.page, o.bbox) for o in self.occurrences]

    # ── serialisation (persisted fields only) ────────────────────────────────

    def to_persist_dict(self) -> dict:
        """Only save fields that should survive across documents."""
        return {
            "word": self.word,
            "is_learned_field": self.is_learned_field,
            "default_tag": self.default_tag,
        }

    @classmethod
    def from_persist_dict(cls, d: dict) -> "WordEntry":
        return cls(
            word=d["word"],
            is_learned_field=d.get("is_learned_field", False),
            default_tag=d.get("default_tag"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# WordRegistry  —  the in-memory + on-disk store
# ──────────────────────────────────────────────────────────────────────────────

class WordRegistry:
    """
    Central word store.

    Layout
    ------
    _store : dict[str, WordEntry]
        Key   = normalised word  (lower().strip())
        Value = WordEntry

    Persistence
    -----------
    Only entries with  is_learned_field=True  are written to disk.
    Ephemeral user-typed redaction words live in _store during the session
    but are never flushed to registry.json.

    Usage
    -----
    registry = WordRegistry("registry.json")
    registry.register_token("john", page=0, bbox=(10,20,60,14), tag="PERSON", sensitive=True)
    registry.learn_field("account number", tag="ACCOUNT_NUMBER")
    registry.save()
    """

    def __init__(self, path: str = "registry.json"):
        self._path = Path(path)
        self._store: dict[str, WordEntry] = {}
        self._load()

    # ── private ───────────────────────────────────────────────────────────────

    def _key(self, word: str) -> str:
        return word.lower().strip()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path) as f:
                data = json.load(f)
            for entry_dict in data.get("learned_fields", []):
                entry = WordEntry.from_persist_dict(entry_dict)
                self._store[self._key(entry.word)] = entry

    # ── public API ────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Flush only learned-field entries to disk."""
        learned = [
            e.to_persist_dict()
            for e in self._store.values()
            if e.is_learned_field
        ]
        with open(self._path, "w") as f:
            json.dump({"learned_fields": learned}, f, indent=2)

    def register_token(
        self,
        word: str,
        page: int,
        bbox: tuple[int, int, int, int],
        tag: Optional[str] = None,
        sensitive: bool = False,
    ) -> WordEntry:
        """
        Record one OCR token into the registry.
        If the word is already a learned field, mark it sensitive immediately.
        """
        k = self._key(word)
        if k not in self._store:
            self._store[k] = WordEntry(word=k)

        entry = self._store[k]

        # Inherit learned field defaults
        if entry.is_learned_field:
            tag = tag or entry.default_tag
            sensitive = True

        occ = Occurrence(bbox=bbox, page=page, tag=tag, sensitive=sensitive)
        entry.add_occurrence(occ)
        return entry

    def get(self, word: str) -> Optional[WordEntry]:
        return self._store.get(self._key(word))

    def get_or_create(self, word: str) -> WordEntry:
        k = self._key(word)
        if k not in self._store:
            self._store[k] = WordEntry(word=k)
        return self._store[k]

    def learn_field(self, field_label: str, tag: str) -> None:
        """
        Teach the system that `field_label` is always sensitive.
        Persisted to disk.  All current-session occurrences are flagged now.
        """
        k = self._key(field_label)
        entry = self.get_or_create(field_label)
        entry.is_learned_field = True
        entry.default_tag = tag
        entry.mark_all_sensitive(tag)
        self.save()

    def redact_word_session(self, word: str) -> list[tuple[int, tuple]]:
        """
        Ephemeral redaction: mark every occurrence of `word` sensitive for
        this session only.  NOT persisted.
        Returns list of (page, bbox) to redact immediately.
        """
        k = self._key(word)
        entry = self.get(word)
        if entry is None:
            return []
        entry.mark_all_sensitive()
        return entry.all_bboxes()

    def sensitive_bboxes_for(self, word: str) -> list[tuple[int, tuple]]:
        entry = self.get(word)
        if entry is None:
            return []
        return entry.sensitive_bboxes()

    def all_sensitive_bboxes(self) -> list[tuple[int, tuple]]:
        """Collect every sensitive bbox across all words — used at final export."""
        result = []
        for entry in self._store.values():
            result.extend(entry.sensitive_bboxes())
        return result

    def learned_fields(self) -> list[WordEntry]:
        return [e for e in self._store.values() if e.is_learned_field]

    def reset_session(self) -> None:
        """Clear per-document occurrences; keep learned-field metadata."""
        for entry in self._store.values():
            entry.occurrences = []


# ──────────────────────────────────────────────────────────────────────────────
# SensitiveEntity  —  one detected entity returned to the UI
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SensitiveEntity:
    """
    A fully resolved detection ready for the review UI.

    id          : sequential index 0…n within the document
    text        : the matched surface form
    tag         : "EMAIL" | "PHONE" | "SSN" | "PERSON" | "ORG" | "ACCOUNT_NUMBER" | …
    source      : "regex" | "ner" | "learned_field" | "user_field" | "user_word"
    page        : 0-based page index
    bbox        : (x, y, w, h) pixel coords on the rasterised page image
    confidence  : OCR confidence 0.0–1.0
    redact      : True → will be redacted (default for auto-detected items)
    """
    id: int
    text: str
    tag: str
    source: str
    page: int
    bbox: tuple[int, int, int, int]
    confidence: float = 1.0
    redact: bool = True
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)