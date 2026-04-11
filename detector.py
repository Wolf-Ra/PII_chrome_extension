from __future__ import annotations
import re
import os
import json
from datetime import datetime
from typing import Optional
from groq import Groq

from models import SensitiveEntity, WordRegistry, Occurrence
from ocr import OCRToken


# ──────────────────────────────────────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union of two (x, y, w, h) bboxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _deduplicate(entities: list[SensitiveEntity]) -> list[SensitiveEntity]:
    """Remove entities whose bboxes overlap > 0.5 IoU on the same page."""
    kept: list[SensitiveEntity] = []
    for ent in entities:
        overlap = any(
            k.page == ent.page and _iou(k.bbox, ent.bbox) > 0.5
            for k in kept
        )
        if not overlap:
            kept.append(ent)
    return kept


# ──────────────────────────────────────────────────────────────────────────────
# Main detector
# ──────────────────────────────────────────────────────────────────────────────

class Detector:
    """
    Detects sensitive entities from a list of OCRTokens.
    Also registers every token into the WordRegistry for fast lookup.
    """

    def __init__(self, registry: WordRegistry, purpose: str = "", website_context: dict = None):
        self.registry = registry
        self.purpose = purpose
        self.website_context = website_context or {}

    # ── LLM Extraction ────────────────────────────────────────────────────────

    def _run_llm(self, tokens: list[OCRToken]) -> list[SensitiveEntity]:

        # ── Debug log (always runs, even if API key is missing) ───────────────
        log_path = r"d:\PII_FLAG\pii_debug.log"

        def _write_log(msg: str) -> None:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_path, "a", encoding="utf-8") as _f:
                _f.write(f"{ts}  {msg}\n")

        _write_log("=" * 70)
        _write_log(f"_run_llm called  |  tokens={len(tokens)}  |  purpose='{self.purpose or 'None'}'")

        api_key = "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"
        if not api_key:
            _write_log("ERROR: GROQ_API_KEY not set — aborting LLM detection")
            print("Warning: GROQ_API_KEY not found. Skipping LLM detection.")
            return []

        if self.purpose:
            print(f"      → Context/Purpose provided to Detector: '{self.purpose}'")
        else:
            print("      → No Context/Purpose provided. Using strict default rules.")
        client = Groq(api_key=api_key)
        _write_log(f"Groq client ready  |  Purpose: '{self.purpose or 'None'}'")
        _write_log("=" * 70)


        by_page: dict[int, list[OCRToken]] = {}
        for tok in tokens:
            by_page.setdefault(tok.page, []).append(tok)
            # Register tokens to ensure complete word index
            self.registry.register_token(
                tok.text, tok.page, tok.bbox,
                tag=None, sensitive=False,
            )

        entities = []
        for page_idx, page_tokens in by_page.items():
            full_text = " ".join(t.text for t in page_tokens)

            # ── Build system prompt ──────────────────────────────────────────
            system_prompt = (
                "You are an expert, highly-precise data privacy auditor. "
                "Your job is to read OCR-extracted text from documents and find EVERY piece of sensitive information. "
                "You must be thorough and have a ZERO miss rate — it is far better to over-flag than to miss anything. "
                "Always return ONLY a raw JSON array. No markdown, no code fences, no explanation outside the array."
            )

            # ── Category definitions shared across both modes ─────────────────
            categories = (
                "Use EXACTLY these category tags:\n"
                "  PERSON           — Full name, first name, last name, initials of any individual\n"
                "  EMAIL            — Any email address\n"
                "  PHONE            — Any phone or mobile number, with or without country code\n"
                "  ADDRESS          — Street address, city, state, postal/ZIP code, country\n"
                "  SSN              — Social Security Number or national ID\n"
                "  CREDIT_CARD      — Credit/debit card numbers\n"
                "  AADHAAR          — Indian Aadhaar numbers\n"
                "  PAN              — Indian PAN card numbers\n"
                "  ACCOUNT_NUMBER   — Bank account or financial account numbers\n"
                "  DOB              — Date of birth\n"
                "  MEDICAL_HISTORY  — Diagnosis, medications, clinical notes, test results\n"
                "  ORG              — Company, institution, or organisation name\n"
                "  GPE              — Specific geopolitical location (city, state, country)\n"
                "  MONEY            — Salary, financial amounts, transaction values\n"
                "  IP_ADDRESS       — IPv4 or IPv6 addresses\n"
                "  PASSPORT         — Passport numbers\n"
                "  DRIVER_LICENSE   — Driver's licence numbers\n"
                "  LAW              — Legal case numbers, statutes referred to\n"
                "  CORPORATE_CONFIDENTIAL — Trade secrets, internal codes, unreleased product names\n"
                "  SENSITIVE        — Anything else that is clearly private or confidential\n"
            )

            # ── Build user prompt based on whether purpose and website context are given ────────────
            context_info = ""
            if self.website_context:
                website_type = self.website_context.get('website_type', 'unknown')
                industry = self.website_context.get('industry', 'unknown')
                primary_pii = self.website_context.get('primary_pii_types', [])
                sensitivity = self.website_context.get('sensitivity_level', 'medium')
                
                context_info = (
                    f"WEBSITE CONTEXT:\n"
                    f"- Website Type: {website_type}\n"
                    f"- Industry: {industry}\n"
                    f"- Sensitivity Level: {sensitivity}\n"
                    f"- Primary PII Types Expected: {', '.join(primary_pii) if primary_pii else 'None specified'}\n"
                    f"- Description: {self.website_context.get('description', 'No description available')}\n\n"
                )
                
                # Add context-specific instructions
                if website_type == 'healthcare':
                    context_info += (
                        "HEALTHCARE-SPECIFIC INSTRUCTIONS:\n"
                        "- Pay special attention to: medical_records, patient_id, insurance_info, medical_history\n"
                        "- Patient names, DOB, and contact info should be redacted unless clearly required\n"
                        "- Medical provider names may be kept if relevant to medical context\n"
                    )
                elif website_type == 'financial':
                    context_info += (
                        "FINANCIAL-SPECIFIC INSTRUCTIONS:\n"
                        "- Pay special attention to: account_numbers, credit_card, transaction_history, ssn\n"
                        "- Bank names and financial institution names may be kept if relevant\n"
                        "- Account holder names should be redacted unless required for transaction\n"
                    )
                elif website_type == 'education':
                    context_info += (
                        "EDUCATION-SPECIFIC INSTRUCTIONS:\n"
                        "- Pay special attention to: academic_records, student_id, grades, ssn\n"
                        "- School/university names may be kept if relevant\n"
                        "- Student names should be redacted unless required for academic records\n"
                    )
                
                context_info += "\n"

            if self.purpose or self.website_context:
                user_prompt = (
                    f"{context_info}"
                    f"PURPOSE OF UPLOAD: \"{self.purpose or 'Document upload'}\"\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Read the text carefully.\n"
                    f"2. Identify every piece of information that belongs to the categories below.\n"
                    f"3. For each identified item, determine STRICTLY whether it is NECESSARY for the stated purpose above.\n"
                    f"   - If it IS necessary for the stated purpose → do NOT include it in the output.\n"
                    f"   - If it is NOT necessary for the stated purpose → include it for redaction.\n"
                    f"4. Use the website context to prioritize PII types that are most relevant for this site type.\n"
                    f"5. Use the purpose as the PRIMARY filter. Do not use general PII judgment when a purpose is given.\n"
                    f"6. Be exhaustive — scan every word. Never skip a potential match.\n"
                    f"7. If a sensitive ID, hash, or code appears fragmented by spaces or OCR artifacts, extract the entire sequence encompassing all fragments.\n\n"
                    f"{categories}\n"
                    f"TEXT TO ANALYSE:\n{full_text}\n\n"
                    f"Return a JSON OBJECT with exactly two keys:\n"
                    f"  \"flagged\": array of items TO REDACT. Each item has:\n"
                    f"    \"text\"   — the EXACT substring as it appears in the text\n"
                    f"    \"tag\"    — one of the category tags listed above\n"
                    f"    \"reason\" — why this is sensitive AND why it is not needed for the stated purpose\n"
                    f"  \"skipped\": array of items that LOOK potentially sensitive but were NOT flagged because they are necessary for the stated purpose. Each item has:\n"
                    f"    \"text\"   — the exact substring\n"
                    f"    \"reason\" — one sentence: why it looks sensitive but is KEPT given the purpose\n"
                    f"Return ONLY the raw JSON object. No markdown, no code fences."
                )
            else:
                user_prompt = (
                    f"INSTRUCTIONS:\n"
                    f"1. Read the text extremely carefully, word by word.\n"
                    f"2. Extract EVERY piece of sensitive or private information you can identify.\n"
                    f"3. Do NOT skip anything — names, addresses, numbers, medical details, financial data, identifiers — flag them ALL.\n"
                    f"4. When in doubt, include it. A false positive is acceptable; a false negative is not.\n"
                    f"5. Each occurrence of a sensitive value must be listed separately if it appears multiple times.\n"
                    f"6. If a sensitive ID, hash, or code appears fragmented by spaces or OCR artifacts, extract the entire sequence encompassing all fragments.\n\n"
                    f"{categories}\n"
                    f"TEXT TO ANALYSE:\n{full_text}\n\n"
                    f"Return a JSON OBJECT with exactly two keys:\n"
                    f"  \"flagged\": array of items TO REDACT. Each item has:\n"
                    f"    \"text\"   — the EXACT substring as it appears in the text\n"
                    f"    \"tag\"    — one of the category tags listed above\n"
                    f"    \"reason\" — one sentence explaining why this is sensitive private information\n"
                    f"  \"skipped\": array of words/phrases that you CONSIDERED as potentially sensitive but DECIDED not to flag. Each item has:\n"
                    f"    \"text\"   — the exact word or phrase you considered\n"
                    f"    \"reason\" — one sentence explaining why you decided it is NOT sensitive\n"
                    f"Return ONLY the raw JSON object. No markdown, no code fences."
                )

            print(f"      → Sending page {page_idx + 1} text ({len(full_text)} chars) to LLM for analysis…")

            _write_log(f"Calling LLM for page {page_idx + 1} ({len(full_text)} chars)...")

            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                )
                content = response.choices[0].message.content.strip()
                _write_log(f"LLM raw response (first 500 chars): {content[:500]}")

                # ── Robust JSON normalisation ─────────────────────────────────
                # The model often returns ["flagged":...] instead of {"flagged":...}
                # Detect and fix the bracket mismatch:
                if content.startswith("[") and '"flagged"' in content:
                    content = "{" + content[1:]
                    if content.endswith("]"):
                        content = content[:-1] + "}"

                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    # Last resort: extract every {...} JSON object from the raw text
                    _write_log("Primary JSON parse failed — trying regex fallback extraction")
                    raw_objects = re.findall(r'\{[^{}]+\}', content, re.DOTALL)
                    parsed = []
                    for raw_obj in raw_objects:
                        try:
                            parsed.append(json.loads(raw_obj))
                        except Exception:
                            pass
                    _write_log(f"Fallback extracted {len(parsed)} object(s)")

                # Support both old (array) and new (object) response shapes
                if isinstance(parsed, list):
                    parsed_items = parsed
                    skipped_items = []
                else:
                    parsed_items = parsed.get("flagged", [])
                    skipped_items = parsed.get("skipped", [])


                # ── Write to debug log ────────────────────────────────────────
                _write_log(f"")
                _write_log(f"--- Page {page_idx + 1}: {len(parsed_items)} flagged, {len(skipped_items)} skipped ---")
                for fi in parsed_items:
                    _write_log(f"  [FLAGGED]  '{fi.get('text','')}' ({fi.get('tag','')}) → {fi.get('reason','')}")
                for sk in skipped_items:
                    _write_log(f"  [SKIPPED]  '{sk.get('text','')}' → {sk.get('reason','')}")

                char_pos = 0
                char_to_tok: dict[int, OCRToken] = {}
                for tok in page_tokens:
                    for c in range(len(tok.text)):
                        char_to_tok[char_pos + c] = tok
                    char_pos += len(tok.text) + 1  # +1 for space

                for item in parsed_items:
                    tag = item.get("tag")

                    item_text = item.get("text", "")
                    reason = item.get("reason", "")
                    if not tag or not item_text:
                        continue

                    # Find occurrences
                    _write_log(f"  [MATCHING]  Looking for '{item_text}' in full_text of length {len(full_text)}")
                    
                    # Try exact match first
                    matches = list(re.finditer(re.escape(item_text), full_text))
                    
                    # If no exact match, try fuzzy matching for emails first, then multi-word entities
                    if not matches:
                        _write_log(f"  [FUZZY]  Exact match failed, trying fuzzy matching")
                        
                        # Handle email addresses with spacing issues FIRST
                        if '@' in item_text:
                            _write_log(f"  [EMAIL FUZZY]  Trying email fuzzy matching")
                            
                            # Create multiple fuzzy patterns for robust email matching
                            fuzzy_patterns = []
                            
                            # Basic space removal patterns
                            fuzzy_patterns.append(item_text.replace(' @', '@'))  # Remove space before @
                            fuzzy_patterns.append(item_text.replace('@ ', '@'))  # Remove space after @
                            fuzzy_patterns.append(item_text.replace(' @ ', '@'))  # Remove spaces around @
                            fuzzy_patterns.append(re.sub(r'\s+@\s+', '@', item_text))  # Remove all spaces around @
                            
                            # Extract email parts and try partial matching
                            if '@' in item_text:
                                parts = item_text.split('@')
                                if len(parts) == 2:
                                    local_part = parts[0].strip()
                                    domain_part = parts[1].strip()
                                    
                                    # Try matching local part variations
                                    local_variations = [
                                        local_part.replace(' ', ''),  # Remove spaces
                                        local_part.replace('.', ''),  # Remove dots
                                        local_part.replace('_', ''),  # Remove underscores
                                    ]
                                    
                                    # Try matching domain part variations  
                                    domain_variations = [
                                        domain_part.replace(' ', ''),  # Remove spaces
                                        domain_part.replace('.', ''),  # Remove dots
                                    ]
                                    
                                    # Combine variations
                                    for local_var in local_variations:
                                        for domain_var in domain_variations:
                                            fuzzy_patterns.append(f"{local_var}@{domain_var}")
                            
                            # Try all patterns
                            for pattern in fuzzy_patterns:
                                matches = list(re.finditer(re.escape(pattern), full_text, re.IGNORECASE))
                                if matches:
                                    _write_log(f"  [EMAIL SUCCESS]  Found match using pattern '{pattern}'")
                                    break
                            
                            # If still no match, try finding any email in text
                            if not matches:
                                _write_log(f"  [EMAIL FALLBACK]  Trying fallback email detection")
                                # Find any email-like pattern in full_text
                                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
                                potential_emails = list(re.finditer(email_pattern, full_text))
                                if potential_emails:
                                    for email_match in potential_emails:
                                        email_found = email_match.group()
                                        # Check if this could be our target email
                                        if (local_part.lower() in email_found.lower() or 
                                            domain_part.lower() in email_found.lower()):
                                            matches = [email_match]
                                            _write_log(f"  [EMAIL FALLBACK SUCCESS]  Found '{email_found}' using fallback detection")
                                            break
                        
                        # Handle multi-word entities (like names) SECOND
                        elif ' ' in item_text and len(item_text.split()) >= 2:
                            _write_log(f"  [MULTI-WORD]  Trying multi-word entity matching for '{item_text}'")
                            words = item_text.split()
                            word_matches = []
                            
                            for word in words:
                                word_matches = list(re.finditer(r'\b' + re.escape(word) + r'\b', full_text, re.IGNORECASE))
                                if word_matches:
                                    word_matches.extend(word_matches)
                                    _write_log(f"  [WORD MATCH]  Found {len(word_matches)} matches for word '{word}'")
                            
                            if word_matches:
                                # Combine all word matches into entity spans
                                matches = word_matches
                                _write_log(f"  [MULTI-WORD SUCCESS]  Combined {len(word_matches)} word matches for '{item_text}'")
                    
                    _write_log(f"  [MATCHES]  Found {len(matches)} matches for '{item_text}'")
                    
                    if not matches:
                        _write_log(f"  [FAILED]  No text matches found for '{item_text}' - entity creation skipped")
                        continue
                        
                    for match in matches:
                        span_tokens_dict = {}
                        for c in range(match.start(), match.end()):
                            if c in char_to_tok:
                                t = char_to_tok[c]
                                span_tokens_dict[id(t)] = t

                        span_tokens = list(span_tokens_dict.values())
                        if not span_tokens:
                            continue

                        xs = [t.bbox[0] for t in span_tokens]
                        ys = [t.bbox[1] for t in span_tokens]
                        x2s = [t.bbox[0] + t.bbox[2] for t in span_tokens]
                        y2s = [t.bbox[1] + t.bbox[3] for t in span_tokens]
                        merged_bbox = (
                            min(xs), min(ys),
                            max(x2s) - min(xs), max(y2s) - min(ys),
                        )
                        avg_conf = sum(t.confidence for t in span_tokens) / len(span_tokens)

                        # Create simple entity for direct redaction
                        entities.append(SensitiveEntity(
                            id=-1, text=item_text, tag=tag,
                            source="llm", page=page_idx,
                            bbox=merged_bbox, confidence=avg_conf,
                            redact=True,
                            reason=reason,
                        ))
                        _write_log(f"  [REDACTION READY]  '{item_text}' ({tag}) --> Ready for redaction")
                        
                        # Mark tokens as sensitive
                        for tok in span_tokens:
                            entry = self.registry.get_or_create(tok.text)
                            for occ in entry.occurrences:
                                if occ.bbox == tok.bbox and occ.page == page_idx:
                                    occ.sensitive = True
                                    occ.tag = tag
            except Exception as e:
                import traceback
                _write_log(f"ERROR on page {page_idx + 1}: {e}")
                _write_log(f"TRACEBACK: {traceback.format_exc()}")
                print(f"LLM extraction error: {e}")

        return entities

    # ── Layer 3: Learned fields ───────────────────────────────────────────────

    def _run_learned_fields(self, tokens: list[OCRToken]) -> list[SensitiveEntity]:
        """
        For every token whose normalised text matches a learned-field key,
        mark it sensitive and emit an entity.
        """
        entities = []
        for tok in tokens:
            entry = self.registry.get(tok.text)
            if entry and entry.is_learned_field:
                # Mark all occurrences in registry
                for occ in entry.occurrences:
                    if occ.bbox == tok.bbox and occ.page == tok.page:
                        occ.sensitive = True
                        occ.tag = entry.default_tag

                entities.append(SensitiveEntity(
                    id=-1, text=tok.text, tag=entry.default_tag or "SENSITIVE",
                    source="learned_field", page=tok.page,
                    bbox=tok.bbox, confidence=tok.confidence,
                    redact=True,
                ))
        return entities

    # ── Public ────────────────────────────────────────────────────────────────

    def detect(self, tokens: list[OCRToken]) -> list[SensitiveEntity]:
        """
        Run detection layers, deduplicate, assign sequential IDs.
        Registers every token into the WordRegistry as a side effect.
        """
        entities: list[SensitiveEntity] = []
        entities.extend(self._run_llm(tokens))
        entities.extend(self._run_learned_fields(tokens))

        entities = _deduplicate(entities)

        # Assign sequential IDs
        for i, ent in enumerate(entities):
            ent.id = i

        return entities


# ──────────────────────────────────────────────────────────────────────────────
# Value extractor  (for "FIELD LABEL : VALUE" parsing)
# ──────────────────────────────────────────────────────────────────────────────

def extract_field_value(
    tokens: list[OCRToken],
    field_label: str,
    window: int = 5,
) -> Optional[OCRToken]:
    """
    Given a field label (e.g. "ACCOUNT NUMBER"), find its value in the token
    stream by looking at the `window` tokens that immediately follow the
    last token of the label.

    Returns the value OCRToken or None if the label isn't found.
    """
    label_words = field_label.lower().split()
    texts = [t.text.lower().strip(":") for t in tokens]
    n = len(label_words)

    for i in range(len(tokens) - n):
        if texts[i : i + n] == label_words:
            # Next non-colon token after the label
            for j in range(i + n, min(i + n + window, len(tokens))):
                candidate = tokens[j].text.strip(":").strip()
                if candidate:
                    return tokens[j]
    return None