"""
app.py — Streamlit UI for the Document Sanitization Pipeline.

Run:
    streamlit run app.py
"""

import io
import os
import tempfile
from pathlib import Path

import sys
try:
    import torch
    torch.classes.__path__ = []
except ImportError:
    pass

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="DocShield — Document Sanitizer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import pipeline (same directory) ─────────────────────────────────────────
from pipeline import SanitizationPipeline, ProcessingResult

# ─────────────────────────────────────────────────────────────────────────────
# TAG → HEX colour (for UI badges, matching redactor BGR colours)
# ─────────────────────────────────────────────────────────────────────────────
TAG_COLORS: dict[str, str] = {
    "EMAIL":               "#228B22",
    "PHONE":               "#FF8C00",
    "SSN":                 "#DC143C",
    "CREDIT_CARD":         "#9400D3",
    "AADHAAR":             "#008080",
    "PAN":                 "#0064C8",
    "PERSON":              "#DC143C",
    "ORG":                 "#FFA500",
    "GPE":                 "#6495ED",
    "MONEY":               "#32CD32",
    "LAW":                 "#C71585",
    "ACCOUNT_NUMBER":      "#00BFFF",
    "USER_WORD":           "#FF4500",
    "SENSITIVE":           "#800080",
    "ADDRESS":             "#B8860B",
    "DOB":                 "#FF1493",
    "PASSPORT":            "#1E90FF",
    "DRIVER_LICENSE":      "#20B2AA",
    "IP_ADDRESS":          "#FF6347",
    "MEDICAL_HISTORY":     "#8B0000",
    "CORPORATE_CONFIDENTIAL": "#4B0082",
}
DEFAULT_TAG_COLOR = "#888888"


def tag_badge(tag: str) -> str:
    color = TAG_COLORS.get(tag.upper(), DEFAULT_TAG_COLOR)
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:11px;font-weight:600;'
        f'letter-spacing:.5px">{tag}</span>'
    )


def source_badge(source: str) -> str:
    colors = {
        "regex":         "#4a90d9",
        "ner":           "#7b68ee",
        "learned_field": "#20b2aa",
        "user_field":    "#20b2aa",
        "user_word":     "#ff6347",
    }
    color = colors.get(source, "#aaa")
    return (
        f'<span style="background:{color}22;color:{color};padding:1px 6px;'
        f'border-radius:3px;font-size:10px;border:1px solid {color}55">'
        f'{source}</span>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "result":        None,   # ProcessingResult
        "pipeline":      None,   # SanitizationPipeline
        "tmp_pdf":       None,   # path to uploaded temp file
        "preview_page":  0,
        "processing":    False,
        "export_bytes":  None,
        "export_name":   None,
        "log":           [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _log(msg: str):
    st.session_state.log.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Page image → PIL (for st.image)
# ─────────────────────────────────────────────────────────────────────────────

def _to_pil(img: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Sidebar width */
section[data-testid="stSidebar"] { min-width: 320px; max-width: 360px; }

/* Entity card */
.ent-card {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-left: 4px solid #ccc;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.ent-card.redacted  { border-left-color: #e53935; background: #fff5f5; }
.ent-card.kept      { border-left-color: #43a047; background: #f5fff5; }
.ent-text { font-family: monospace; font-size: 14px; font-weight: 600; flex: 1; }
.ent-meta  { font-size: 11px; color: #777; }

/* Stat pills */
.stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1rem; }
.stat-pill {
    background: #f0f2f6;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 13px;
    font-weight: 600;
}

/* Step indicator */
.step { color: #1976d2; font-weight: 700; font-size: 13px; text-transform: uppercase;
        letter-spacing: .5px; margin-bottom: 4px; }

/* Success banner */
.export-banner {
    background: #e8f5e9; border: 1px solid #a5d6a7;
    border-radius: 8px; padding: 12px 16px;
    font-size: 14px; color: #2e7d32;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.markdown("## 🛡️ DocShield")
        st.caption("Document Sanitization Pipeline")
        st.divider()

        # ── Upload ────────────────────────────────────────────────────────────
        st.markdown('<div class="step">① Upload PDF</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose a PDF file", type=["pdf"], label_visibility="collapsed"
        )

        st.divider()

        # ── Settings ──────────────────────────────────────────────────────────
        st.markdown('<div class="step">② Settings</div>', unsafe_allow_html=True)
        groq_api_key = st.text_input("Groq API Key (required)", type="password")
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        purpose = st.text_area("Purpose of upload (Context)", placeholder="e.g. For medical review, public release, etc.")
        ocr_backend = st.selectbox(
            "OCR backend",
            ["tesseract", "easyocr", "paddleocr"],
            help="tesseract = fastest · easyocr = better on noisy scans · paddleocr = best accuracy"
        )
        dpi = st.slider("Scan DPI", 150, 400, 300, 50,
                        help="Higher = better OCR accuracy but slower")
        conf = st.slider("Min OCR confidence", 0.1, 0.9, 0.4, 0.05,
                         help="Tokens below this confidence are discarded")

        st.divider()

        # ── Run ───────────────────────────────────────────────────────────────
        st.markdown('<div class="step">③ Analyse</div>', unsafe_allow_html=True)
        run_btn = st.button("🔍  Analyse document", use_container_width=True,
                            type="primary", disabled=(uploaded is None or not groq_api_key.strip()))

        # ── Learned fields info ───────────────────────────────────────────────
        if st.session_state.pipeline:
            learned = st.session_state.pipeline.registry.learned_fields()
            if learned:
                st.divider()
                st.markdown("**Learned fields** (auto-detected in future docs)")
                for entry in learned:
                    st.markdown(
                        f'<span style="font-size:12px;font-family:monospace;">'
                        f'• {entry.word} → {entry.default_tag}</span>',
                        unsafe_allow_html=True,
                    )

        return uploaded, ocr_backend, dpi, conf, run_btn, purpose


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(uploaded, ocr_backend, dpi, conf, purpose):
    # Save upload to a temp file
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.read())
        tmp_path = f.name
    st.session_state.tmp_pdf = tmp_path
    st.session_state.export_bytes = None
    st.session_state.export_name = None
    st.session_state.log = []
    st.session_state.preview_page = 0

    pipe = SanitizationPipeline(
        registry_path="registry.json",
        ocr_backend=ocr_backend,
        dpi=dpi,
        conf_threshold=conf,
        purpose=purpose,
    )
    st.session_state.pipeline = pipe

    progress = st.progress(0, text="Rasterising pages…")
    with st.spinner(""):
        try:
            # Monkey-patch print → log capture
            import builtins
            _orig_print = builtins.print
            def _capture(*args, **kwargs):
                msg = " ".join(str(a) for a in args)
                _log(msg)
                _orig_print(*args, **kwargs)
            builtins.print = _capture

            progress.progress(10, text="Rasterising pages…")
            result = pipe.ingest(tmp_path)
            progress.progress(100, text="Done!")

            builtins.print = _orig_print
        except Exception as e:
            builtins.print = _orig_print
            st.error(f"Pipeline error: {e}")
            st.exception(e)
            return

    st.session_state.result = result
    progress.empty()


# ─────────────────────────────────────────────────────────────────────────────
# Entity review panel
# ─────────────────────────────────────────────────────────────────────────────

def _entity_panel(result: ProcessingResult):
    summary = result.summary()
    total   = len(result.entities)
    redact  = sum(1 for e in result.entities if e.redact)

    # Stats row
    st.markdown(
        f'<div class="stat-row">'
        f'<div class="stat-pill">📄 {len(result.raw_pages)} page(s)</div>'
        f'<div class="stat-pill">🔍 {total} entities detected</div>'
        f'<div class="stat-pill" style="background:#ffeaea">🚫 {redact} to redact</div>'
        f'<div class="stat-pill" style="background:#eafaea">✅ {total-redact} kept</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not summary:
        st.info("No sensitive entities detected automatically.")
        return

    # One expander per tag group
    for tag, items in sorted(summary.items()):
        color = TAG_COLORS.get(tag.upper(), DEFAULT_TAG_COLOR)
        redact_count = sum(1 for i in items if i["redact"])
        with st.expander(
            f"**{tag}** — {len(items)} item(s), {redact_count} redacted",
            expanded=True,
        ):
            for item in items:
                col_chk, col_info = st.columns([0.08, 0.92])
                with col_chk:
                    checked = st.checkbox(
                        "", value=item["redact"],
                        key=f"ent_{item['id']}",
                        label_visibility="collapsed",
                    )
                    # Sync toggle back to result
                    if checked != item["redact"]:
                        if checked:
                            result.select(item["id"])
                        else:
                            result.deselect(item["id"])
                        item["redact"] = checked

                with col_info:
                    card_cls = "redacted" if item["redact"] else "kept"
                    reason_html = f'<div style="font-size:11px;color:#666;margin-top:4px;"><i>Reason: {item["reason"]}</i></div>' if item.get("reason") else ""
                    st.markdown(
                        f'<div class="ent-card {card_cls}" style="flex-direction:column;align-items:start;">'
                        f'<div style="display:flex;width:100%;align-items:center;gap:10px;">'
                        f'<span class="ent-text">"{item["text"]}"</span>'
                        f'<span class="ent-meta">page {item["page"]+1}</span>'
                        f'{source_badge(item["source"])}'
                        f'</div>'
                        f'{reason_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# User actions panel
# ─────────────────────────────────────────────────────────────────────────────

def _actions_panel(result: ProcessingResult):
    st.markdown("---")
    st.markdown("### ✏️ Add more redactions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "**Teach a new field** — system learns it permanently  \n"
            "<span style='font-size:12px;color:#666'>e.g. `ACCOUNT NUMBER`, `POLICY ID`</span>",
            unsafe_allow_html=True,
        )
        field_input = st.text_input(
            "Field label", placeholder="ACCOUNT NUMBER",
            key="field_input", label_visibility="collapsed"
        )
        if st.button("📌  Learn & redact field", key="btn_field",
                     use_container_width=True, disabled=not field_input.strip()):
            new_ents = result.add_field(field_input.strip())
            if new_ents:
                st.success(
                    f"Learned **{field_input.upper()}** — "
                    f"{len(new_ents)} value(s) flagged. Will auto-detect in future docs."
                )
            else:
                st.warning(
                    f"Label '{field_input}' taught, but no adjacent value found on this page. "
                    f"It will auto-detect in future documents."
                )
            st.rerun()

    with col2:
        st.markdown(
            "**Redact a specific word** — this session only  \n"
            "<span style='font-size:12px;color:#666'>e.g. `Confidential`, a name</span>",
            unsafe_allow_html=True,
        )
        word_input = st.text_input(
            "Word to redact", placeholder="Confidential",
            key="word_input", label_visibility="collapsed"
        )
        if st.button("🔒  Redact word (session only)", key="btn_word",
                     use_container_width=True, disabled=not word_input.strip()):
            new_ents = result.redact_word(word_input.strip())
            if new_ents:
                st.success(f"**{word_input}** — {len(new_ents)} occurrence(s) will be redacted.")
            else:
                st.warning(f"Word '{word_input}' not found in document.")
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Page preview panel
# ─────────────────────────────────────────────────────────────────────────────

def _preview_panel(result: ProcessingResult):
    n_pages = len(result.raw_pages)
    st.markdown("### 🖼️ Document preview")

    col_prev, col_page, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("◀", disabled=(st.session_state.preview_page == 0)):
            st.session_state.preview_page -= 1
            st.rerun()
    with col_page:
        st.markdown(
            f"<div style='text-align:center;font-weight:600;padding:6px 0'>"
            f"Page {st.session_state.preview_page + 1} / {n_pages}</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("▶", disabled=(st.session_state.preview_page >= n_pages - 1)):
            st.session_state.preview_page += 1
            st.rerun()

    pg = st.session_state.preview_page

    preview_img = result.preview_page(pg)
    st.image(_to_pil(preview_img), use_container_width=True)

    # Colour legend
    tags_on_page = list({
        e.tag for e in result.entities
        if e.page == pg and e.redact
    })
    if tags_on_page:
        legend_html = "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:6px'>"
        for t in sorted(tags_on_page):
            legend_html += tag_badge(t) + " "
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

    # Raw Text Expander
    page_text = " ".join([tok.text for tok in result.tokens if tok.page == pg])
    with st.expander("📝 View Raw Extracted OCR Text", expanded=False):
        st.caption("This is exactly what the OCR engine saw and passed to the LLM:")
        st.code(page_text, language="text")


# ─────────────────────────────────────────────────────────────────────────────
# Export panel
# ─────────────────────────────────────────────────────────────────────────────

def _export_panel(result: ProcessingResult, original_name: str):
    st.markdown("---")
    st.markdown("### 📥 Export redacted PDF")

    redact_count = sum(1 for e in result.entities if e.redact)
    st.markdown(
        f"**{redact_count}** entit{'y' if redact_count==1 else 'ies'} "
        f"will be permanently redacted with black boxes."
    )

    if st.button("⬛  Generate redacted PDF", type="primary", use_container_width=True):
        stem = Path(original_name).stem
        out_name = f"{stem}_redacted.pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            out_path = f.name

        with st.spinner("Applying redactions and building PDF…"):
            result.export(out_path)

        with open(out_path, "rb") as f:
            st.session_state.export_bytes = f.read()
        st.session_state.export_name = out_name
        os.unlink(out_path)
        st.rerun()

    if st.session_state.export_bytes:
        st.markdown(
            '<div class="export-banner">✅ Redacted PDF is ready for download.</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            label="⬇️  Download redacted PDF",
            data=st.session_state.export_bytes,
            file_name=st.session_state.export_name,
            mime="application/pdf",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Processing log (collapsed)
# ─────────────────────────────────────────────────────────────────────────────

def _log_panel():
    if st.session_state.log:
        with st.expander("🪵 Processing log", expanded=False):
            st.code("\n".join(st.session_state.log))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _init_state()
    uploaded, ocr_backend, dpi, conf, run_btn, purpose = _sidebar()

    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown("# 🛡️ DocShield")
    st.markdown(
        "Upload a scanned PDF · auto-detect PII & sensitive entities · "
        "review · redact · download the clean version."
    )
    st.divider()

    # ── Trigger pipeline ──────────────────────────────────────────────────────
    if run_btn and uploaded:
        _run_pipeline(uploaded, ocr_backend, dpi, conf, purpose)

    result: ProcessingResult = st.session_state.result

    # ── No result yet ─────────────────────────────────────────────────────────
    if result is None:
        st.markdown(
            """
            <div style="text-align:center;padding:60px 0;color:#888">
                <div style="font-size:56px">📄</div>
                <div style="font-size:18px;margin-top:12px">
                    Upload a PDF in the sidebar and click <b>Analyse document</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Main layout: entities left, preview right ─────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### 🔍 Detected entities")
        st.caption("Uncheck any entity you want to keep visible in the final PDF.")
        _entity_panel(result)
        _actions_panel(result)
        _export_panel(
            result,
            original_name=uploaded.name if uploaded else "document.pdf",
        )

    with right:
        _preview_panel(result)
        _log_panel()


if __name__ == "__main__":
    main()