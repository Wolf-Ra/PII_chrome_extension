import os
import io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pipeline import SanitizationPipeline, Config

app = FastAPI(title="Context-Aware PII Redaction API")

# Allow Chrome extension to call this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For the extension to bypass CORS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline with default config (can be overridden per request)
pipeline = SanitizationPipeline()

def _get_website_context(domain: str) -> dict:
    """Uses LLM to analyze website domain and extract detailed context for PII redaction.
    Returns a dictionary with website type, industry, and specific PII considerations.
    """
    if not domain or domain.lower() in ("localhost", "127.0.0.1", ""):
        return {}

    #api_key = os.environ.get("GROQ_API_KEY")
    api_key = "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"
    if not api_key:
        print("Warning: GROQ_API_KEY not found. Skipping dynamic domain profiling.")
        return {}

    try:
        client = Groq(api_key=api_key)
        prompt = (
            f"You are a web context analyzer specializing in PII (Personally Identifiable Information) redaction. "
            f"Given a website domain, analyze and categorize it for intelligent PII filtering.\n\n"
            f"Respond with a JSON object containing:\n"
            f"- website_type: (e.g., 'healthcare', 'financial', 'government', 'education', 'ecommerce', 'social_media', 'corporate', 'job_portal', 'legal', 'general')\n"
            f"- industry: specific industry name (e.g., 'banking', 'hospital', 'insurance', 'university', 'retail')\n"
            f"- primary_pii_types: list of PII types most relevant for this site (e.g., ['medical_records', 'account_numbers', 'ssn', 'credit_card', 'academic_records'])\n"
            f"- sensitivity_level: (low, medium, high, critical) - how sensitive the PII on this site typically is\n"
            f"- description: brief one-sentence description of the site's purpose\n\n"
            f"Examples:\n"
            f"- 'hdfcbank.com' -> {{\"website_type\": \"financial\", \"industry\": \"banking\", \"primary_pii_types\": [\"account_numbers\", \"credit_card\", \"ssn\", \"transaction_history\"], \"sensitivity_level\": \"critical\", \"description\": \"Banking and financial services platform\"}}\n"
            f"- 'mayoclinic.com' -> {{\"website_type\": \"healthcare\", \"industry\": \"hospital\", \"primary_pii_types\": [\"medical_records\", \"patient_id\", \"insurance_info\", \"ssn\"], \"sensitivity_level\": \"critical\", \"description\": \"Healthcare provider patient portal\"}}\n"
            f"- 'github.com' -> {{\"website_type\": \"corporate\", \"industry\": \"technology\", \"primary_pii_types\": [\"email\", \"full_name\", \"api_keys\"], \"sensitivity_level\": \"medium\", \"description\": \"Code hosting and development platform\"}}\n\n"
            f"Domain: {domain}\n"
            f"Return ONLY the JSON object. No other text."
        )
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        
        import json
        context = json.loads(response.choices[0].message.content.strip())
        print(f"Domain Profiler: '{domain}' -> {context}")
        return context
    except Exception as e:
        print(f"Domain Profiler error: {e}")
        return {}

@app.post("/redact")
async def redact_document(
    file: UploadFile = File(...),
    domain: str = Form(""),
):
    print(f"\n--- New Upload Intercepted ---")
    print(f"File: {file.filename}")
    print(f"Target Domain: {domain}")

    # Step 1: Get detailed website context from the domain
    website_context = _get_website_context(domain)
    pipeline.set_website_context(website_context)

    # Step 2: Read the uploaded file bytes
    file_bytes = await file.read()

    # Step 3: Check if it's a PDF. (Pipeline expects a PDF file path, but supports BytesIO in fitz)
    # We need to save the stream temporarily since our pipeline expects a path right now
    import tempfile
    suffix = os.path.splitext(file.filename)[1] or '.pdf'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        print("Processing document through pipeline...")
        # Step 4: Process it via the pipeline
        result = pipeline.process_document(tmp_path)
        print(f"Found {len(result.entities)} entities across {len(result.raw_pages)} pages.")

        # Step 5: Generate the redacted PDF
        # Save redacted file to current working directory
        out_path = os.path.join(os.getcwd(), "redacted.pdf")
        print(f"Redacted file saved to: {out_path}")
        result.export_pdf(out_path)

        with open(out_path, "rb") as f:
            redacted_bytes = f.read()

        # Don't clean up - keep the redacted file for user
        print(f"🔒 REDACTION COMPLETE! 🔒")
        print(f"📁 Redacted file saved to: {out_path}")
        print(f"📤 You can now upload this securely redacted file to any website!")
        print(f"🌐 Context-aware PII redaction applied based on: {website_context.get('website_type', 'unknown')}")

        safe_filename = file.filename.encode("latin-1", errors="replace").decode("latin-1")
        return Response(
            content=redacted_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="redacted_{safe_filename}"'
            }
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
