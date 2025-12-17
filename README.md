# RFP → Relevant Case Studies (MVP)

## What it does
- Bundles the provided case-study index Excel inside the app (no Excel upload step).
- User uploads an RFP (PDF/DOCX/TXT).
- Returns a ranked shortlist of relevant case studies (metadata-based).
- Generates:
  - 6 concise response angles (GPT‑5.1, verbosity low)
  - a 6-row “Requirements → Suggested case study → Why” mapping table (Markdown)
- Selectable reasoning effort (default: high).

## Local run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Render
Use the included `render.yaml` (Blueprint).

### Required env vars
- `OPENAI_API_KEY` (required for angles + mapping)
