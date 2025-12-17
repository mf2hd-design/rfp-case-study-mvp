import io
import os
import re
from datetime import datetime
from typing import List

import pandas as pd
import streamlit as st
import requests
import time
from docx import Document as DocxDocument


APP_TITLE = "RFP → Relevant Case Studies (MVP)"
BUNDLED_XLSX = "Case Study List & Services.xlsx"
DEFAULT_MODEL = "gpt-5.1"
DEFAULT_TOPN = 15


def extract_text_from_pdf(file_bytes: bytes) -> str:
    import fitz  # PyMuPDF
    parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text(file_name: str, file_bytes: bytes) -> str:
    name = file_name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    return file_bytes.decode("utf-8", errors="ignore")


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    for col in ["PRIORITY", "CLIENT", "INDUSTRY", "MARKET", "SERVICES", "OWNER"]:
        if col not in df.columns:
            df[col] = ""
    return df


def split_services(s: object) -> List[str]:
    if pd.isna(s):
        return []
    return [p.strip() for p in str(s).split(",") if p.strip()]


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-&/]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def build_candidate_text(row: pd.Series) -> str:
    bits = [
        str(row.get("CLIENT", "")),
        str(row.get("INDUSTRY", "")),
        str(row.get("MARKET", "")),
        str(row.get("SERVICES", "")),
    ]
    return normalize(" ".join(bits))


def rank_cases(df: pd.DataFrame, rfp_text: str) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    rfp_norm = normalize(rfp_text)
    docs = df["CANDIDATE_TEXT"].tolist()

    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vect.fit_transform([rfp_norm] + docs)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()

    def boost(row: pd.Series) -> float:
        b = 0.0
        ind = normalize(str(row.get("INDUSTRY", "")))
        mkt = normalize(str(row.get("MARKET", "")))
        svcs = [normalize(s) for s in split_services(row.get("SERVICES", ""))]

        if ind and ind in rfp_norm:
            b += 0.10
        if mkt and mkt in rfp_norm:
            b += 0.05
        for s in svcs:
            if s and s in rfp_norm:
                b += 0.03

        pr = str(row.get("PRIORITY", "")).strip().lower()
        if pr == "high":
            b += 0.05
        elif pr == "medium":
            b += 0.02
        return b

    scores = sims + df.apply(boost, axis=1).to_numpy()
    out = df.copy()
    out["SCORE"] = scores
    return out.sort_values("SCORE", ascending=False)


def explain_match(row: pd.Series, rfp_text: str) -> str:
    t = normalize(rfp_text)
    hits = []

    ind = str(row.get("INDUSTRY", "")).strip()
    mkt = str(row.get("MARKET", "")).strip()
    pr = str(row.get("PRIORITY", "")).strip()
    svcs = split_services(row.get("SERVICES", ""))

    if ind and normalize(ind) in t:
        hits.append(f"Industry: {ind}")
    if mkt and normalize(mkt) in t:
        hits.append(f"Market: {mkt}")

    svc_hits = [s for s in svcs if normalize(s) in t]
    if svc_hits:
        hits.append("Services: " + ", ".join(svc_hits[:6]))

    if pr:
        hits.append(f"Priority: {pr}")

    return " • ".join(hits) if hits else "Matched by overall similarity to case-study metadata (industry/market/services)."


def openai_chat_completions(
    api_key: str,
    model: str,
    user_prompt: str,
    verbosity: str = "low",
    reasoning_effort: str = "high",
    max_retries: int = 3,
) -> str:
    """
    Calls OpenAI Chat Completions with:
    - higher read timeout (high reasoning can exceed 60s on Render)
    - basic retries with exponential backoff for transient timeouts
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a senior bid strategist for a brand consultancy. Be concrete and concise."},
            {"role": "user", "content": user_prompt},
        ],
        "verbosity": verbosity,
        "reasoning_effort": reasoning_effort,
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            # timeout=(connect_timeout_seconds, read_timeout_seconds)
            r = requests.post(url, headers=headers, json=payload, timeout=(10, 180))
            if r.status_code != 200:
                raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text[:800]}")
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            last_err = e
        except requests.exceptions.RequestException as e:
            last_err = e

        # backoff: 1s, 2s, 4s...
        time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"OpenAI request failed after {max_retries} attempts: {last_err}")

def build_angles_prompt(rfp_text: str, ranked: pd.DataFrame) -> str:
    top = ranked.head(8)[["CLIENT", "INDUSTRY", "MARKET", "SERVICES", "PRIORITY"]].fillna("").to_dict(orient="records")
    return f"""
RFP TEXT (excerpt):
{rfp_text[:3000]}

TOP CANDIDATE CASE STUDIES (metadata only):
{top}

Task:
1) Write 6 response angles for our proposal.
- Each angle must be 1 sentence max.
- Keep them specific to likely buyer concerns: governance, rollout, adoption, risk, speed, clarity, stakeholder alignment.
- Where helpful, reference 1–2 relevant case studies by CLIENT name.

2) Then create a 6-row table mapping requirements to proof.
- First, infer 6 key REQUIREMENTS from the RFP (short phrases).
- For each requirement, pick the best matching case study from the list above.
- Explain WHY in one short sentence (based only on the metadata provided).

Output format (STRICT):
- Section A: bullets only (6 bullets).
- Blank line.
- Section B: a Markdown table with header:
| Requirement | Suggested case study | Why |
|---|---|---|
No intro, no outro.
""".strip()


@st.cache_data(show_spinner=False)
def load_bundled_catalogue() -> pd.DataFrame:
    if not os.path.exists(BUNDLED_XLSX):
        raise FileNotFoundError(f"Bundled Excel file not found: {BUNDLED_XLSX}")
    df = pd.read_excel(BUNDLED_XLSX, sheet_name=0)
    df = clean_cols(df)
    df["CANDIDATE_TEXT"] = df.apply(build_candidate_text, axis=1)
    return df


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Upload an RFP (PDF/DOCX) → relevant case studies + GPT‑5.1 response angles + requirements→case mapping.")

    with st.sidebar:
        st.header("Inputs")
        rfp = st.file_uploader("Upload RFP (PDF or DOCX)", type=["pdf", "docx", "txt"])
        topn = st.slider("How many case studies to show?", 5, 50, DEFAULT_TOPN)

        st.divider()
        st.header("Angles + mapping (GPT‑5.1)")
        enable_angles = st.toggle("Generate response angles + mapping", value=True)
        model = st.text_input("Model", value=DEFAULT_MODEL)
        verbosity = st.selectbox("Verbosity", ["low", "medium", "high"], index=0)
        reasoning = st.selectbox("Reasoning effort", ["low", "medium", "high"], index=2)

        st.caption("Set OPENAI_API_KEY in Render env vars to enable angles.")
        st.caption("High reasoning is best for extracting requirements + mapping, but can be slower.")

    df = load_bundled_catalogue()

    if not rfp:
        st.info("Upload an RFP to generate recommendations.")
        st.stop()

    rfp_text = extract_text(rfp.name, rfp.getvalue())
    if len(rfp_text.strip()) < 200:
        st.warning("Extracted RFP text is very short. If this is a scanned PDF, upload a text-based PDF or DOCX.")

    st.subheader("RFP preview (first ~1,500 chars)")
    st.code(rfp_text[:1500])

    ranked = rank_cases(df, rfp_text).head(topn).copy()
    ranked["WHY_IT_MATCHES"] = ranked.apply(lambda r: explain_match(r, rfp_text), axis=1)

    show_cols = ["SCORE", "PRIORITY", "CLIENT", "INDUSTRY", "MARKET", "SERVICES", "OWNER", "WHY_IT_MATCHES"]
    st.subheader("Recommended case studies")
    st.dataframe(ranked[show_cols], use_container_width=True, hide_index=True)

    st.download_button(
        "Download shortlist (CSV)",
        data=ranked[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="rfp_case_study_shortlist.csv",
        mime="text/csv",
    )

    if enable_angles:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            st.warning("OPENAI_API_KEY is not set. Add it in Render → Environment.")
        else:
            with st.spinner("Generating angles + mapping (GPT‑5.1)…"):
                prompt = build_angles_prompt(rfp_text, ranked)
                try:
                    out = openai_chat_completions(
                        api_key=api_key,
                        model=model,
                        user_prompt=prompt,
                        verbosity=verbosity,
                        reasoning_effort=reasoning,
                    )
                except Exception as e:
                    st.error(f"OpenAI call failed: {e}")
                    st.info("Quick fixes: (1) set Reasoning effort to medium, (2) upload DOCX instead of scanned PDF, (3) retry.")
                    return

            table_start = out.find("| Requirement")
            bullets = out.strip()
            table_md = ""
            if table_start != -1:
                bullets = out[:table_start].strip()
                table_md = out[table_start:].strip()

            st.subheader("Response angles (concise)")
            st.markdown(bullets if bullets else out)

            st.subheader("Requirements → case study mapping")
            if table_md:
                st.markdown(table_md)
            else:
                st.info("No table detected. Try again or set reasoning to high.")

    st.divider()
    st.markdown("**MVP scope:** metadata-only matching via the bundled Excel index. No Box API, no deck/PDF ingestion.")


if __name__ == "__main__":
    main()
