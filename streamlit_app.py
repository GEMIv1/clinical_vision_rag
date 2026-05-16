import os
import streamlit as st
import requests
import base64

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Vision RAG", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .source-card {
        background: #1a1d23;
        border: 1px solid #2d333b;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .source-header {
        color: #58a6ff;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 8px;
    }
    .source-meta {
        color: #8b949e;
        font-size: 12px;
    }
    .answer-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-top: 16px;
    }
    .image-preview-label {
        color: #8b949e;
        font-size: 12px;
        margin-bottom: 6px;
    }
    .ocr-badge {
        background: #1f6feb;
        color: white;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Vision RAG — Clinical Case Search")
st.caption("Search past patient case reports for similar cases, treatments, and outcomes.")

tab_search, tab_ocr = st.tabs(["Main", "Extract Text"])

# ═══════════════════════════════════════════
# SEARCH TAB
# ═══════════════════════════════════════════
with tab_search:
    col_input, col_filters = st.columns([3, 1])

    with col_input:
        query = st.text_area(
            "Ask a clinical question",
            placeholder="Have we seen a case of a child with respiratory failure after COVID-19? How was it treated?",
            height=100,
        )

        st.markdown("**Attach an image** *(optional — extract text and append to your query)*")
        uploaded_image = st.file_uploader(
            "Upload a medical document, scan, or photo",
            type=["png", "jpg", "jpeg", "webp", "tiff"],
            key="search_image",
        )

        if uploaded_image:
            st.image(uploaded_image, caption="Attached image", use_container_width=False, width=320)
            st.markdown('<span class="ocr-badge">📷 Image attached — text will be extracted and added to your query</span>', unsafe_allow_html=True)

    with col_filters:
        st.markdown("**Filters**")
        top_k = st.slider("Results", 1, 20, 5)
        min_age = st.number_input("Min age", min_value=0.0, value=None, step=1.0, format="%.1f")
        max_age = st.number_input("Max age", min_value=0.0, value=None, step=1.0, format="%.1f")
        gender = st.selectbox("Gender", [None, "M", "F"], format_func=lambda x: "Any" if x is None else x)

    col_search, col_retrieve = st.columns(2)
    search_clicked = col_search.button("🔍 Search (RAG)", use_container_width=True, type="primary")
    retrieve_clicked = col_retrieve.button("📄 Retrieve Only", use_container_width=True)

    if search_clicked or retrieve_clicked:
        if not query and not uploaded_image:
            st.warning("Please enter a question or attach an image.")
        else:
            effective_query = query or ""

            if uploaded_image:
                with st.spinner("Extracting text from image..."):
                    try:
                        files = {
                            "file": (
                                uploaded_image.name,
                                uploaded_image.getvalue(),
                                uploaded_image.type,
                            )
                        }
                        ocr_resp = requests.post(
                            f"{API_BASE}/ocr/",
                            files=files,
                            params={"output_format": "plain"},
                            timeout=60,
                        )
                        ocr_resp.raise_for_status()
                        extracted_text = ocr_resp.json().get("text", "").strip()
                        if extracted_text:
                            if effective_query:
                                effective_query = f"{effective_query}\n\n[Extracted from attached image]:\n{extracted_text}"
                            else:
                                effective_query = f"[Extracted from attached image]:\n{extracted_text}"
                            st.info(f"📄 Extracted text appended to query ({len(extracted_text)} characters).")
                    except requests.ConnectionError:
                        st.error("Cannot connect to the API for OCR. Make sure the FastAPI server is running.")
                        st.stop()
                    except requests.HTTPError as e:
                        st.error(f"OCR API error: {e.response.status_code} — {e.response.text}")
                        st.stop()

            endpoint = "/search/" if search_clicked else "/search/retrieve"
            payload = {"query": effective_query, "top_k": top_k}
            if min_age is not None:
                payload["min_age"] = min_age
            if max_age is not None:
                payload["max_age"] = max_age
            if gender is not None:
                payload["gender"] = gender

            with st.spinner("Searching and generating answer..." if search_clicked else "Searching..."):
                try:
                    resp = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()

                    if search_clicked:
                        st.subheader("💬 Answer")
                        st.markdown(
                            f'<div class="answer-box">{data["answer"]}</div>',
                            unsafe_allow_html=True,
                        )

                        if data.get("sources"):
                            st.subheader("📚 Sources")
                            for src in data["sources"]:
                                meta = src.get("metadata", {})
                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                        <div class="source-header">Patient {meta.get('patient_uid', 'N/A')} | RRF Score: {src.get('rrf_score', 'N/A')}</div>
                                        <div class="source-meta">Age: {meta.get('age', 'N/A')} | Gender: {meta.get('gender', 'N/A')} | Chunk: {meta.get('chunk_index', '?')}/{meta.get('total_chunks', '?')}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.subheader("📄 Retrieved Documents")
                        for r in data.get("results", []):
                            meta = r.get("metadata", {})
                            with st.expander(
                                f"🔹 {meta.get('patient_uid', 'N/A')} — {meta.get('title', '')[:80]}"
                            ):
                                st.markdown(
                                    f"**Age:** {meta.get('age', 'N/A')} | **Gender:** {meta.get('gender', 'N/A')} | **RRF Score:** {r.get('rrf_score', 'N/A')}"
                                )
                                st.text(r.get("text", ""))

                except requests.ConnectionError:
                    st.error("Cannot connect to the API. Make sure the FastAPI server is running on localhost:8000.")
                except requests.HTTPError as e:
                    st.error(f"API error: {e.response.status_code} — {e.response.text}")

# ═══════════════════════════════════════════
# OCR TAB
# ═══════════════════════════════════════════
with tab_ocr:
    uploaded_file = st.file_uploader("Upload a medical document image", type=["png", "jpg", "jpeg", "webp", "tiff"])

    col_ocr1, col_ocr2 = st.columns(2)
    with col_ocr1:
        ocr_prompt = st.text_input("Custom prompt (optional)", placeholder="Extract all text from this image")
    with col_ocr2:
        output_format = st.selectbox("Output format", ["plain", "json"])

    if st.button("🔍 Extract Text", type="primary") and uploaded_file:
        with st.spinner("Running OCR..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {"output_format": output_format}
                if ocr_prompt:
                    params["prompt"] = ocr_prompt


                resp = requests.post(f"{API_BASE}/ocr/", files=files, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                st.subheader("📝 Extracted Text")
                st.code(data.get("text", ""), language=None)
            except requests.ConnectionError:
                st.error("Cannot connect to the API.")
            except requests.HTTPError as e:
                st.error(f"API error: {e.response.status_code} — {e.response.text}")