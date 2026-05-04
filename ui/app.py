"""
Streaming Video-RAG — Streamlit UI
Run: streamlit run ui/app.py
"""

import streamlit as st
from tabs.ingest_tab import render_ingest_tab
from tabs.library_tab import render_library_tab
from tabs.qa_tab import render_qa_tab
from tabs.search_tab import render_search_tab
from tabs.summarize_tab import render_summarize_tab
from utils import api_get, apply_custom_css, load_video_options

st.set_page_config(
    page_title="Video RAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

# ── Sidebar — system status ───────────────────────────────────────────────────

with st.sidebar:
    st.title("🎬 Video RAG")
    st.caption("Streaming Video-RAG System")

    health = api_get("/health")
    if health:
        st.success("API Connected")
        st.markdown(f"""
        | Setting | Value |
        |---|---|
        | LLM | `{health.get("llm_provider")}/{health.get("llm_model")}` |
        | Whisper | `{health.get("whisper_mode")}/{health.get("whisper_model")}` |
        | Embeddings | `{health.get("embedding_mode")}` |
        | Vector Store | `{health.get("vector_store")}` |
        """)
    else:
        st.error("API Offline")

    st.divider()

    # Video library summary
    videos_data = api_get("/videos", params={"status": "indexed", "limit": 100})
    total = videos_data.get("total", 0) if videos_data else 0
    st.metric("Indexed Videos", total)


# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_ingest, tab_qa, tab_search, tab_summarize, tab_library = st.tabs(
    ["📥 Ingest", "❓ Q&A", "🔍 Search", "📝 Summarize", "📚 Library"]
)

video_options, _ = load_video_options()

with tab_ingest:
    render_ingest_tab()

with tab_qa:
    render_qa_tab(video_options)

with tab_search:
    render_search_tab(video_options)

with tab_summarize:
    render_summarize_tab()

with tab_library:
    render_library_tab()
