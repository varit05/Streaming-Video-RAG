"""
Streaming Video-RAG — Streamlit UI
Run: streamlit run ui/app.py
"""

import os
import time

import httpx
import streamlit as st

API_BASE = os.getenv("UI_API_BASE_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video RAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .result-card {
        background: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 4px;
    }
    .citation-badge {
        background: #e3f2fd;
        color: #1565c0;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 600;
    }
    .timestamp-badge {
        background: #f3e5f5;
        color: #6a1b9a;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85em;
    }
    .score-badge {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path: str, **kwargs):
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:200]}")
        return None
    except httpx.ConnectError:
        st.error("Cannot connect to API. Make sure the FastAPI server is running.")
        return None


def api_post(path: str, json: dict):
    try:
        r = httpx.post(f"{API_BASE}{path}", json=json, timeout=120)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:300]}")
        return None
    except httpx.ConnectError:
        st.error("Cannot connect to API. Make sure the FastAPI server is running.")
        return None


def poll_job(job_id: str, placeholder) -> dict:
    """Poll ingestion job until done or error."""
    for _ in range(300):   # max ~5 minutes
        data = api_get(f"/ingest/{job_id}")
        if not data:
            return {}
        status = data.get("status", "")
        msg = data.get("progress_message", "Working...")
        placeholder.info(f"⏳ {msg} (status: {status})")
        if status in ("done", "error"):
            return data
        time.sleep(2)
    return {}


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
        | LLM | `{health.get('llm_provider')}/{health.get('llm_model')}` |
        | Whisper | `{health.get('whisper_mode')}/{health.get('whisper_model')}` |
        | Embeddings | `{health.get('embedding_mode')}` |
        | Vector Store | `{health.get('vector_store')}` |
        """)
    else:
        st.error("API Offline")

    st.divider()

    # Video library summary
    videos_data = api_get("/videos", params={"status": "indexed", "limit": 100})
    total = videos_data.get("total", 0) if videos_data else 0
    st.metric("Indexed Videos", total)


# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_ingest, tab_qa, tab_search, tab_summarize, tab_library = st.tabs([
    "📥 Ingest", "❓ Q&A", "🔍 Search", "📝 Summarize", "📚 Library"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: INGEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.header("Ingest a Video")
    st.caption("Add a video source to your RAG library. Supports YouTube URLs, local files, live streams, and video APIs.")

    source_type = st.radio(
        "Source type",
        ["YouTube / URL", "Local File Path", "Live Stream (HLS/RTMP)", "Video API (Vimeo/Twitch)"],
        horizontal=True,
    )

    source = st.text_input(
        "Source",
        placeholder={
            "YouTube / URL": "https://www.youtube.com/watch?v=...",
            "Local File Path": "/path/to/video.mp4",
            "Live Stream (HLS/RTMP)": "https://example.com/stream.m3u8",
            "Video API (Vimeo/Twitch)": "https://vimeo.com/123456789",
        }.get(source_type, ""),
    )

    col1, col2 = st.columns(2)
    with col1:
        language = st.text_input("Language (optional)", placeholder="en, es, fr, de, ...")
    with col2:
        if "Video API" in source_type:
            platform = st.selectbox("Platform", ["vimeo", "twitch"])
        else:
            platform = None

    source_type_map = {
        "YouTube / URL": "youtube",
        "Local File Path": "local_file",
        "Live Stream (HLS/RTMP)": "live_stream",
        "Video API (Vimeo/Twitch)": "video_api",
    }

    if st.button("🚀 Start Ingestion", type="primary", disabled=not source.strip()):
        payload = {
            "source": source.strip(),
            "source_type": source_type_map[source_type],
            "language": language.strip() or None,
            "platform": platform,
        }
        resp = api_post("/ingest", payload)
        if resp:
            job_id = resp["job_id"]
            st.success(f"Job submitted — ID: `{job_id}`")
            status_placeholder = st.empty()
            final = poll_job(job_id, status_placeholder)
            if final.get("status") == "done":
                status_placeholder.success(f"✅ Ingestion complete! Video ID: `{final.get('video_id')}`")
                st.balloons()
            elif final.get("status") == "error":
                status_placeholder.error(f"❌ Error: {final.get('error_message')}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Q&A
# ══════════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.header("Ask a Question")
    st.caption("Get answers grounded in your indexed video content, with timestamped citations.")

    # Optional: scope to a specific video
    video_options = {"All videos": None}
    if videos_data and videos_data.get("videos"):
        for v in videos_data["videos"]:
            video_options[f"{v['title'][:50]} ({v['id'][:8]})"] = v["id"]

    selected_video_label = st.selectbox("Scope (optional)", list(video_options.keys()))
    selected_video_id = video_options[selected_video_label]

    question = st.text_area(
        "Your question",
        placeholder="What did the speaker say about transformer attention mechanisms?",
        height=80,
    )

    top_k = st.slider("Chunks to retrieve", 3, 20, 5)

    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving and generating answer..."):
            resp = api_post("/query", {
                "question": question.strip(),
                "video_id": selected_video_id,
                "top_k": top_k,
            })

        if resp:
            st.subheader("Answer")
            st.markdown(resp["answer"])

            if resp.get("sources"):
                st.subheader(f"Sources ({len(resp['sources'])})")
                for s in resp["sources"]:
                    deep_link = s.get("deep_link") or s.get("source_url", "#")
                    st.markdown(f"""
<div class="result-card">
<span class="citation-badge">[{s['index']}]</span>&nbsp;
<strong>{s['title'][:60]}</strong>&nbsp;
<span class="timestamp-badge">⏱ {s['start_ts']} – {s['end_ts']}</span>&nbsp;
<span class="score-badge">score: {s['score']}</span><br>
<small><a href="{deep_link}" target="_blank">🔗 Open at timestamp</a></small>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.header("Semantic Search")
    st.caption("Find relevant video segments by meaning, not just keywords.")

    search_video_label = st.selectbox(
        "Scope (optional)", list(video_options.keys()), key="search_scope"
    )
    search_video_id = video_options[search_video_label]

    query = st.text_input("Search query", placeholder="neural network training techniques")

    col1, col2 = st.columns(2)
    with col1:
        top_k_search = st.slider("Max results", 3, 30, 10, key="search_topk")
    with col2:
        min_score = st.slider("Min relevance score", 0.0, 1.0, 0.0, 0.05)

    if st.button("🔍 Search", type="primary", disabled=not query.strip()):
        with st.spinner("Searching..."):
            resp = api_post("/search", {
                "query": query.strip(),
                "video_id": search_video_id,
                "top_k": top_k_search,
                "min_score": min_score,
            })

        if resp:
            st.caption(f"Found {resp['total_results']} results for \"{resp['query']}\"")
            for r in resp["results"]:
                deep_link = r.get("deep_link") or r.get("source_url", "#")
                with st.expander(
                    f"#{r['rank']}  {r['title'][:50]}  ·  {r['start_ts']}–{r['end_ts']}  ·  score: {r['score']}",
                    expanded=r["rank"] <= 3,
                ):
                    if r.get("chapter"):
                        st.caption(f"📖 Chapter: {r['chapter']}")
                    st.markdown(f"> {r['text']}")
                    st.markdown(f"[🔗 Open at timestamp]({deep_link})")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: SUMMARIZE
# ══════════════════════════════════════════════════════════════════════════════
with tab_summarize:
    st.header("Summarize a Video")
    st.caption("Generate an overall summary and per-chapter breakdown using map-reduce.")

    if videos_data and videos_data.get("videos"):
        video_choices = {
            f"{v['title'][:60]} ({v['id'][:8]})": v
            for v in videos_data["videos"]
        }
        selected_label = st.selectbox("Select a video", list(video_choices.keys()))
        selected_video = video_choices[selected_label]

        include_chapters = st.checkbox("Include chapter summaries", value=True)

        if st.button("📝 Generate Summary", type="primary"):
            with st.spinner("Summarizing... this may take a minute for long videos."):
                resp = api_post("/summarize", {
                    "video_id": selected_video["id"],
                    "include_chapters": include_chapters,
                })

            if resp:
                st.subheader(f"Summary: {resp['title']}")
                st.caption(f"Based on {resp['chunk_count']} indexed segments")
                st.markdown(resp["overall_summary"])

                if resp.get("chapter_summaries"):
                    st.subheader("Chapter Summaries")
                    for ch in resp["chapter_summaries"]:
                        with st.expander(ch["chapter"]):
                            st.write(ch["summary"])
    else:
        st.info("No indexed videos yet. Go to the Ingest tab to add some.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_library:
    st.header("Video Library")
    st.caption("All videos that have been ingested into your RAG system.")

    if st.button("🔄 Refresh"):
        st.rerun()

    all_videos = api_get("/videos", params={"limit": 100})
    if all_videos and all_videos.get("videos"):
        for v in all_videos["videos"]:
            status_icon = {
                "indexed": "✅",
                "processing": "⏳",
                "error": "❌",
                "pending": "🕐",
            }.get(v["status"], "❓")

            with st.expander(f"{status_icon} {v['title'][:70]}", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Chunks", v.get("chunk_count", 0))
                with col2:
                    dur = v.get("duration_seconds")
                    st.metric("Duration", f"{int(dur//60)}m {int(dur%60)}s" if dur else "—")
                with col3:
                    st.metric("Language", v.get("language", "—").upper())

                st.caption(f"ID: `{v['id']}`  ·  Type: `{v['source_type']}`  ·  Status: `{v['status']}`")
                if v.get("source_url"):
                    st.caption(f"Source: {v['source_url'][:80]}")
                if v.get("error_message"):
                    st.error(f"Error: {v['error_message']}")

                if v["status"] == "indexed":
                    if st.button(f"🗑 Delete", key=f"del_{v['id']}"):
                        result = httpx.delete(f"{API_BASE}/videos/{v['id']}", timeout=30)
                        if result.status_code == 200:
                            st.success("Deleted")
                            st.rerun()
                        else:
                            st.error("Delete failed")
    else:
        st.info("No videos in the library yet. Head to the Ingest tab to add your first video.")
