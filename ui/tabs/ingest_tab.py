"""
Ingest Tab Component
"""

import streamlit as st
from utils import api_post, poll_job


def render_ingest_tab() -> None:
    st.header("Ingest a Video")
    st.caption(
        "Add a video source to your RAG library. Supports YouTube URLs, local files, live streams, and video APIs."
    )

    source_type = st.radio(
        "Source type",
        [
            "YouTube / URL",
            "Local File Path",
            "Live Stream (HLS/RTMP)",
            "Video API (Vimeo/Twitch)",
        ],
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
        language = st.text_input(
            "Language (optional)", placeholder="en, es, fr, de, ..."
        )
    with col2:
        platform = (
            st.selectbox("Platform", ["vimeo", "twitch"])
            if "Video API" in source_type
            else None
        )

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
                status_placeholder.success(
                    f"✅ Ingestion complete! Video ID: `{final.get('video_id')}`"
                )
                st.balloons()
            elif final.get("status") == "error":
                status_placeholder.error(f"❌ Error: {final.get('error_message')}")
