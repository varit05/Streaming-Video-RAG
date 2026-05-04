"""
Library Tab Component
"""
import httpx
import streamlit as st
from utils import api_get, API_BASE


def render_library_tab():
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
                    st.metric("Duration", f"{int(dur // 60)}m {int(dur % 60)}s" if dur else "—")
                with col3:
                    st.metric("Language", v.get("language", "—").upper())

                st.caption(f"ID: `{v['id']}`  ·  Type: `{v['source_type']}`  ·  Status: `{v['status']}`")
                if v.get("source_url"):
                    st.caption(f"Source: {v['source_url'][:80]}")
                if v.get("error_message"):
                    st.error(f"Error: {v['error_message']}")

                if v["status"] == "indexed" and st.button("🗑 Delete", key=f"del_{v['id']}"):
                    result = httpx.delete(f"{API_BASE}/videos/{v['id']}", timeout=30)
                    if result.status_code == 200:
                        st.success("Deleted")
                        st.rerun()
                    else:
                        st.error("Delete failed")
    else:
        st.info("No videos in the library yet. Head to the Ingest tab to add your first video.")