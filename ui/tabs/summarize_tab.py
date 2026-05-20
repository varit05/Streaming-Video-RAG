"""
Summarize Tab Component
"""

from typing import Any

import streamlit as st
from utils import api_get, api_post


def _get_video_choices(videos: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build video selection choices from video list."""
    return {f"{v['title'][:60]} ({v['id'][:8]})": v for v in videos}


def _render_no_videos_message() -> None:
    """Display message when no indexed videos are available."""
    st.info("No indexed videos yet. Go to the Ingest tab to add some.")


def _render_summary_results(response: dict[str, Any]) -> None:
    """Render the summary and chapter results from API response."""
    st.subheader(f"Summary: {response['title']}")
    st.caption(f"Based on {response['chunk_count']} indexed segments")
    st.markdown(response["overall_summary"])

    chapter_summaries = response.get("chapter_summaries")
    if chapter_summaries:
        _render_chapter_summaries(chapter_summaries)


def _render_chapter_summaries(chapter_summaries: list[dict[str, Any]]) -> None:
    """Render individual chapter summaries as expanders."""
    st.subheader("Chapter Summaries")
    for chapter in chapter_summaries:
        with st.expander(chapter["chapter"]):
            st.write(chapter["summary"])


def _fetch_indexed_videos() -> list[dict[str, Any]]:
    """Fetch list of indexed videos from API."""
    videos_data = api_get("/videos", params={"status": "indexed", "limit": 100})
    if videos_data is None:
        return []
    videos: list[dict[str, Any]] = videos_data.get("videos", [])
    return videos


def _handle_generate_summary(
    video_id: str, include_chapters: bool
) -> dict[str, Any] | None:
    """Call API to generate summary and return response."""
    with st.spinner("Summarizing... this may take a minute for long videos."):
        response: dict[str, Any] | None = api_post(
            "/summarize",
            {"video_id": video_id, "include_chapters": include_chapters},
        )
        return response


def render_summarize_tab() -> None:
    """Render the summarize tab UI."""
    st.header("Summarize a Video")
    st.caption(
        "Generate an overall summary and per-chapter breakdown using map-reduce."
    )

    videos = _fetch_indexed_videos()

    if not videos:
        _render_no_videos_message()
        return

    video_choices = _get_video_choices(videos)
    selected_label = st.selectbox("Select a video", list(video_choices.keys()))
    selected_video = video_choices[selected_label]

    include_chapters = st.checkbox("Include chapter summaries", value=True)

    if st.button("📝 Generate Summary", type="primary"):
        response = _handle_generate_summary(selected_video["id"], include_chapters)
        if response:
            _render_summary_results(response)
