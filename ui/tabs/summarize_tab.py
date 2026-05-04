"""
Summarize Tab Component
"""
import streamlit as st
from utils import api_get, api_post


def render_summarize_tab():
    st.header("Summarize a Video")
    st.caption("Generate an overall summary and per-chapter breakdown using map-reduce.")

    videos_data = api_get("/videos", params={"status": "indexed", "limit": 100})

    if videos_data and videos_data.get("videos"):
        video_choices = {f"{v['title'][:60]} ({v['id'][:8]})": v for v in videos_data["videos"]}
        selected_label = st.selectbox("Select a video", list(video_choices.keys()))
        selected_video = video_choices[selected_label]

        include_chapters = st.checkbox("Include chapter summaries", value=True)

        if st.button("📝 Generate Summary", type="primary"):
            with st.spinner("Summarizing... this may take a minute for long videos."):
                resp = api_post(
                    "/summarize",
                    {
                        "video_id": selected_video["id"],
                        "include_chapters": include_chapters,
                    },
                )

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