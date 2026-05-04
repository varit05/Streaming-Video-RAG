"""
Search Tab Component
"""

import streamlit as st
from utils import api_post


def render_search_tab(video_options):
    st.header("Semantic Search")
    st.caption("Find relevant video segments by meaning, not just keywords.")

    search_video_label = st.selectbox("Scope (optional)", list(video_options.keys()), key="search_scope")
    search_video_id = video_options[search_video_label]

    query = st.text_input("Search query", placeholder="neural network training techniques")

    col1, col2 = st.columns(2)
    with col1:
        top_k_search = st.slider("Max results", 3, 30, 10, key="search_topk")
    with col2:
        min_score = st.slider("Min relevance score", 0.0, 1.0, 0.0, 0.05)

    if st.button("🔍 Search", type="primary", disabled=not query.strip()):
        with st.spinner("Searching..."):
            resp = api_post(
                "/search",
                {
                    "query": query.strip(),
                    "video_id": search_video_id,
                    "top_k": top_k_search,
                    "min_score": min_score,
                },
            )

        if resp:
            st.caption(f'Found {resp["total_results"]} results for "{resp["query"]}"')
            for r in resp["results"]:
                deep_link = r.get("deep_link") or r.get("source_url", "#")
                with st.expander(
                    f"#{r['rank']}  {r['title'][:50]}  ·  {r['start_ts']} - {r['end_ts']}  ·  score: {r['score']}",
                    expanded=r["rank"] <= 3,
                ):
                    if r.get("chapter"):
                        st.caption(f"📖 Chapter: {r['chapter']}")
                    st.markdown(f"> {r['text']}")
                    st.markdown(f"[🔗 Open at timestamp]({deep_link})")
