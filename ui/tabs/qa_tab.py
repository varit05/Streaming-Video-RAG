"""
Q&A Tab Component
"""
import streamlit as st
from utils import api_post


def render_qa_tab(video_options):
    st.header("Ask a Question")
    st.caption("Get answers grounded in your indexed video content, with timestamped citations.")

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
            resp = api_post(
                "/query",
                {
                    "question": question.strip(),
                    "video_id": selected_video_id,
                    "top_k": top_k,
                },
            )

        if resp:
            st.subheader("Answer")
            st.markdown(resp["answer"])

            if resp.get("sources"):
                st.subheader(f"Sources ({len(resp['sources'])})")
                for s in resp["sources"]:
                    deep_link = s.get("deep_link") or s.get("source_url", "#")
                    st.markdown(
                        f"""
<div class="result-card">
<span class="citation-badge">[{s["index"]}]</span>&nbsp;
<strong>{s["title"][:60]}</strong>&nbsp;
<span class="timestamp-badge">⏱ {s["start_ts"]} - {s["end_ts"]}</span>&nbsp;
<span class="score-badge">score: {s["score"]}</span><br>
<small><a href="{deep_link}" target="_blank">🔗 Open at timestamp</a></small>
</div>
""",
                        unsafe_allow_html=True,
                    )