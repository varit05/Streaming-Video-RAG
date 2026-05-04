"""
Shared utilities and API helpers for Streaming Video-RAG UI
"""

import os
import time

import httpx
import streamlit as st

API_BASE = os.getenv("UI_API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.getenv("UI_REQUEST_TIMEOUT", 600))


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
    except httpx.TimeoutException:
        st.error("""
        ⏱ Request timed out.
        
        If you are using local Ollama with long videos:
        1. Increase `UI_REQUEST_TIMEOUT` in your .env file
        2. Recommended value: 600 (10 minutes)
        3. Or use the REST API directly instead of the Web UI
        """)
        return None


def api_post(path: str, json: dict[str, object | None] | dict[str, str]):
    try:
        r = httpx.post(f"{API_BASE}{path}", json=json, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:300]}")
        return None
    except httpx.ConnectError:
        st.error("Cannot connect to API. Make sure the FastAPI server is running.")
        return None
    except httpx.TimeoutException:
        st.error(f"""
        ⏱ Request timed out after {REQUEST_TIMEOUT} seconds.
        
        ✅ Solution for local Ollama users:
        - Add/modify `UI_REQUEST_TIMEOUT=600` in your .env file (10 minutes)
        - Restart the Streamlit UI
        - For videos longer than 60 minutes, use the REST API directly
        
        Local processing takes approximately 1x realtime (1hr video = 1hr processing)
        """)
        return None


def poll_job(job_id: str, placeholder: st.delta_generator.DeltaGenerator) -> dict[str, object]:
    """Poll ingestion job until done or error."""
    for _ in range(300):  # max ~5 minutes
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


def load_video_options():
    """Load available videos for selection dropdowns"""
    videos_data = api_get("/videos", params={"status": "indexed", "limit": 100})
    video_options = {"All videos": None}

    if videos_data and videos_data.get("videos"):
        for v in videos_data["videos"]:
            video_options[f"{v['title'][:50]} ({v['id'][:8]})"] = v["id"]

    return video_options, videos_data


def apply_custom_css():
    """Apply custom CSS styles"""
    st.markdown(
        """
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
    """,
        unsafe_allow_html=True,
    )
