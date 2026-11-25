"""Streamlit entry-point that serves the Loan Document AI application.

Architecture:
1. Checks if the backend server (FastAPI at 127.0.0.1:8000) is running.
   If not yet running, starts it in a background thread and shows a visible
   loading status until all models finish loading.
2. Embeds the application in a full-height iframe pointing to http://127.0.0.1:8000,
   allowing all CSS, Lucide icons, sample documents, and API endpoints
   (/api/reconstruct, /api/vae/scenarios, /api/cgan/synthesize, /api/diffusion/synthesize, /api/extract)
   to execute seamlessly on the exact same origin.
"""

import os
import sys
import time
import socket
import threading
import streamlit as st
import streamlit.components.v1 as components

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", 8000))
_SERVER_LOCK = threading.Lock()
_SERVER_STARTED = False

st.set_page_config(
    page_title="Loan Document AI - Autoencoder & Field Extraction",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Clean, full-bleed styling for Streamlit
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header[data-testid="stHeader"] {display: none !important;}
    .stDeployButton {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    .stApp {
        background: #ffffff;
    }
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }
    .stMainBlockContainer {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }
    iframe {
        border: none !important;
        width: 100% !important;
        min-height: 98vh !important;
        height: 1000px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _is_server_up(host: str, port: int) -> bool:
    """Check if the backend server is reachable."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _start_fastapi(host: str, port: int) -> None:
    """Run the FastAPI / uvicorn server in a daemon thread."""
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        log_level="info",
    )


def _ensure_backend_running(host: str, port: int, timeout: float = 90.0) -> bool:
    """Ensure the FastAPI server is running, starting it in a thread if needed."""
    global _SERVER_STARTED
    if _is_server_up(host, port):
        return True

    with _SERVER_LOCK:
        if not _SERVER_STARTED and not _is_server_up(host, port):
            _SERVER_STARTED = True
            t = threading.Thread(target=_start_fastapi, args=(host, port), daemon=True)
            t.start()

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_server_up(host, port):
            return True
        time.sleep(1.0)
    return False


# Check backend availability
if not _is_server_up(BACKEND_HOST, BACKEND_PORT):
    with st.spinner("Initializing Generative AI Models (Autoencoder, VAE, Layout Transformer, cGAN, Diffusion)..."):
        ready = _ensure_backend_running(BACKEND_HOST, BACKEND_PORT, timeout=90.0)
else:
    ready = True

if not ready:
    st.error(f"⚠️ Backend server failed to start on port {BACKEND_PORT}. Please verify console logs.")
    st.stop()

# Embed the web app directly into Streamlit
backend_url = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

if hasattr(st, "iframe"):
    st.iframe(src=backend_url, height=1000, width="stretch")
else:
    components.iframe(src=backend_url, height=1000, scrolling=True)
