import os

import streamlit as st

CSS = """<style>
.chip { display:inline-block; border-radius:999px; padding:2px 12px; margin:0 6px 6px 0; font-size:0.85rem;
        border:1px solid #c9d8f5; background:#eef3fc; color:#0b2e6b; }
.chip.live { background:#1a56db; color:#fff; border-color:#1a56db; }
.cite { background:#e8f0fe; color:#0b3d91; border-radius:4px; padding:0 4px; font-weight:600; }
</style>"""


def secret(name):
    try:
        return st.secrets.get(name) or os.environ.get(name)
    except Exception:  # no secrets file when run locally
        return os.environ.get(name)


def md(text):
    # Streamlit reads "$...$" as maths, so dollar signs are escaped before display.
    return text.replace("$", "\\$")
