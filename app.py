import streamlit as st

from common import CSS

st.set_page_config(page_title="Telco AI Agents", page_icon="📱", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

page = st.navigation([
    st.Page("cs_page.py", title="GenAI Customer Service Agent", icon="💬", url_path="customer-service", default=True),
    st.Page("hp_page.py", title="AI Hyper-personalisation Agent", icon="🎯", url_path="hyper-personalisation"),
])
page.run()
