import re

import streamlit as st

from common import md, secret
from research import FACTS, REPORTS, respond, source

st.title("📚 Telco AI Research Agent")
st.write("Ask anything about how telecom companies are using AI agents. This agent answers **only from the industry "
         "reports shared with it**, and every fact shows **which report and page** it came from **[F…]**. "
         "If the reports don't cover a question, it says so instead of guessing.")
st.markdown("Reports in the library: " + "".join(
    f"<span class='chip live'>{r['publisher']}: {r['title']}</span>" for r in REPORTS.values()), unsafe_allow_html=True)
st.caption("Demo by Rohit Chauhan. The library holds short notes in our own words with page numbers, not the full reports.")

examples = ["Where do telcos use AI agents the most?", "How should a customer service agent be different from a chatbot?",
            "How can AI agents help network operations?", "What results have real telcos reported?"]
cols = st.columns(len(examples))
pressed = [q for col, q in zip(cols, examples) if col.button(q, use_container_width=True)]
question = st.chat_input("Ask about telco AI, e.g. how are telcos using AI for security?") or (pressed[0] if pressed else None)
chat = st.session_state.setdefault("research_chat", [])


def show_sources(r):
    for f in r["cited"]:
        st.markdown(md(f"<span class='cite'>{f['id']}</span> {f['text']}  \n*{source(f)}*"), unsafe_allow_html=True)


box = st.container(height=560)
with box:
    if not chat:
        with st.chat_message("assistant"):
            st.markdown("Hi, I'm the Telco AI Research Agent. Ask me what the industry reports say about AI in telecom.")
    if question:
        history = [{"role": m["role"], "content": m["content"]} for m in chat]
        chat.append({"role": "user", "content": question})
        with st.spinner("Searching the reports, then asking the AI..."):
            result = respond(question, history, secret("GROQ_API_KEY"), secret("GEMINI_API_KEY"))
        chat.append({"role": "assistant", "content": result["answer"], "result": result})
    for m in chat:
        with st.chat_message(m["role"]):
            r = m.get("result")
            if not r:
                st.markdown(md(m["content"]))
                continue
            # Each [F..] is shown with its page, e.g. [F21 · p.19].
            text = re.sub(r"\[(F\d+)\]", lambda x: f"[{x.group(1)} · p.{FACTS[x.group(1)]['page']}]"
                          if x.group(1) in FACTS else x.group(0), m["content"])
            st.markdown(md(text))
            st.caption(f"Answered by: {r['engine']}")
            if r["cited"]:
                with st.expander("Sources (report and page)"):
                    show_sources(r)
            if r["prompt"]:
                with st.expander("Exact prompt sent to the AI"):
                    st.code(r["prompt"], language=None)

st.divider()
st.subheader("How this agent works")
st.markdown("""
1. **Library**: each report Rohit shares is turned into short fact notes, each with its page number. New reports make the agent know more.
2. **Retrieve (RAG)**: a keyword search (BM25) picks the notes that best match the question.
3. **Generate**: a free LLM (GPT-OSS 120B on Groq, or Gemini Flash) writes a short answer **from those notes only**, citing each one.
4. **Check**: every citation shows the report and page, so anyone can verify it.
5. **Guardrails**: if no note matches, it says the reports don't cover it. If the AI is unavailable, it lists the matching notes instead.

**Business value**: teams get quick, sourced answers from industry research instead of reading 40-page reports, and every claim can be checked.
""")
with st.expander(f"All notes in the library ({len(FACTS)})"):
    for f in FACTS.values():
        st.markdown(md(f"**{f['id']} {f['topic']}** ({source(f)}): {f['text']}"))
