import os

import streamlit as st

from agent import CUSTOMERS, INTENTS, SECTIONS, TARIFF, money, respond


def secret(name):
    try:
        return st.secrets.get(name) or os.environ.get(name)
    except Exception:  # no secrets file when run locally
        return os.environ.get(name)


def md(text):
    # Streamlit reads "$...$" as maths, so dollar signs are escaped before display.
    return text.replace("$", "\\$")


st.set_page_config(page_title="GenAI Customer Service Agent", page_icon="💬", layout="wide")
st.markdown("""<style>
.chip { display:inline-block; border-radius:999px; padding:2px 12px; margin:0 6px 6px 0; font-size:0.85rem;
        border:1px solid #c9d8f5; background:#eef3fc; color:#0b2e6b; }
.chip.live { background:#1a56db; color:#fff; border-color:#1a56db; }
.cite { background:#e8f0fe; color:#0b3d91; border-radius:4px; padding:0 4px; font-weight:600; }
</style>""", unsafe_allow_html=True)

st.title("💬 GenAI Customer Service Agent")
st.write("A chat agent for a mobile operator. It works out what the customer needs, then answers **only from "
         "the customer's own records and the price guide**, quoting the bill lines **[B…]**, payments **[PAY…]** "
         "and price guide sections **[T…]** it used.")
st.markdown("Use cases: " + "".join(
    f"<span class='chip {'live' if status == 'Live' else ''}'>{label}: {'live' if status == 'Live' else 'coming next'}</span>"
    for key, (label, status) in INTENTS.items() if key != "other"), unsafe_allow_html=True)
st.caption("Demo by Rohit Chauhan. All customers, bills and prices are made up.")

left, right = st.columns([2, 3], gap="large")

with left:
    names = {f"{c['name']}: {c['story']}": c for c in CUSTOMERS}
    choice = st.selectbox("Chat as a sample customer", list(names))
    customer = names[choice]
    if st.session_state.get("customer") != customer["id"]:
        st.session_state.customer, st.session_state.chat = customer["id"], []
    cur, prev = customer["bills"]["current"], customer["bills"]["previous"]
    st.subheader(f"{cur['period']} bill")
    st.caption(md(customer["plan"]))
    st.table([{"Line": l["id"], "Item": l["item"], "Amount": money(l["amount"])} for l in cur["lines"]])
    t_cur = sum(l["amount"] for l in cur["lines"])
    t_prev = sum(l["amount"] for l in prev["lines"])
    a, b = st.columns(2)
    a.metric("This bill", money(t_cur), f"{money(t_cur - t_prev)} vs {prev['period']}", delta_color="inverse")
    b.metric("Last bill", money(t_prev))
    if customer["payments"]:
        st.caption(md("Card payments: " + "; ".join(
            f"[{p['id']}] {p['date']} {p['item']} {money(p['amount'])}" for p in customer["payments"])))

with right:
    st.subheader("Chat with the agent")
    examples = ["Why is my bill higher this month?", "I think I was charged twice", "How do I recharge?",
                "My home WiFi keeps dropping"]
    cols = st.columns(len(examples))
    pressed = [q for col, q in zip(cols, examples) if col.button(q, use_container_width=True)]
    clicked = pressed[0] if pressed else None
    typed = st.chat_input("Type your question, e.g. why is my bill so high?")
    question = typed or clicked

    box = st.container(height=520)
    with box:
        if not st.session_state.chat:
            with st.chat_message("assistant"):
                st.markdown(f"Hi {customer['name']}, I'm your virtual assistant. How can I help today?")
        if question:
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat]
            st.session_state.chat.append({"role": "user", "content": question})
            with st.spinner("Checking your account and the price guide..."):
                result = respond(customer, question, history, secret("GROQ_API_KEY"), secret("GEMINI_API_KEY"))
            st.session_state.chat.append({"role": "assistant", "content": result["answer"], "result": result})
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(md(m["content"]))
                r = m.get("result")
                if not r:
                    continue
                label, status = INTENTS[r["intent"]]
                st.caption(f"Understood as: {label} · Answered by: {r['engine']}")
                if r["ticket"]:
                    st.info(md(f"**Complaint ticket {r['ticket']} logged.** Handed to the Master AI Agent with: "
                            f"customer {customer['id']} ({customer['segment']}), bill {money(r['diff']['current_total'])} "
                            f"vs usual {money(r['diff']['previous_total'])}, suspected double charge "
                            f"{money(sum(l['amount'] for l, _ in r['diff']['doubles']))}. (Simulated, nothing real is sent.)"))
                if "diff" in r:
                    with st.expander("Sources the agent used"):
                        st.markdown("**Bill changes worked out by code (no AI, so the maths is exact)**")
                        st.table([{"Line": c["line"]["id"], "Item": c["line"]["item"], "Before": money(c["before"]),
                                   "Now": money(0 if c["status"] == "gone" else c["line"]["amount"]),
                                   "Change": money(c["change"])} for c in r["diff"]["changes"]])
                        st.markdown("**Price guide sections retrieved**")
                        for s in r["sections"]:
                            st.markdown(md(f"<span class='cite'>{s['id']}</span> **{s['title']}**: {s['text']}"),
                                        unsafe_allow_html=True)
                    with st.expander("Exact prompt sent to the AI"):
                        st.code(r["prompt"], language=None)

st.divider()
st.subheader("How this agent works")
st.markdown("""
1. **Understand**: a router works out what the customer needs: bill issue, recharge, internet/WiFi, or other.
2. **Check the facts in code**: for a bill issue, plain Python compares this bill with last month's and with card payments already made, so every number is exact and double charges are caught.
3. **Retrieve (RAG)**: it pulls the price guide sections linked to each changed line, plus any that match the question (BM25 keyword search).
4. **Generate**: a free LLM (Llama 3.3 70B on Groq, or Gemini Flash) writes a plain-English answer **from those sources only**, citing each one.
5. **Act**: if something is wrong, it logs a complaint ticket and hands the case to the **Master AI Agent** (simulated).
6. **Guardrails**: low temperature, "answer only from sources" rules, and a built-in no-AI explainer if the AI is unavailable, so the demo always works.

**Business value**: fewer "bill shock" calls to customer support (lower OPEX), faster answers, consistent explanations that match the tariff, and billing errors caught early (revenue protected, customers kept).
""")
with st.expander(f"Full price guide: {TARIFF['name']}"):
    for s in SECTIONS.values():
        st.markdown(md(f"**{s['id']} {s['title']}**: {s['text']}"))
