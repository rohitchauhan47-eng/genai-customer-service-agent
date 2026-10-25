import streamlit as st

from agent import money
from common import md, secret
from personalise import CASES, CATALOGUE, recommend

st.title("🎯 AI Hyper-personalisation Agent")
st.write("When a customer case comes in, this agent picks the **right remedy for that customer**: what they are owed, "
         "plus one extra offer that suits how they use their phone and is **worth more than it costs**. "
         "A **person approves** it before anything reaches the customer.")
st.caption("Demo by Rohit Chauhan. All customers, cases and offers are made up. Nothing real is sent.")

labels = {f"{c['case']}: {c['summary']}": c for c in CASES}
case = labels[st.selectbox("Case handed over by the GenAI Customer Service Agent", list(labels))]
state = st.session_state.setdefault("hp", {})
if state.get("case") != case["case"]:
    state.clear()
    state["case"] = case["case"]

left, right = st.columns([2, 3], gap="large")
with left:
    st.subheader(f"Customer {case['customer']}")
    for f in case["facts"]:
        st.markdown(md(f"<span class='cite'>{f['id']}</span> {f['text']}"), unsafe_allow_html=True)

with right:
    st.subheader("Agent's recommendation")
    if st.button("Recommend a remedy", type="primary") or "rec" in state:
        if "rec" not in state:
            with st.spinner("Checking the rules, working out the value of each offer, then asking the AI..."):
                state["rec"] = recommend(case, secret("GROQ_API_KEY"), secret("GEMINI_API_KEY"))
        rec, pl = state["rec"], state["rec"]["plan"]
        package = pl["owed"] + ([rec["pick"]] if rec["pick"] else [])
        for r in package:
            st.markdown(md(f"✅ **{r['offer']['name']}** <span class='cite'>{r['offer']['id']}</span> · cost {money(r['cost'])}"),
                        unsafe_allow_html=True)
        if not package:
            st.markdown("No remedy needed.")
        cost = sum(r["cost"] for r in package)
        a, b = st.columns(2)
        a.metric("Total cost", money(cost))
        if rec["pick"]:
            b.metric("Value of the extra offer (a year)", money(rec["pick"]["value"]))
        st.markdown(md(f"**Why:** {rec['reason']}"))
        st.caption(f"Answered by: {rec['engine']}")

        st.markdown("#### 👤 Supervisor approval")
        if "decision" not in state:
            msg = st.text_area("Message to the customer (you can edit it)", rec["message"], height=120)
            c1, c2 = st.columns(2)
            if c1.button("Approve and send", type="primary", use_container_width=True):
                state["decision"], state["final"] = "approved", msg
                st.rerun()
            if c2.button("Reject", use_container_width=True):
                state["decision"] = "rejected"
                st.rerun()
        elif state["decision"] == "approved":
            st.success("Approved by supervisor. Actions (simulated, nothing real is sent):")
            for r in package:
                st.markdown(md(f"- {r['offer']['name']}: {money(r['cost'])} queued"))
            st.markdown(f"- Message sent to customer {case['customer']}:")
            st.info(md(state["final"]))
            st.markdown(f"- Case **{case['case']}** closed and logged for the Master AI Agent.")
        else:
            st.warning("Rejected. The case goes back to the team for a person to decide. Nothing was sent.")
        if "decision" in state and st.button("Start again"):
            state.pop("decision", None)
            state.pop("rec", None)
            st.rerun()

        with st.expander("How the offers were checked (rules and value, worked out in code)"):
            st.markdown(md(f"Goodwill budget for this segment: **{money(pl['budget'])}**. "
                           "Value = yearly spend × expected cut in churn risk − cost."))
            st.table([{"Offer": f"{r['offer']['id']} {r['offer']['name']}", "Cost": money(r["cost"]),
                       "Value": money(r["value"]), "Result": "Allowed"} for r in pl["options"]] +
                     [{"Offer": f"{r['offer']['id']} {r['offer']['name']}", "Cost": money(r["cost"]),
                       "Value": money(r["value"]), "Result": f"Ruled out: {r['why']}"} for r in pl["ruled_out"]])
        with st.expander("Exact prompt sent to the AI"):
            st.code(rec["prompt"], language=None)

st.divider()
st.subheader("How this agent works")
st.markdown("""
1. **Case in**: the GenAI Customer Service Agent hands over a case (for example, a confirmed double charge), with facts about the customer. Churn risk would come from the AI Churn Prediction Agent.
2. **Rules first (code)**: anything the customer is owed is always added, for the exact amount: a refund of a billing error, or credit for a network outage.
3. **Value check (code)**: each extra offer is checked against the customer's profile and the goodwill budget for their segment, and kept only if it is worth more than it costs (yearly spend × churn-risk cut − cost).
4. **GenAI decides and writes**: a free LLM picks one of the allowed offers (it cannot invent one), explains why with citations, and drafts a warm message to the customer.
5. **Human in the loop**: a supervisor approves, edits or rejects. Nothing reaches the customer without a person's OK.

**Business value**: the right remedy for each customer instead of one-size-fits-all credits, goodwill spend kept within budget, and fewer valuable customers lost (revenue protected).
""")
with st.expander("Full offer catalogue"):
    for o in CATALOGUE["offers"]:
        st.markdown(md(f"**{o['id']} {o['name']}**: {o['rule']}"))
