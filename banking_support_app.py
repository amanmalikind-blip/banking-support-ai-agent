"""
Banking Customer Support AI Agent — Streamlit UI  (Capstone Part 2)
==================================================================

Interactive dashboard for the multi-agent support system. Run with:

    pip install -r requirements.txt
    streamlit run banking_support_app.py

Put your OpenAI key in a `.env` file (OPENAI_API_KEY=sk-...) in this folder.
The app opens at http://localhost:8501
"""

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from support_agents_core import Orchestrator, evaluate_system, EVAL_CASES

load_dotenv()
st.set_page_config(page_title="Banking Support AI Agent", page_icon="🏦", layout="wide")

# On Streamlit Community Cloud the API key is provided via Secrets. Surface it as
# an environment variable so the agents (which read os.getenv) pick it up.
try:
    if "OPENAI_API_KEY" in st.secrets and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

HAS_KEY = bool(os.getenv("OPENAI_API_KEY"))


@st.cache_resource(show_spinner="Starting agents & database…")
def get_orchestrator():
    # One orchestrator per session so the logger and DB persist across clicks.
    return Orchestrator()


orch = get_orchestrator()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🏦 Banking Support")
st.sidebar.caption("Multi-Agent Customer Support AI")
page = st.sidebar.radio("Navigate",
    ["Support Assistant", "Support Tickets (DB)", "Logs & Debug", "Evaluation"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Agents**")
st.sidebar.markdown("• Classifier Agent\n\n• Feedback Handler Agent\n\n• Query Handler Agent")
st.sidebar.markdown("---")
if HAS_KEY:
    st.sidebar.success("OpenAI key detected ✓ (LLM agents active)")
else:
    st.sidebar.info("No OPENAI_API_KEY — running with the built-in rule-based "
                    "fallback. Add a key in .env for full LLM responses.")


CLR = {"Positive Feedback": "🟢", "Negative Feedback": "🔴", "Query": "🔵"}


def render_result(res):
    c1, c2, c3 = st.columns(3)
    c1.metric("Classification", f"{CLR.get(res['classification'],'')} {res['classification']}")
    c2.metric("Routed to", res["agent"])
    c3.metric("Ticket", res["ticket_number"] or "—")
    st.markdown("**Agent response**")
    if res["success"]:
        st.success(res["response"])
    else:
        st.warning(res["response"])


# ---------------------------------------------------------------- Assistant
if page == "Support Assistant":
    st.title("Support Assistant")
    st.caption("Type a customer message. The Classifier Agent routes it to the "
               "right handler — thank-you, new ticket, or ticket status lookup.")

    name = st.text_input("Customer name (optional)", "")
    msg = st.text_area("Customer message",
                       "My debit card replacement still hasn't arrived.", height=80)
    if st.button("Send to agents", type="primary"):
        with st.spinner("Agents working…"):
            res = orch.process(msg, customer_name=name or None)
        render_result(res)

    st.markdown("---")
    st.subheader("Test scenarios")
    st.caption("One-click examples for each agent role.")
    ex = {
        "🟢 Positive feedback": "Thanks for sorting out my net banking login issue!",
        "🔴 Negative feedback": "My debit card replacement still hasn't arrived.",
        "🔵 Ticket query": "Could you check the status of ticket 650932?",
    }
    cols = st.columns(3)
    for (label, text), col in zip(ex.items(), cols):
        if col.button(label):
            with st.spinner("Agents working…"):
                res = orch.process(text)
            st.write(f"*Input:* {text}")
            render_result(res)


# ---------------------------------------------------------------- DB view
elif page == "Support Tickets (DB)":
    st.title("Support Tickets Database")
    st.caption("Live contents of the `support_tickets` table (SQLite). New "
               "negative-feedback tickets are inserted here automatically.")
    df = pd.DataFrame(orch.db.all_tickets())
    if not df.empty:
        df = df[["ticket_number", "customer_name", "issue", "status", "created_at"]]
    st.dataframe(df, width="stretch")
    counts = df["status"].value_counts().to_dict() if not df.empty else {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Total tickets", len(df))
    c2.metric("Resolved", counts.get("Resolved", 0))
    c3.metric("Open / In progress",
              counts.get("Unresolved", 0) + counts.get("In Progress", 0))


# ---------------------------------------------------------------- Logs
elif page == "Logs & Debug":
    st.title("Logs & Debugging View")
    st.caption("Trace of every interaction: classification, agent, action, "
               "ticket, and success — plus the agent success rate.")
    logs = orch.logger.as_dicts()
    if not logs:
        st.info("No interactions yet. Use the Support Assistant to generate logs.")
    else:
        st.metric("Agent success rate", f"{orch.logger.success_rate()*100:.0f}%")
        df = pd.DataFrame(logs)[["timestamp", "user_input", "classification",
                                 "agent", "action", "ticket_number", "success"]]
        st.dataframe(df, width="stretch")
        with st.expander("Full responses (prompt trace)"):
            for e in reversed(logs):
                st.markdown(f"**[{e['classification']}] → {e['agent']}**  "
                            f"(ticket: {e['ticket_number'] or '—'})")
                st.write(e["response"]); st.markdown("---")


# ---------------------------------------------------------------- Evaluation
elif page == "Evaluation":
    st.title("Model Evaluation")
    st.caption("Classification accuracy, agent routing success rate, and "
               "per-test-case coverage of the Classifier Agent.")
    if st.button("Run evaluation", type="primary"):
        with st.spinner("Evaluating classifier over test cases…"):
            ev = evaluate_system(orch)
        c1, c2, c3 = st.columns(3)
        c1.metric("Classification accuracy", f"{ev['classification_accuracy']*100:.0f}%")
        c2.metric("Routing success rate", f"{ev['routing_success_rate']*100:.0f}%")
        c3.metric("Cases", f"{ev['n_correct']}/{ev['n']}")
        df = pd.DataFrame(ev["details"])
        df["result"] = df["correct"].map({True: "✅", False: "❌"})
        st.dataframe(df[["message", "expected", "predicted", "result"]],
                     width="stretch")
