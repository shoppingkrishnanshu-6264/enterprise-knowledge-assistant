"""
Streamlit frontend for the Enterprise Knowledge Assistant.

Run from the project root with:
    streamlit run app.py

This UI lets you chat with the agent and see, for each answer:
- the sub-questions the Planner generated
- which tool (vector search / SQL) answered each sub-question
- how many retries the Critic triggered
- the final synthesized answer with citations
"""

import streamlit as st
from src.agents.graph import ask

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="🧠", layout="centered")

st.title("🧠 Enterprise Knowledge Assistant")
st.caption("Agentic RAG over company policies + sales data — powered by a local LLM (Ollama / llama3.1)")

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ..., "trace": {...}}

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About this assistant")
    st.markdown(
        """
        This assistant can answer questions about:
        - **Company policies**: leave, refund, code of conduct, remote work, expenses
        - **Sales data**: revenue, contract value, refund rates by quarter/region/product

        It plans multi-step retrieval, chooses the right tool (document search vs. SQL),
        verifies its own evidence, and cites sources.
        """
    )
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Try asking:")
    st.code("How does our Q2 2025 refund rate compare to our stated refund policy threshold?", language=None)
    st.code("How many days of sick leave do employees get?", language=None)
    st.code("What is the home office stipend for remote employees?", language=None)


# ---------- Render chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander("🔍 Show agent reasoning trace"):
                trace = msg["trace"]
                st.markdown(f"**Retries triggered by critic:** {trace['retry_count']}")
                st.markdown("**Sub-tasks planned:**")
                for task in trace["sub_tasks"]:
                    st.markdown(f"- `{task.get('tool', 'unknown')}` → {task.get('sub_question', '')}")
                st.markdown("**Evidence collected:**")
                for e in trace["evidence"]:
                    source = e.get("source", "unknown")
                    content_preview = str(e.get("content", ""))[:200]
                    st.markdown(f"- **[{e.get('tool')}] {source}**: {content_preview}...")


# ---------- Chat input ----------
user_question = st.chat_input("Ask about company policies or sales data...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Planning, retrieving, and verifying evidence..."):
            result = ask(user_question)
            answer = result["final_answer"]
            st.markdown(answer)

            with st.expander("🔍 Show agent reasoning trace"):
                st.markdown(f"**Retries triggered by critic:** {result['retry_count']}")
                st.markdown("**Sub-tasks planned:**")
                for task in result["sub_tasks"]:
                    st.markdown(f"- `{task.get('tool', 'unknown')}` → {task.get('sub_question', '')}")
                st.markdown("**Evidence collected:**")
                for e in result["evidence"]:
                    source = e.get("source", "unknown")
                    content_preview = str(e.get("content", ""))[:200]
                    st.markdown(f"- **[{e.get('tool')}] {source}**: {content_preview}...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "trace": {
            "retry_count": result["retry_count"],
            "sub_tasks": result["sub_tasks"],
            "evidence": result["evidence"],
        },
    })
