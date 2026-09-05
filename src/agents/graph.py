"""
Agent Orchestrator for the Enterprise Knowledge Assistant.
Built with LangGraph. This is the "agentic" core of the system.

Flow:
    PLANNER -> EXECUTOR -> CRITIC -> (loop back to PLANNER if evidence is weak) -> SYNTHESIZER

Nodes:
- planner:      breaks the user question into sub-questions, each tagged with a tool
                 ("vector" for docs, "sql" for structured sales data)
- executor:      calls the right tool for each sub-question and collects evidence
- critic:        checks whether the collected evidence actually answers the question;
                 if not, sends feedback back to the planner (up to MAX_RETRIES times)
- synthesizer:   writes the final, cited answer from all verified evidence

Run this file directly to test the full pipeline with example questions.
"""

import json
import re
import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.tools.vector_tool import vector_search
from src.tools.sql_tool import run_sql_tool

load_dotenv()

# Defensively disable LangSmith tracing — if left enabled without a valid setup,
# it can inject extra request metadata/headers that caused a Unicode encoding crash.
os.environ["LANGCHAIN_TRACING_V2"] = "false"

MODEL_NAME = "openai/gpt-oss-120b"
MAX_RETRIES = 2

# Strip any accidental whitespace/hidden characters from the key and enforce plain ASCII,
# since a stray non-ASCII character in the key or environment previously broke HTTP headers.
_groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
_groq_api_key = _groq_api_key.encode("ascii", "ignore").decode("ascii")

llm = ChatGroq(model=MODEL_NAME, temperature=0, api_key=_groq_api_key)


# ---------- STATE ----------
class AgentState(TypedDict):
    question: str
    sub_tasks: list[dict]        # [{"sub_question": ..., "tool": "vector"|"sql"}]
    evidence: list[dict]         # collected results from tools, tagged with source
    critic_feedback: str         # reason evidence was rejected, fed back to planner on retry
    retry_count: int
    final_answer: str


# ---------- HELPERS ----------
def extract_json(text: str):
    """
    Local LLMs often wrap JSON in markdown fences or add commentary.
    This pulls out the first {...} or [...] block found in the text.
    """
    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if match:
        text = match.group(1)

    return json.loads(text)


# ---------- NODES ----------
def planner_node(state: AgentState) -> AgentState:
    """
    Decomposes the question into sub-tasks, each routed to a tool.
    On retry, incorporates the critic's feedback to reformulate.
    """
    feedback_block = ""
    if state.get("critic_feedback"):
        feedback_block = f"\nNote: a previous attempt was insufficient because: {state['critic_feedback']}\nAdjust your sub-questions to address this."

    prompt = f"""You are a planning agent for a company knowledge assistant.
Break the user's question into 1-3 sub-questions. For each sub-question, decide which tool should answer it:
- "vector": ONLY for questions that reference company policies, HR rules, refund policy text, code of conduct, or remote work rules
- "sql": for questions about numeric sales data, revenue, contract values, refund rates/amounts, by region/quarter/product

IMPORTANT: If the user's question is PURELY about sales numbers (revenue, contract value, refund amounts, counts)
with no mention of a policy, rule, or threshold, use ONLY "sql" sub-tasks. Do NOT invent a policy-related
sub-question that the user did not ask about — this pollutes the evidence with irrelevant information.
Only use "vector" when the question explicitly asks about a policy, rule, or threshold, or when comparing
a number to a stated policy/threshold.

Examples:
- "What was the total contract value in Q1?" -> ONLY sql sub-task(s). Do NOT add a vector sub-task.
- "What is the refund policy threshold?" -> ONLY vector sub-task(s).
- "How does our refund rate compare to the policy threshold?" -> ONE sql sub-task (get the rate) AND ONE vector sub-task (get the threshold).

Respond ONLY with a JSON array, no explanation, in this exact format:
[{{"sub_question": "...", "tool": "vector"}}, {{"sub_question": "...", "tool": "sql"}}]
{feedback_block}

User question: {state['question']}

JSON array:"""

    response = llm.invoke(prompt)

    try:
        sub_tasks = extract_json(response.content)
        if not isinstance(sub_tasks, list) or len(sub_tasks) == 0:
            raise ValueError("Empty or invalid sub_tasks")
    except Exception:
        # Fallback: if planning fails, just try both tools on the original question
        sub_tasks = [
            {"sub_question": state["question"], "tool": "vector"},
            {"sub_question": state["question"], "tool": "sql"},
        ]

    state["sub_tasks"] = sub_tasks
    return state


def executor_node(state: AgentState) -> AgentState:
    """
    Calls the appropriate tool for each sub-task and collects evidence.
    """
    evidence = []

    for task in state["sub_tasks"]:
        sub_q = task.get("sub_question", state["question"])
        tool = task.get("tool", "vector")

        if tool == "sql":
            result = run_sql_tool(sub_q, model=MODEL_NAME)
            evidence.append({
                "sub_question": sub_q,
                "tool": "sql",
                "content": result.get("rows"),
                "source": f"SQL query: {result.get('sql')}",
                "error": result.get("error"),
            })
        else:
            results = vector_search(sub_q, n_results=2)
            for r in results:
                evidence.append({
                    "sub_question": sub_q,
                    "tool": "vector",
                    "content": r["text"],
                    "source": r["source"],
                    "error": None,
                })

    state["evidence"] = evidence
    return state


def critic_node(state: AgentState) -> AgentState:
    """
    Checks whether the collected evidence is sufficient and relevant.
    Sets critic_feedback if not; leaves it empty if evidence passes.
    """
    evidence_summary = "\n".join(
        f"- [{e['tool']} | {e['source']}]: {str(e['content'])[:1500]}"
        for e in state["evidence"]
    )

    prompt = f"""You are a critical fact-checking agent. A user asked a question, and evidence was gathered to answer it.
Judge whether the evidence below is SUFFICIENT and RELEVANT to fully answer the question.

Question: {state['question']}

Evidence gathered:
{evidence_summary}

Respond ONLY with JSON in this exact format:
{{"sufficient": true or false, "reason": "short explanation"}}

JSON:"""

    response = llm.invoke(prompt)

    try:
        verdict = extract_json(response.content)
        sufficient = bool(verdict.get("sufficient", True))
        reason = verdict.get("reason", "")
    except Exception:
        # If the critic itself fails to respond in valid JSON, default to accepting the evidence
        # rather than looping forever on a weak local model.
        sufficient = True
        reason = ""

    if sufficient:
        state["critic_feedback"] = ""
    else:
        state["critic_feedback"] = reason

    return state


def should_retry(state: AgentState) -> Literal["retry", "proceed"]:
    """Conditional edge: decide whether to loop back to planner or move to synthesis."""
    if state["critic_feedback"] and state["retry_count"] < MAX_RETRIES:
        return "retry"
    return "proceed"


def increment_retry_node(state: AgentState) -> AgentState:
    state["retry_count"] += 1
    return state


def synthesizer_node(state: AgentState) -> AgentState:
    """
    Combines all evidence into one final, cited answer.
    """
    evidence_block = "\n".join(
        f"- [Source: {e['source']}]: {str(e['content'])[:1500]}"
        for e in state["evidence"]
    )

    prompt = f"""You are an enterprise knowledge assistant. Answer the user's question using ONLY the evidence below.
Cite the source (document name or SQL query) for each fact you use. If the evidence is insufficient, say so honestly
rather than guessing.

Question: {state['question']}

Evidence:
{evidence_block}

If the question involves comparing two or more numbers (e.g., an actual value vs. a threshold, or one period vs. another),
you MUST work through the comparison explicitly before concluding:
1. State each number clearly, labeled (e.g., "Actual Q2 2025 refund rate: 2.47%", "Policy threshold: 3%").
2. State the direct numeric comparison (e.g., "2.47% is LESS THAN 3%").
3. Only then state the correct conclusion based on that comparison.
Do not state a conclusion that contradicts the numbers above it — re-check your comparison direction (greater/less than) before finalizing.

CRITICAL RULE ABOUT MULTIPLE NUMBERS IN THE SAME DOCUMENT:
A single evidence chunk or document may contain several different numbers that apply to DIFFERENT, UNRELATED
conditions (e.g., one number for a general rule, a different number for a specific exception case). Before
using any number, quote the exact sentence it comes from to yourself, and confirm that sentence actually
answers the question asked. Do NOT combine a number from one clause with a topic/condition from a different
clause just because they appear in the same chunk or the same document.

Example of what NOT to do: if the evidence says "customers get a refund within 30 days" in one sentence,
and separately says "no substitute offered within 60 days" in an unrelated sentence about a different topic,
and the question asks about the general refund period, the correct answer is 30 days — NOT 60 days, and NOT
a blend of both sentences. Focus on which NUMBER belongs to which CONDITION/TOPIC, based on meaning — not on
matching exact wording or phrases. Read for what each clause actually means, not for literal keyword matches.

CRITICAL RULE ABOUT NUMBERS FROM SQL EVIDENCE:
Numeric values in the evidence (especially from "SQL query" sources) are ALREADY in their final form.
If a SQL result shows a number like 1.63, and the question is about a rate or percentage, report it EXACTLY
as "1.63%" — do NOT multiply, divide, or otherwise convert it again. Copy numeric values from the evidence
character-for-character into your answer. Do not round, rescale, or reformat them.

CRITICAL RULE ABOUT ARITHMETIC:
NEVER perform addition, subtraction, multiplication, or division by hand in your answer — you are unreliable
at multi-step arithmetic and will make errors. If the evidence gives you multiple numbers that need to be
combined (e.g., summed across regions) and no single combined total is provided, say so explicitly
("the evidence provides a breakdown but not a combined total") rather than attempting the sum yourself.

Write a clear, concise final answer with inline citations like (Source: refund_policy.txt) or (Source: SQL query):"""

    response = llm.invoke(prompt)
    state["final_answer"] = response.content
    return state


# ---------- GRAPH ----------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("increment_retry", increment_retry_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "critic")

    graph.add_conditional_edges(
        "critic",
        should_retry,
        {
            "retry": "increment_retry",
            "proceed": "synthesizer",
        },
    )
    graph.add_edge("increment_retry", "planner")
    graph.add_edge("synthesizer", END)

    return graph.compile()


def ask(question: str) -> dict:
    """
    Main entry point: runs the full agent graph on a question.
    Returns the final state, including the answer and the evidence trail.
    """
    app = build_graph()
    initial_state: AgentState = {
        "question": question,
        "sub_tasks": [],
        "evidence": [],
        "critic_feedback": "",
        "retry_count": 0,
        "final_answer": "",
    }
    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    test_questions = [
        "What is our refund cooling-off period for enterprise customers?",
        "How does our Q2 2025 refund rate compare to our stated refund policy threshold?",
    ]

    for q in test_questions:
        print("=" * 80)
        print(f"QUESTION: {q}")
        print("=" * 80)

        result = ask(q)

        print(f"\nSub-tasks planned: {result['sub_tasks']}")
        print(f"Retries used: {result['retry_count']}")
        print(f"\nFINAL ANSWER:\n{result['final_answer']}\n")
