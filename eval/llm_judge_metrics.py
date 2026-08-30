"""
Lightweight LLM-as-judge evaluation metrics, replicating RAGAS's core methodology
without RAGAS's heavy multi-provider dependency chain (which had a version conflict
with langchain-community's vertexai module in this environment).

Metrics (same definitions RAGAS uses):
- faithfulness: what fraction of the answer's claims are actually supported by the
  retrieved evidence (contexts)? Catches hallucination.
- answer_relevancy: does the answer actually address the question asked, without
  padding or going off-topic?

Both are scored 0.0-1.0 by asking the local LLM to judge, with a strict JSON response format.
"""

import json
import re
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "llama-3.3-70b-versatile"
judge_llm = ChatGroq(model=JUDGE_MODEL, temperature=0, api_key=os.getenv("GROQ_API_KEY"))


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def score_faithfulness(answer: str, contexts: list[str]) -> dict:
    """
    Judges whether the answer's claims are grounded in the provided evidence (contexts).
    Returns {"score": float 0-1, "reason": str}
    """
    context_block = "\n---\n".join(contexts)

    prompt = f"""You are a strict fact-checking judge. Compare the ANSWER below against the EVIDENCE.
Determine what fraction of the factual claims in the answer are actually supported by the evidence.
A score of 1.0 means every claim is fully supported. A score of 0.0 means the answer is entirely
unsupported or contradicts the evidence. Partial support should get a proportional score (e.g. 0.5, 0.75).

EVIDENCE:
{context_block}

ANSWER:
{answer}

Respond ONLY with JSON in this exact format:
{{"score": 0.0, "reason": "short explanation"}}

JSON:"""

    response = judge_llm.invoke(prompt)
    try:
        result = _extract_json(response.content)
        score = float(result.get("score", 0.5))
        score = max(0.0, min(1.0, score))  # clamp to valid range
        reason = result.get("reason", "")
    except Exception:
        score, reason = 0.5, "Judge response could not be parsed; defaulted to neutral score."

    return {"score": score, "reason": reason}


def score_answer_relevancy(question: str, answer: str) -> dict:
    """
    Judges whether the answer actually addresses the question asked.
    Returns {"score": float 0-1, "reason": str}
    """
    prompt = f"""You are a judge evaluating answer relevancy. Determine how directly and completely
the ANSWER addresses the QUESTION asked. A score of 1.0 means the answer fully and directly addresses
the question with no irrelevant padding. A score of 0.0 means the answer is off-topic or doesn't
address the question at all.

QUESTION:
{question}

ANSWER:
{answer}

Respond ONLY with JSON in this exact format:
{{"score": 0.0, "reason": "short explanation"}}

JSON:"""

    response = judge_llm.invoke(prompt)
    try:
        result = _extract_json(response.content)
        score = float(result.get("score", 0.5))
        score = max(0.0, min(1.0, score))
        reason = result.get("reason", "")
    except Exception:
        score, reason = 0.5, "Judge response could not be parsed; defaulted to neutral score."

    return {"score": score, "reason": reason}
