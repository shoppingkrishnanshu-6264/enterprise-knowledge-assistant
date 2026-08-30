# Enterprise Knowledge Assistant using Agentic RAG

An internal AI assistant that answers employee questions using company documents (HR/finance policies) and structured sales data. Built with a **local, free stack** (Ollama + llama3.1, ChromaDB, LangGraph, Streamlit) — no API costs during development.

Unlike basic "retrieve-then-generate" RAG, this system is **agentic**: it plans multi-step retrieval, routes sub-questions to the right tool (document search vs. SQL), critiques its own evidence before answering, and cites sources.

---

## 1. Architecture

```
User (Streamlit chat UI)
        │
        ▼
┌─────────────────────────────────────────────┐
│         LangGraph Agent Orchestrator          │
│                                               │
│  PLANNER → EXECUTOR → CRITIC → SYNTHESIZER   │
│               ▲___________|                  │
│           (retry loop, max 2x)               │
└─────────────────────────────────────────────┘
        │                        │
        ▼                        ▼
  Vector Search Tool        SQL Tool
  (ChromaDB +                (SQLite,
  sentence-transformers)    LLM-generated SQL)
        │                        │
        ▼                        ▼
  data/docs/*.txt          data/structured/sales.db
  (5 policy documents)      (511 synthetic sales records)
```

**Nodes:**
- **Planner** — breaks the question into 1-3 sub-questions, each tagged with a tool (`vector` or `sql`)
- **Executor** — calls the assigned tool for each sub-question, collects evidence
- **Critic** — judges whether the collected evidence is sufficient; if not, sends feedback back to the Planner (up to 2 retries)
- **Synthesizer** — writes the final answer using only the verified evidence, with inline citations

---

## 2. Tech Stack (as actually built)

| Component | Tool used |
|---|---|
| LLM (agent reasoning + judge) | Ollama, running `llama3.1` (8B) locally — free, no API key |
| Agent orchestration | LangGraph |
| Vector DB | ChromaDB (persisted locally in `chroma_db/`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`), local |
| Structured DB | SQLite (`data/structured/sales.db`) |
| Frontend | Streamlit |
| Evaluation | Custom LLM-as-judge (faithfulness + answer relevancy), replicating RAGAS methodology without RAGAS's dependency chain (see Section 5) |

---

## 3. Project Structure

```
enterprise-knowledge-assistant/
├── data/
│   ├── docs/                  # 5 synthetic policy .txt files
│   └── structured/            # sales_data.csv + sales.db
├── chroma_db/                 # persisted vector store (generated)
├── src/
│   ├── ingestion/
│   │   ├── ingest_docs.py     # chunk + embed + store docs in ChromaDB
│   │   └── test_retrieval.py  # sanity-check vector search
│   ├── db/
│   │   └── load_sales_db.py   # load CSV into SQLite
│   ├── tools/
│   │   ├── vector_tool.py     # vector_search() function
│   │   └── sql_tool.py        # NL -> SQL -> results, with safety checks
│   └── agents/
│       └── graph.py           # LangGraph agent: Planner/Executor/Critic/Synthesizer
├── eval/
│   ├── eval_dataset.py        # 12 ground-truth Q&A pairs, 3 categories
│   ├── llm_judge_metrics.py   # faithfulness + answer_relevancy scoring
│   ├── run_eval.py            # runs full eval, saves eval/results.csv
│   └── results.csv            # generated after running eval
├── app.py                     # Streamlit chat UI
├── requirements.txt
└── README.md
```

---

## 4. How to Run

```bash
# 1. Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Pull the local model (one-time)
ollama pull llama3.1

# 3. Ingest documents into ChromaDB
python3 src/ingestion/ingest_docs.py

# 4. Load sales data into SQLite
python3 src/db/load_sales_db.py

# 5. Run the agent directly (optional sanity check)
python3 -m src.agents.graph

# 6. Launch the chat UI
python3 -m streamlit run app.py

# 7. Run the evaluation suite
python3 -m eval.run_eval
```

---

## 5. Evaluation Methodology

**Note on RAGAS**: the `ragas` Python package was originally planned, but its dependency chain conflicted with `langchain-community` in this environment (`No module named 'langchain_community.chat_models.vertexai'`). Rather than fight version-pinning across a fast-moving ecosystem, evaluation was implemented as a lightweight, dependency-free **LLM-as-judge** module (`eval/llm_judge_metrics.py`) using the same local model, replicating RAGAS's two core metrics:

- **Faithfulness** — what fraction of the answer's claims are actually supported by the retrieved evidence? (catches hallucination)
- **Answer Relevancy** — does the answer actually address the question asked?

**Evaluation set**: 12 questions across 3 categories — `policy_only` (vector search), `sql_only` (structured data), `multi_hop` (both tools — the core "agentic" showcase).

### Bugs found and fixed during evaluation

Running the evaluation surfaced three concrete, reproducible bugs in the agent's reasoning — a direct benefit of building a systematic eval harness rather than only spot-checking a few queries manually.

| # | Bug | Example | Root Cause | Fix |
|---|---|---|---|---|
| 1 | Backwards numeric comparison | Agent said 2.47% "exceeds" a 3% threshold | Synthesizer jumped to a conclusion without explicitly stating the comparison direction | Rewrote synthesizer prompt to force explicit step-by-step comparison before concluding |
| 2 | 100x unit conversion error | Agent reported a refund rate of "163.106%" when the correct value was 1.63% | SQL tool returned an unrounded float (`1.6310603...`); synthesizer re-converted an already-final percentage | (a) SQL tool now rounds floats to 2 decimals at the source; (b) synthesizer prompt forbids re-converting numbers already in evidence |
| 3 | Hallucinated irrelevant sub-question + unreliable manual arithmetic | For "total contract value across all regions," Planner invented an unrelated policy sub-question, and the Synthesizer manually summed 4 numbers incorrectly (off by $2,000,000) | Planner prompt didn't constrain tool selection for purely numeric questions; SQL used `GROUP BY`, forcing error-prone manual addition downstream | (a) Planner prompt now explicitly restricts "vector" tool use to policy-related questions only, with examples; (b) SQL prompt now generates a single aggregate query (no `GROUP BY`) for "total across all X" questions, avoiding manual addition entirely |

### Results: before vs. after fixes

| Metric | Before fixes | After fixes (avg across 2 re-runs) |
|---|---|---|
| Overall Faithfulness | 0.646 | ~0.70 |
| **Multi-hop Faithfulness** (key metric) | **0.50** | **~0.79** |
| Overall Answer Relevancy | 0.917 | ~0.82 |

The most important result is the **multi-hop faithfulness improvement (0.50 → ~0.79)**, since multi-hop questions (combining SQL + document evidence) are what distinguish this system from basic RAG. This is exactly the category the bugs above were found in.

### Known limitation: LLM-as-judge variance

Running the same evaluation set multiple times after the fixes produced overall faithfulness scores ranging from 0.688 to 0.708, and answer relevancy from 0.767 to 0.867, on identical questions and (mostly) identical agent code. This is attributable to **non-determinism in using an 8B local model as an evaluation judge** — even at `temperature=0`, small models running on limited hardware show run-to-run variance in judgment, and occasionally penalize technically-correct answers on formatting grounds (e.g., docking faithfulness for including a citation). This is documented here rather than hidden, as an honest limitation of local-only evaluation; a production system would benefit from a stronger, more consistent judge model (e.g., Claude or GPT-4) for evaluation specifically, even if the production agent itself runs locally.

---

## 6. Known Limitations & Future Improvements

- **Local model reasoning ceiling**: llama3.1 (8B) is capable of correct multi-hop reasoning when prompts are scaffolded explicitly (as shown by the fixes above), but is unreliable at free-form arithmetic and can hallucinate plausible-sounding sub-questions without tight prompt constraints.
- **Recommended production upgrade**: swap the local Ollama model for Claude (via Anthropic API) for the Synthesizer and Critic nodes specifically, where reasoning quality matters most, while keeping ingestion/embeddings local for cost efficiency.
- **Evaluation judge**: as noted above, a stronger model as judge would reduce scoring variance and give more trustworthy evaluation numbers.
- **Retrieval quality**: currently uses fixed-size character chunking (800 chars, 150 overlap); semantic chunking or a re-ranker (e.g., Cohere Rerank) could improve precision on ambiguous queries.
- **Not yet implemented** (noted in original scope, deferred due to time): role-based access control / permission-aware retrieval, hybrid search (BM25 + vector), web search tool for external/non-internal questions.

---

## 7. Demo Queries

These showcase the agentic behavior most clearly:

1. **Multi-hop (SQL + vector)**: *"How does our Q2 2025 refund rate compare to our stated refund policy threshold?"*
2. **Pure SQL, single aggregate**: *"What was the total contract value across all regions in 2025-Q1?"*
3. **Pure policy lookup**: *"What is the home office stipend for remote employees?"*
