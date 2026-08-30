"""
SQL Tool for the Enterprise Knowledge Assistant.

Given a natural-language question, this tool:
1. Sends the question + table schema to the local LLM (Ollama/llama3.1)
2. Asks the LLM to generate a SQL query (SELECT-only, for safety)
3. Executes the query against the SQLite sales database
4. Returns the result rows as a string the agent can use to answer the user

This is a standalone, testable module — the agent orchestrator will import
and call `run_sql_tool(question)` from here later.
"""

import sqlite3
import re
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "data/structured/sales.db"
TABLE_NAME = "sales"

# Schema description given to the LLM so it knows what columns exist.
# Keeping this explicit (rather than dumping raw DB schema) gives more reliable SQL generation
# from smaller local models like llama3.1.
SCHEMA_DESCRIPTION = """
Table name: sales

Columns:
- order_id (INTEGER): unique order identifier
- quarter (TEXT): fiscal quarter, format like '2025-Q1', '2025-Q2', ..., '2026-Q2'
- region (TEXT): one of 'North America', 'EMEA', 'APAC', 'LATAM'
- product (TEXT): one of 'Northwind CRM', 'Northwind Analytics', 'Northwind Support Desk', 'Northwind Billing Suite'
- customer_tier (TEXT): one of 'Business', 'Enterprise'
- contract_value_usd (REAL): total contract value in USD
- refunded (TEXT): 'True' or 'False' (whether this order had any refund)
- refund_amount_usd (REAL): amount refunded in USD (0.0 if not refunded)
- sales_rep (TEXT): name of the sales representative
"""

SQL_GENERATION_PROMPT = """You are a SQL expert. Given the table schema below and a user question,
write a single valid SQLite SELECT query that answers the question.

{schema}

Rules:
- Only write SELECT queries. Never write INSERT, UPDATE, DELETE, or DROP.
- Return ONLY the raw SQL query. No explanation, no markdown formatting, no backticks.
- If calculating a refund rate, use: SUM(refund_amount_usd) / SUM(contract_value_usd) * 100
- If the question asks for a single TOTAL "across all" regions/products/quarters (e.g. "total contract value
  across all regions"), write ONE query that returns ONE aggregate number (a single SUM with no GROUP BY).
  Do NOT use GROUP BY for these questions — a breakdown forces manual addition later, which is error-prone.
  Only use GROUP BY when the user explicitly asks for a breakdown "by region" / "by product" / "per quarter".

User question: {question}

SQL query:"""


def clean_sql(raw_sql: str) -> str:
    """
    Strips markdown code fences and extra whitespace/text the LLM might add,
    since local models don't always follow 'no formatting' instructions perfectly.
    """
    sql = raw_sql.strip()
    sql = re.sub(r"^```sql", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"^```", "", sql).strip()
    sql = re.sub(r"```$", "", sql).strip()
    return sql


def is_safe_select(sql: str) -> bool:
    """
    Basic safety guard: only allow SELECT statements.
    This blocks the LLM (or a malicious prompt) from generating destructive SQL.
    """
    normalized = sql.strip().lower()
    forbidden = ["insert", "update", "delete", "drop", "alter", "create", "attach", "pragma"]
    if not normalized.startswith("select"):
        return False
    if any(word in normalized for word in forbidden):
        return False
    return True


def generate_sql(question: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Uses the Groq-hosted LLM to turn a natural-language question into SQL."""
    llm = ChatGroq(model=model, temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    prompt = SQL_GENERATION_PROMPT.format(schema=SCHEMA_DESCRIPTION, question=question)
    response = llm.invoke(prompt)
    return clean_sql(response.content)


def execute_sql(sql: str) -> list[dict]:
    """Runs the SQL query against the SQLite database and returns rows as dicts.
    Numeric values are rounded to 2 decimal places to avoid long floats (e.g. 1.6310603001297017)
    confusing the downstream LLM when it writes the final answer."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = [dict(row) for row in cursor.fetchall()]

    for row in rows:
        for key, value in row.items():
            if isinstance(value, float):
                row[key] = round(value, 2)

    conn.close()
    return rows


def run_sql_tool(question: str, model: str = "llama-3.3-70b-versatile") -> dict:
    """
    Main entry point for the SQL tool.
    Returns a dict with the generated SQL, the result rows, and any error.
    """
    sql = generate_sql(question, model=model)

    if not is_safe_select(sql):
        return {
            "question": question,
            "sql": sql,
            "error": "Generated SQL failed safety check (must be a single SELECT statement).",
            "rows": None,
        }

    try:
        rows = execute_sql(sql)
        return {
            "question": question,
            "sql": sql,
            "error": None,
            "rows": rows,
        }
    except Exception as e:
        return {
            "question": question,
            "sql": sql,
            "error": str(e),
            "rows": None,
        }


if __name__ == "__main__":
    # Quick manual test
    test_questions = [
        "What was the total contract value in Q2 2025?",
        "What is the refund rate for Q2 2025?",
        "Which region had the highest contract value in 2025-Q4?",
    ]

    for q in test_questions:
        print("=" * 70)
        print(f"QUESTION: {q}")
        result = run_sql_tool(q)
        print(f"GENERATED SQL: {result['sql']}")
        if result["error"]:
            print(f"ERROR: {result['error']}")
        else:
            print(f"RESULT: {result['rows']}")
        print()
