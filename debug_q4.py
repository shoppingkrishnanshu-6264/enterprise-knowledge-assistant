"""
Debug script: isolates exactly what SQL is generated for the Q4 2025 refund rate question,
to diagnose the 100x error found during evaluation (163.106% instead of the correct 1.631%).
"""

from src.tools.sql_tool import run_sql_tool

question = "What is the refund rate for 2025-Q4?"
result = run_sql_tool(question)

print("QUESTION:", question)
print("GENERATED SQL:", result["sql"])
print("ERROR:", result["error"])
print("RAW ROWS:", result["rows"])
