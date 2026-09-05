"""
Diagnostic: checks exactly what vector search evidence was retrieved for the
refund cooling-off period question, to investigate why the agent answered
"60 days" instead of the correct "30 days".
"""

from src.tools.vector_tool import vector_search

query = "What is the refund cooling-off period for enterprise customers?"
results = vector_search(query, n_results=3)

for i, r in enumerate(results):
    print(f"--- Result {i+1} (source: {r['source']}, distance: {r['distance']:.4f}) ---")
    print(r["text"])
    print()
