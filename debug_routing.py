"""
Diagnostic: checks exactly what sub-tasks the planner generates for a pure-SQL question,
to investigate whether it's unnecessarily also triggering a vector search
(suspected from the evaluation run's Q7 faithfulness failure).
"""

from src.agents.graph import ask

question = "What was the total contract value across all regions in 2025-Q1?"
result = ask(question)

print("QUESTION:", question)
print("\nSUB-TASKS PLANNED:")
for task in result["sub_tasks"]:
    print(f"  - tool={task.get('tool')} | sub_question={task.get('sub_question')}")

print("\nEVIDENCE COLLECTED:")
for e in result["evidence"]:
    print(f"  - [{e.get('tool')}] source={e.get('source')}")
    print(f"    content preview: {str(e.get('content'))[:150]}")

print("\nFINAL ANSWER:")
print(result["final_answer"])
