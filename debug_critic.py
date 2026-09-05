"""
Diagnostic: runs ONLY the planner -> executor -> critic sequence once (no retry loop),
to see exactly what evidence was gathered on the FIRST attempt and what the Critic
decided about it. This isolates whether the Critic is incorrectly rejecting good evidence.
"""

from src.agents.graph import planner_node, executor_node, critic_node

state = {
    "question": "What is our refund cooling-off period for enterprise customers?",
    "sub_tasks": [],
    "evidence": [],
    "critic_feedback": "",
    "retry_count": 0,
    "final_answer": "",
}

state = planner_node(state)
print("SUB-TASKS:", state["sub_tasks"])
print()

state = executor_node(state)
print("EVIDENCE GATHERED ON FIRST ATTEMPT:")
for e in state["evidence"]:
    print(f"  [{e['tool']}] source={e['source']}")
    print(f"  content: {str(e['content'])[:400]}")
    print()

state = critic_node(state)
print("CRITIC VERDICT:")
print(f"  critic_feedback (empty = accepted): '{state['critic_feedback']}'")
