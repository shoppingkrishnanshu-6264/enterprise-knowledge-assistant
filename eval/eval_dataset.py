"""
Ground-truth evaluation set for the Enterprise Knowledge Assistant.

Each entry has:
- question: what we ask the agent
- ground_truth: the correct answer (used for reference; RAGAS' faithfulness/answer_relevancy
  metrics don't strictly require this, but it's included for manual review and for metrics
  that do use it, like context_recall, if you add them later)
- category: helps you see performance broken down by question type in the results
"""

EVAL_QUESTIONS = [
    # --- Policy-only (vector search) questions ---
    {
        "question": "What is the refund cooling-off period for enterprise customers?",
        "ground_truth": "Enterprise customers are eligible for a full refund if they cancel in writing within 30 days of the initial contract signing date.",
        "category": "policy_only",
    },
    {
        "question": "How many days of paid sick leave do employees get per year?",
        "ground_truth": "Employees are entitled to 12 days of paid sick leave per year.",
        "category": "policy_only",
    },
    {
        "question": "What is the one-time home office setup stipend for fully remote employees?",
        "ground_truth": "Northwind Technologies provides a one-time home office setup stipend of $500 for employees approved for fully remote work.",
        "category": "policy_only",
    },
    {
        "question": "How many weeks of paid maternity leave are eligible employees entitled to?",
        "ground_truth": "Eligible employees are entitled to 26 weeks of paid maternity leave.",
        "category": "policy_only",
    },
    {
        "question": "What is the daily meal allowance cap while traveling for business?",
        "ground_truth": "The daily meal allowance while traveling is capped at $75/day.",
        "category": "policy_only",
    },
    {
        "question": "Within how many days must an expense report be submitted?",
        "ground_truth": "Expense reports must be submitted within 30 days of the expense being incurred.",
        "category": "policy_only",
    },

    # --- SQL-only (structured data) questions ---
    {
        "question": "What was the total contract value across all regions in 2025-Q1?",
        "ground_truth": "This requires summing contract_value_usd for quarter = '2025-Q1' in the sales database.",
        "category": "sql_only",
    },
    {
        "question": "Which product had the highest total contract value in 2025-Q3?",
        "ground_truth": "This requires grouping by product for quarter = '2025-Q3' and finding the max total contract_value_usd.",
        "category": "sql_only",
    },
    {
        "question": "What is the overall refund rate across all quarters?",
        "ground_truth": "This requires SUM(refund_amount_usd) / SUM(contract_value_usd) * 100 across the entire sales table.",
        "category": "sql_only",
    },

    # --- Multi-hop (both tools, the "agentic" showcase questions) ---
    {
        "question": "How does our Q2 2025 refund rate compare to our stated refund policy threshold?",
        "ground_truth": "The Q2 2025 refund rate (approximately 2.47%) is below the internal refund rate threshold of 3% stated in the refund policy.",
        "category": "multi_hop",
    },
    {
        "question": "Is our 2025-Q4 refund rate within the threshold defined in the refund policy?",
        "ground_truth": "This requires comparing the calculated 2025-Q4 refund rate against the 3% threshold stated in refund_policy.txt.",
        "category": "multi_hop",
    },
    {
        "question": "If an enterprise customer's refund exceeds our internal quarterly threshold, what does policy say should happen, and was that threshold exceeded in 2026-Q1?",
        "ground_truth": "Policy states refund rates exceeding 3% per quarter trigger an internal review by Finance and Customer Success leadership; this requires checking whether 2026-Q1's actual refund rate exceeded 3%.",
        "category": "multi_hop",
    },
]
