import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.graph import handle_query

queries = [
    "tính 25 * 4 + 10",
    "bây giờ là mấy giờ",
    "xin chào bạn",
    "langgraph là gì",
]

for q in queries:
    print(f"\n{'='*50}\nQUERY: {q}\n{'='*50}")
    res = handle_query(session_id="test_quick", user_query=q)
    print("Intent:", res.get("intent"))
    print("Confidence:", res.get("confidence_score"))
    print("Final answer snippet:\n", res.get("final_answer", "")[:250])
    print("\nTimings:")
    for t in res.get("agent_timings", []):
        print(f" - {t['agent']}: {t['duration']:.4f}s (called by: {t['called_by']})")

